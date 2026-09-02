"""H5 snippet dataset (see h5_format.md).

The h5 file is opened lazily per worker process: only the path is stored at
construction time, and the file handle is created on first __getitem__ call in
whichever process ends up using the dataset. This is required for
DataLoader(num_workers > 0) on Windows (spawn start method), where an open
h5py.File handle cannot be pickled into workers.

Small per-sample arrays (labels, split, gt) are read eagerly into memory;
only image reads go through the lazy handle.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from torchvision.transforms import v2

RESNET_INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST = 0, 1, 2
SPLIT_NAMES = {SPLIT_TRAIN: "train", SPLIT_VAL: "validate", SPLIT_TEST: "test"}

HARD_NEGATIVE_NAME = "hard_negative"

# Allowed image dtypes and their expected pixel ranges (h5_format.md):
#   uint8 0-255 · uint16 0-65535 · float16/float32 already scaled to [0, 1]
IMAGE_DTYPES = {"uint8", "uint16", "float16", "float32"}


def to_model_input(arr: np.ndarray) -> np.ndarray:
    """Storage dtype -> what the transform pipeline consumes. uint8 stays
    uint8 (ToDtype scales it by 255 later; augmentations see the classic
    0-255 crop); every other allowed dtype becomes float32 in [0, 1]
    (uint16 divided by 65535, float16 widened, float32 as-is)."""
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    return arr.astype(np.float32, copy=False)


def to_display_uint8(arr: np.ndarray) -> np.ndarray:
    """Storage dtype -> uint8 for thumbnails and previews (curate GUI,
    report sample grids), regardless of how the file stores pixels."""
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr // 257).astype(np.uint8)
    return (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


# Training-only augmentation catalog (opt-in via --augment). Never applied
# to validation, evaluation, or inference. Every entry is parameterizable:
# CLI spec "name:key=val,key=val" (ranges as lo-hi, e.g. sigma=0.1-1.5) or a
# config-JSON object {"name": ..., key: val, ...}. Each factory's keyword
# defaults ARE the accepted parameters and their conservative defaults; a
# probability p wraps the transform in random-apply (p=1 applies always).
# CAUTION: the photometric ops (colorjitter, invert, solarize, equalize,
# autocontrast, posterize, grayscale) alter intensity/color — if your
# classes are distinguished by intensity, they can destroy the label. The
# geometric ops (rotation, perspective), blur/sharpness, and erasing are
# the safer subset for intensity-coded classes.
def _maybe(p, transform):
    return transform if p >= 1 else v2.RandomApply([transform], p=p)


class _Solarize(torch.nn.Module):
    """Solarize with a dtype-aware default threshold: 128 for uint8 crops,
    0.5 for float crops ([0,1] storage). An explicit threshold is used
    as-is - pass it in the crop's own scale."""

    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def forward(self, img):
        thr = self.threshold
        if thr is None:
            thr = 128 if img.dtype == torch.uint8 else 0.5
        return v2.functional.solarize(img, thr)


AUGMENTATIONS = {
    "grayscale": lambda p=0.2: v2.RandomGrayscale(p=p),
    "colorjitter": lambda p=0.5, brightness=0.2, contrast=0.2,
        saturation=0.2, hue=0.05:
        _maybe(p, v2.ColorJitter(brightness, contrast, saturation, hue)),
    "gaussianblur": lambda p=0.5, kernel=3, sigma=(0.1, 1.5):
        _maybe(p, v2.GaussianBlur(int(kernel), sigma)),
    "perspective": lambda p=0.3, distortion=0.3:
        v2.RandomPerspective(distortion_scale=distortion, p=p),
    "rotation": lambda p=0.5, degrees=15:
        _maybe(p, v2.RandomRotation(degrees)),
    "invert": lambda p=0.3: v2.RandomInvert(p=p),
    "posterize": lambda p=0.3, bits=4: v2.RandomPosterize(int(bits), p=p),
    "solarize": lambda p=0.3, threshold=None:
        _maybe(p, _Solarize(threshold)),
    "sharpness": lambda p=0.3, factor=2.0:
        v2.RandomAdjustSharpness(factor, p=p),
    "autocontrast": lambda p=0.3: v2.RandomAutocontrast(p=p),
    "equalize": lambda p=0.3: v2.RandomEqualize(p=p),
    # random black rectangle (Cutout / Random Erasing); applied AFTER
    # resize/normalize so the erased fraction is consistent at model scale
    "erasing": lambda p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value=0.0:
        v2.RandomErasing(p=p, scale=scale, ratio=ratio, value=value),
    # policy-based (tuned on natural-image benchmarks; include the
    # intensity-altering ops above — same caution applies, amplified)
    "autoaugment": lambda: v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET),
    "randaugment": lambda num_ops=2, magnitude=9:
        v2.RandAugment(num_ops=int(num_ops), magnitude=int(magnitude)),
    "trivialaugment": lambda: v2.TrivialAugmentWide(),
}

# entries applied after the resize/normalize transform (float tensors);
# everything else runs before it, on the raw uint8 crop
POST_RESIZE = {"erasing"}

_LOADED_PLUGINS: set[str] = set()


def load_augmentation_plugins(paths) -> list[str]:
    """Merge user augmentation modules into the catalog (--augment-plugin).

    Each file is executed as a module and must define AUGMENTATIONS
    ({name: factory}) using the same contract as the built-ins; an optional
    POST_RESIZE set names entries that run after resize/normalize. Names
    must not collide with built-ins or other plugins. Idempotent per file
    (an Optuna study loads plugins once, not once per trial). Returns the
    catalog names the given files provide."""
    import importlib.util

    added: list[str] = []
    for i, path in enumerate(paths or []):
        p = Path(path)
        if not p.is_file():
            raise SystemExit(f"--augment-plugin: {p} not found")
        key = str(p.resolve())
        if key in _LOADED_PLUGINS:
            continue
        spec = importlib.util.spec_from_file_location(
            f"_mic_augment_plugin_{i}_{p.stem}", p)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception as e:
            raise SystemExit(f"--augment-plugin: {p} failed to import: {e}")
        catalog = getattr(mod, "AUGMENTATIONS", None)
        if not isinstance(catalog, dict) or not catalog:
            raise SystemExit(f"--augment-plugin: {p} must define a non-empty "
                             "AUGMENTATIONS dict of {name: factory}")
        post = set(getattr(mod, "POST_RESIZE", ()))
        for name in post - set(catalog):
            raise SystemExit(f"--augment-plugin: {p}: POST_RESIZE names "
                             f"'{name}', which its AUGMENTATIONS does not "
                             "define")
        for name, factory in catalog.items():
            if not callable(factory):
                raise SystemExit(f"--augment-plugin: {p}: '{name}' must map "
                                 "to a factory callable")
            if name in AUGMENTATIONS:
                raise SystemExit(f"--augment-plugin: {p}: '{name}' already "
                                 "exists (built-in or another plugin) - "
                                 "pick a distinct name")
        for name, factory in catalog.items():
            AUGMENTATIONS[name] = factory
            added.append(name)
        POST_RESIZE.update(post)
        _LOADED_PLUGINS.add(key)
    return added


def _parse_value(v: str):
    try:
        return float(v)
    except ValueError:
        pass
    parts = v.split("-")
    if len(parts) == 2:
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    return v  # leave as string; the factory will reject it if invalid


def parse_augment_spec(spec) -> tuple[str, dict]:
    """'name' / 'name:k=v,k=v' / {'name': ..., k: v} -> (name, params)."""
    if isinstance(spec, dict):
        params = dict(spec)
        name = params.pop("name", None)
        if not name:
            raise ValueError(f"augmentation object needs a 'name' key: {spec}")
        return str(name), params
    text = str(spec)
    name, _, rest = text.partition(":")
    params = {}
    if rest:
        for pair in rest.split(","):
            key, sep, value = pair.partition("=")
            if not sep:
                raise ValueError(f"augmentation '{text}': expected key=value, "
                                 f"got '{pair}'")
            params[key.strip()] = _parse_value(value.strip())
    return name.strip(), params


def _instantiate(name: str, params: dict):
    factory = AUGMENTATIONS.get(name)
    if factory is None:
        raise ValueError(f"unknown augmentation '{name}'; "
                         f"choose from {sorted(AUGMENTATIONS)}")
    import inspect
    allowed = set(inspect.signature(factory).parameters)
    unknown = set(params) - allowed
    if unknown:
        raise ValueError(
            f"augmentation '{name}': unknown parameter(s) {sorted(unknown)}; "
            f"accepts {sorted(allowed) if allowed else 'no parameters'}")
    params = {k: tuple(v) if isinstance(v, list) else v
              for k, v in params.items()}
    return factory(**params)


def build_augmentation(specs) -> tuple[v2.Compose | None, v2.Compose | None]:
    """(pre_resize, post_resize) pipelines from a list of augmentation specs,
    each stage preserving the order given; (None, None) if empty."""
    if not specs:
        return None, None
    pre, post = [], []
    for spec in specs:
        name, params = parse_augment_spec(spec)
        (post if name in POST_RESIZE else pre).append(_instantiate(name, params))
    return (v2.Compose(pre) if pre else None,
            v2.Compose(post) if post else None)


def build_transform(imagenet_norm: bool, resize: bool = True) -> v2.Compose:
    """The model input pipeline: CHW uint8 -> resize 224 -> float [0,1] ->
    optional ImageNet normalization. Shared by training, evaluation, and
    sliding-window inference so preprocessing can never diverge.

    resize=False skips the resize op for sources already at model size
    (e.g. files pre-resized by optimize_h5.py)."""
    ops = []
    if resize:
        ops.append(v2.Resize(
            (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ))
    ops.append(v2.ToDtype(torch.float32, scale=True))  # uint8 -> [0, 1]
    if imagenet_norm:
        ops.append(v2.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return v2.Compose(ops)


def validate_h5(path: str) -> dict:
    """Sanity-check an h5 file against h5_format.md.

    Returns a summary dict: classes, hard-negative index, and per-split
    genuine / hard-negative counts. Raises ValueError on format violations.
    """
    with h5py.File(path, "r") as f:
        for name in ("images", "labels", "gt", "split", "classes"):
            if name not in f:
                raise ValueError(f"{path}: missing dataset '{name}'")

        n = f["images"].shape[0]
        for name in ("labels", "gt", "split"):
            if f[name].shape[0] != n:
                raise ValueError(
                    f"{path}: '{name}' length {f[name].shape[0]} != images length {n}"
                )

        if f["images"].ndim != 4 or f["images"].shape[3] not in (1, 3):
            raise ValueError(
                f"{path}: images must be (N, H, W, C) with C in (1, 3), "
                f"got shape {f['images'].shape}"
            )

        dt = f["images"].dtype
        if dt.name not in IMAGE_DTYPES:
            raise ValueError(
                f"{path}: images dtype {dt} unsupported; allowed: "
                + ", ".join(sorted(IMAGE_DTYPES))
                + " (uint8 0-255, uint16 0-65535, float16/float32 in [0, 1])")
        if dt.kind == "f" and n:
            sample = f["images"][:: max(1, n // 64)]
            lo, hi = float(sample.min()), float(sample.max())
            if lo < -0.01 or hi > 1.01:
                raise ValueError(
                    f"{path}: float images must be scaled to [0, 1]; a "
                    f"sample spans [{lo:.4g}, {hi:.4g}]")

        classes = f["classes"].asstr()[:]
        if classes[-1] != HARD_NEGATIVE_NAME:
            raise ValueError(
                f"{path}: last class must be '{HARD_NEGATIVE_NAME}', got '{classes[-1]}'"
            )
        hn_index = len(classes) - 1

        labels = f["labels"][:]
        gt = f["gt"][:].astype(bool)
        split = f["split"][:]
        removed = np.zeros(n, dtype=bool)
        if "removed" in f:  # optional curation mask (curate.py)
            if f["removed"].shape[0] != n:
                raise ValueError(f"{path}: 'removed' length "
                                 f"{f['removed'].shape[0]} != images length {n}")
            removed = f["removed"][:].astype(bool)

        if labels.max(initial=0) >= len(classes):
            raise ValueError(f"{path}: label index out of range of classes")
        if not np.isin(split, (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)).all():
            raise ValueError(f"{path}: split contains values outside {{0, 1, 2}}")

        # gt is redundant with labels == hn_index; assert they agree.
        if not np.array_equal(~gt, labels == hn_index):
            raise ValueError(
                f"{path}: gt flag inconsistent with '{HARD_NEGATIVE_NAME}' labels"
            )

        counts = {}
        for value, name in SPLIT_NAMES.items():
            mask = (split == value) & ~removed
            counts[name] = {
                "genuine": int((mask & gt).sum()),
                "hard_negative": int((mask & ~gt).sum()),
                "removed": int(((split == value) & removed).sum()),
            }

    return {"classes": list(classes), "hard_negative_index": hn_index,
            "counts": counts, "dtype": dt.name}


class H5SnippetDataset(Dataset):
    """One split of an h5 snippet file, ready for a ResNet.

    __getitem__ returns (image, label, index) where index is the position
    within this split — used by the mining tracker to attribute per-sample
    errors back to dataset entries.

    Transform pipeline: uint8 HWC -> CHW tensor -> resize 224 -> float [0,1]
    -> optional ImageNet normalization. Grayscale is repeated to 3 channels.
    """

    def __init__(self, h5_path: str, split: int, imagenet_norm: bool = False,
                 augment=None):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {list(SPLIT_NAMES)}, got {split}")
        self.h5_path = h5_path
        self.split = split
        self.imagenet_norm = imagenet_norm
        self.augment_pre, self.augment_post = build_augmentation(augment)
        self._file: h5py.File | None = None  # opened lazily per worker

        with h5py.File(h5_path, "r") as f:
            self.classes = list(f["classes"].asstr()[:])
            all_split = f["split"][:]
            keep = all_split == split
            if "removed" in f:  # curated-out snippets leave every split
                keep &= ~f["removed"][:].astype(bool)
            self.indices = np.flatnonzero(keep)  # h5 row per sample
            self.labels = f["labels"][:][self.indices].astype(np.int64)
            self.gt = f["gt"][:][self.indices].astype(bool)

            images = f["images"]
            src_hw = tuple(images.shape[1:3])
            # Every read is a per-sample random access. A contiguous
            # uncompressed images dataset is a direct offset read (the OS
            # page cache does the rest); a compressed multi-image-chunk
            # layout decompresses a whole slab per sample and is
            # pathologically slow.
            if images.compression and images.chunks and images.chunks[0] > 1:
                print(f"WARNING: {h5_path} stores compressed multi-image "
                      f"chunks {images.chunks}; per-sample reads will be very "
                      "slow. Rewrite it with: python optimize_h5.py "
                      f"{h5_path} <out.h5>")

        self.hard_negative_index = len(self.classes) - 1
        self.num_classes = len(self.classes)

        # sources already at model size (e.g. pre-resized by optimize_h5.py)
        # skip the redundant resize op
        self.transform = build_transform(
            imagenet_norm,
            resize=src_hw != (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        img = self._file["images"][self.indices[i]]  # (H, W, C) storage dtype
        img = to_model_input(img)  # uint8 stays; uint16/float16 -> f32 [0,1]
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # CHW
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        if self.augment_pre is not None:  # train split; storage-scale crop
            img = self.augment_pre(img.contiguous())
        img = self.transform(img)
        if self.augment_post is not None:  # e.g. erasing, at model scale
            img = self.augment_post(img)

        return img, int(self.labels[i]), i

    def __getstate__(self):
        # Never pickle an open handle into a worker; each worker reopens.
        state = self.__dict__.copy()
        state["_file"] = None
        return state
