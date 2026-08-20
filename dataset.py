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
    "solarize": lambda p=0.3, threshold=128:
        v2.RandomSolarize(threshold, p=p),
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


def build_transform(imagenet_norm: bool) -> v2.Compose:
    """The model input pipeline: CHW uint8 -> resize 224 -> float [0,1] ->
    optional ImageNet normalization. Shared by training, evaluation, and
    sliding-window inference so preprocessing can never diverge."""
    ops = [
        v2.Resize(
            (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE),
            interpolation=v2.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.ToDtype(torch.float32, scale=True),  # uint8 -> [0, 1]
    ]
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

        classes = f["classes"].asstr()[:]
        if classes[-1] != HARD_NEGATIVE_NAME:
            raise ValueError(
                f"{path}: last class must be '{HARD_NEGATIVE_NAME}', got '{classes[-1]}'"
            )
        hn_index = len(classes) - 1

        labels = f["labels"][:]
        gt = f["gt"][:].astype(bool)
        split = f["split"][:]

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
            mask = split == value
            counts[name] = {
                "genuine": int((mask & gt).sum()),
                "hard_negative": int((mask & ~gt).sum()),
            }

    return {"classes": list(classes), "hard_negative_index": hn_index, "counts": counts}


class H5SnippetDataset(Dataset):
    """One split of an h5 snippet file, ready for a ResNet.

    __getitem__ returns (image, label, index) where index is the position
    within this split — used by the mining tracker to attribute per-sample
    errors back to dataset entries.

    Transform pipeline: uint8 HWC -> CHW tensor -> resize 224 -> float [0,1]
    -> optional ImageNet normalization. Grayscale is repeated to 3 channels.
    """

    # auto-cache the split's images in RAM when they fit in this budget;
    # per-sample streaming from a compressed h5 can be orders of magnitude
    # slower (every read decompresses a whole multi-image chunk)
    CACHE_BUDGET_BYTES = 2 * 1024**3

    def __init__(self, h5_path: str, split: int, imagenet_norm: bool = False,
                 augment=None, cache_images="auto"):
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
            self.indices = np.flatnonzero(all_split == split)  # h5 row per sample
            self.labels = f["labels"][:][self.indices].astype(np.int64)
            self.gt = f["gt"][:][self.indices].astype(bool)

            images = f["images"]
            split_bytes = len(self.indices) * int(np.prod(images.shape[1:]))
            if cache_images == "auto":
                cache_images = split_bytes <= self.CACHE_BUDGET_BYTES
            self.cached = images[self.indices] if cache_images else None
            if self.cached is None and images.compression and \
                    images.chunks and images.chunks[0] > 1:
                print(f"WARNING: {h5_path} streams compressed multi-image "
                      f"chunks {images.chunks} per sample read — expect very "
                      "slow loading. Cache the split in RAM (default when it "
                      "fits) or rewrite the file with per-image chunks.")

        self.hard_negative_index = len(self.classes) - 1
        self.num_classes = len(self.classes)

        self.transform = build_transform(imagenet_norm)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self.cached is not None:
            img = self.cached[i]  # (H, W, C) uint8, RAM
        else:
            if self._file is None:
                self._file = h5py.File(self.h5_path, "r")
            img = self._file["images"][self.indices[i]]  # (H, W, C) uint8
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # CHW
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        if self.augment_pre is not None:  # training split only; uint8 here
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
