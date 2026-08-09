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

    def __init__(self, h5_path: str, split: int, imagenet_norm: bool = False):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {list(SPLIT_NAMES)}, got {split}")
        self.h5_path = h5_path
        self.split = split
        self.imagenet_norm = imagenet_norm
        self._file: h5py.File | None = None  # opened lazily per worker

        with h5py.File(h5_path, "r") as f:
            self.classes = list(f["classes"].asstr()[:])
            all_split = f["split"][:]
            self.indices = np.flatnonzero(all_split == split)  # h5 row per sample
            self.labels = f["labels"][:][self.indices].astype(np.int64)
            self.gt = f["gt"][:][self.indices].astype(bool)

        self.hard_negative_index = len(self.classes) - 1
        self.num_classes = len(self.classes)

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
        self.transform = v2.Compose(ops)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")

        img = self._file["images"][self.indices[i]]  # (H, W, C) uint8
        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)  # CHW
        if img.shape[0] == 1:
            img = img.expand(3, -1, -1)
        img = self.transform(img)

        return img, int(self.labels[i]), i

    def __getstate__(self):
        # Never pickle an open handle into a worker; each worker reopens.
        state = self.__dict__.copy()
        state["_file"] = None
        return state
