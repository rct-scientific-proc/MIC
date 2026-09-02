# HDF5 Dataset Format

All datasets used for training, validation, and testing are stored in a single `.h5` file.
Every array shares the same first axis length `N` (total number of samples), giving a 1-to-1
index mapping across all datasets.

---

## Datasets

| Name      | dtype   | Shape        | Description |
|-----------|---------|--------------|-------------|
| `images`  | see below | (N, H, W, C) | Pixel values. `C=1` for grayscale (expanded to 3 channels at model input), `C=3` for RGB. Allowed dtypes and ranges in the table below. |
| `labels`  | uint16  | (N,)         | Integer class index. Look up the name via `classes[labels[i]]`. |
| `gt`      | bool    | (N,)         | `True` = genuine example. `False` = hard negative. |
| `split`   | uint8   | (N,)         | `0` = train, `1` = validate, `2` = test. |
| `classes` | str     | (K,)         | Ordered list of class name strings. Last entry is always `"hard_negative"`. K = number unique labels uint16 values |
| `removed` | bool    | (N,)         | *Optional.* `True` = snippet curated out (written by `curate.py`); every loader excludes it from **all** splits. Purge permanently with `optimize_h5.py --drop-removed`. |

---

## Image dtypes

Allowed dtypes for the `images` dataset, and the pixel range each is
expected to hold:

| dtype     | expected pixel range | notes |
|-----------|----------------------|-------|
| `uint8`   | 0 – 255              | The classic choice; smallest on disk. |
| `uint16`  | 0 – 65535            | 16-bit integer imagery; the loader divides by 65535. |
| `float16` | 0.0 – 1.0            | Already-scaled 16-bit radiometry; widened to float32 for compute. |
| `float32` | 0.0 – 1.0            | Already-scaled floats. 4x the disk/IO of uint8 - use only when the data truly has more than 8 bits of dynamic range. |

`validate_h5` rejects any other dtype, and rejects float files whose sampled
values fall outside [0, 1]. All dtypes reach the model as float32 in [0, 1],
so models are interchangeable across storage dtypes of the same imagery.
Training-time augmentations see uint8 files at 0-255 and everything else as
float32 [0, 1] (the plugin contract in example_augment_plugin.py covers
both). `optimize_h5.py` preserves the source dtype, including through
`--resize`.

---

## Split Values

| Value | Meaning  |
|-------|----------|
| `0`   | Train    |
| `1`   | Validate |
| `2`   | Test     |

---

## Ground Truth Flag (`gt`)

`gt` separates *what class something is* (`labels`) from *whether it is a real example of that class* (`gt`).

| `gt`    | Meaning |
|---------|---------|
| `True`  | Genuine labelled example |
| `False` | Hard negative — looks like a class but is not one |

Hard negatives are assigned to the `"hard_negative"` class (last index in `classes`) and are
distributed across train/val/test splits proportionally to the genuine sample counts.

---

## Class Labels

`labels[i]` is a `uint16` index into the `classes` array:

```python
with h5py.File("dataset.h5", "r") as f:
    classes = f["classes"].asstr()[:]   # numpy array of strings
    label   = int(f["labels"][i])
    name    = classes[label]            # e.g. "3" or "hard_negative"
```

---

## Reading a Split

```python
import h5py
import numpy as np

with h5py.File("dataset.h5", "r") as f:
    split  = f["split"][:]
    images = f["images"][split == 0]   # all train images
    labels = f["labels"][split == 0]
    gt     = f["gt"][split == 0]
```

---

