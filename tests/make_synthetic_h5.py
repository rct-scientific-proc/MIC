"""Generate a small synthetic dataset in the h5_format.md layout.

Classes are brightness bands (learnable by a ResNet in a few epochs), and
hard negatives sit between the genuine bands. Doubles as a worked example of
the h5 format.

    python tests/make_synthetic_h5.py out.h5 --genuine-per-class 30 --hn-factor 5
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


def make_dataset(
    path: str,
    num_genuine_classes: int = 2,
    genuine_per_class: int = 30,
    hn_factor: float = 5.0,
    image_hw: tuple[int, int] = (28, 28),
    channels: int = 1,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    class_names = [f"band{i}" for i in range(num_genuine_classes)] + ["hard_negative"]
    hn_index = num_genuine_classes

    # Genuine classes get well-separated brightness bands; hard negatives sit
    # halfway between two neighboring bands (they "look like" a class).
    band = np.linspace(30, 225, num_genuine_classes)

    labels, splits = [], []
    for split in (0, 1, 2):
        n_gen = genuine_per_class if split == 0 else max(2, genuine_per_class // 3)
        for i in range(n_gen * num_genuine_classes):
            labels.append(i % num_genuine_classes)
            splits.append(split)
        for _ in range(int(n_gen * num_genuine_classes * hn_factor)):
            labels.append(hn_index)
            splits.append(split)

    labels = np.array(labels, dtype=np.uint16)
    splits = np.array(splits, dtype=np.uint8)
    gt = labels != hn_index

    brightness = np.empty(len(labels))
    genuine_mask = gt
    brightness[genuine_mask] = band[labels[genuine_mask]]
    if num_genuine_classes > 1:
        midpoints = (band[:-1] + band[1:]) / 2
        brightness[~genuine_mask] = rng.choice(midpoints, (~genuine_mask).sum())
    else:
        brightness[~genuine_mask] = band[0] + 60

    h, w = image_hw
    noise = rng.integers(-20, 21, (len(labels), h, w, channels))
    images = (brightness[:, None, None, None] + noise).clip(0, 255).astype(np.uint8)

    with h5py.File(path, "w") as f:
        f["images"] = images
        f["labels"] = labels
        f["gt"] = gt
        f["split"] = splits
        f["classes"] = np.array(class_names, dtype=object)

    counts = {s: int((splits == s).sum()) for s in (0, 1, 2)}
    print(f"wrote {path}: {len(labels)} samples {images.shape[1:]}, "
          f"classes {class_names}, split sizes {counts}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("out", help="output .h5 path")
    p.add_argument("--num-classes", type=int, default=2, help="genuine classes")
    p.add_argument("--genuine-per-class", type=int, default=30,
                   help="genuine train samples per class")
    p.add_argument("--hn-factor", type=float, default=5.0,
                   help="hard negatives per genuine sample")
    p.add_argument("--size", type=int, default=28, help="image height/width")
    p.add_argument("--channels", type=int, choices=(1, 3), default=1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    make_dataset(args.out, args.num_classes, args.genuine_per_class,
                 args.hn_factor, (args.size, args.size), args.channels, args.seed)


if __name__ == "__main__":
    main()
