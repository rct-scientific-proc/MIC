"""Download a public image-classification dataset and convert it to the
project's h5 format (h5_format.md), with a subset of classes relabeled as
hard negatives.

Datasets come from torchvision (downloaded once into --data-dir, cached
thereafter): mnist, fashionmnist (28x28 grayscale), cifar10, cifar100
(32x32 RGB). A random --hn-classes subset of the original classes (or an
explicit --hn-names list) becomes the single hard_negative class — "things
that look like objects but are not the objects we care about" — and the
remaining classes become the genuine classes. The torchvision train split
is carved into train/validate (stratified per class); the torchvision test
split becomes the test split.

    python tests/make_real_h5.py fashionmnist fashion.h5 --hn-classes 4
    python tests/make_real_h5.py cifar100 cifar.h5 --hn-classes 80 \
        --subset-per-class 200

Tip: the more classes you assign to hard_negative, the closer the h5 gets
to this pipeline's heavy-imbalance regime (cifar100 with 80+ HN classes is
a good stress test).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from dataset import HARD_NEGATIVE_NAME, validate_h5  # noqa: E402

DATASETS = ("mnist", "fashionmnist", "cifar10", "cifar100")


def _load_split(name: str, data_dir: str, train: bool):
    import torchvision.datasets as tvd
    ctor = {"mnist": tvd.MNIST, "fashionmnist": tvd.FashionMNIST,
            "cifar10": tvd.CIFAR10, "cifar100": tvd.CIFAR100}[name]
    ds = ctor(root=data_dir, train=train, download=True)
    data = ds.data
    images = data.numpy() if hasattr(data, "numpy") else np.asarray(data)
    if images.ndim == 3:  # (N, H, W) grayscale -> (N, H, W, 1)
        images = images[..., None]
    labels = np.asarray(ds.targets, dtype=np.int64)
    class_names = [c.replace(" ", "_") for c in ds.classes]
    return images.astype(np.uint8), labels, class_names


def _cap_per_class(images, labels, cap, rng):
    if cap is None:
        return images, labels
    keep = []
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        rng.shuffle(pos)
        keep.append(pos[:cap])
    keep = np.sort(np.concatenate(keep))
    return images[keep], labels[keep]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("out", help="output .h5 path")
    p.add_argument("--hn-classes", type=int, default=None,
                   help="number of ORIGINAL classes to relabel as "
                        "hard_negative, chosen at random with --seed "
                        "(required unless --hn-names is given)")
    p.add_argument("--hn-names", default=None,
                   help="comma-separated original class names to relabel as "
                        "hard_negative (overrides --hn-classes)")
    p.add_argument("--subset-per-class", type=int, default=None,
                   help="cap samples per original class in each source split "
                        "(keeps the h5 small for quick experiments)")
    p.add_argument("--val-frac", type=float, default=0.15,
                   help="fraction of the torchvision train split carved into "
                        "validation (stratified per class)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", default=str(Path(__file__).parent / "data"),
                   help="torchvision download/cache directory")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    tr_images, tr_labels, class_names = _load_split(args.dataset, args.data_dir, True)
    te_images, te_labels, _ = _load_split(args.dataset, args.data_dir, False)

    # --- choose the hard-negative classes --------------------------------
    if args.hn_names:
        wanted = [n.strip() for n in args.hn_names.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in class_names]
        if unknown:
            p.error(f"--hn-names: unknown class(es) {unknown}; "
                    f"available: {', '.join(class_names)}")
        hn_orig = sorted(class_names.index(n) for n in wanted)
    elif args.hn_classes:
        if args.hn_classes >= len(class_names) - 1:
            p.error(f"--hn-classes must leave at least 2 genuine classes "
                    f"(dataset has {len(class_names)})")
        hn_orig = sorted(rng.choice(len(class_names), args.hn_classes,
                                    replace=False).tolist())
    else:
        p.error("give --hn-classes N or --hn-names a,b,...")

    genuine_orig = [c for c in range(len(class_names)) if c not in hn_orig]
    if len(genuine_orig) < 2:
        p.error("at least 2 genuine classes must remain")
    remap = {orig: new for new, orig in enumerate(genuine_orig)}
    hn_index = len(genuine_orig)
    remap.update({orig: hn_index for orig in hn_orig})
    out_classes = [class_names[c] for c in genuine_orig] + [HARD_NEGATIVE_NAME]
    print(f"hard_negative <- {', '.join(class_names[c] for c in hn_orig)}")
    print(f"genuine classes ({len(genuine_orig)}): "
          + ", ".join(class_names[c] for c in genuine_orig))

    # --- cap, remap, and carve validation out of train -------------------
    tr_images, tr_labels = _cap_per_class(tr_images, tr_labels,
                                          args.subset_per_class, rng)
    te_images, te_labels = _cap_per_class(te_images, te_labels,
                                          args.subset_per_class, rng)
    tr_new = np.array([remap[c] for c in tr_labels], dtype=np.uint16)
    te_new = np.array([remap[c] for c in te_labels], dtype=np.uint16)

    tr_split = np.zeros(len(tr_new), dtype=np.uint8)  # 0 train, 1 validate
    for c in np.unique(tr_new):
        pos = np.flatnonzero(tr_new == c)
        rng.shuffle(pos)
        tr_split[pos[:max(1, round(args.val_frac * len(pos)))]] = 1

    images = np.concatenate([tr_images, te_images])
    labels = np.concatenate([tr_new, te_new])
    split = np.concatenate([tr_split, np.full(len(te_new), 2, dtype=np.uint8)])

    order = rng.permutation(len(labels))  # shuffle rows; index map stays 1:1
    images, labels, split = images[order], labels[order], split[order]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        # contiguous and uncompressed: a per-sample read is a direct offset
        # into the file (compression/chunking make shuffled per-sample reads
        # slow; disk is cheap, training time is not)
        f.create_dataset("images", data=images)
        f["labels"] = labels
        f["gt"] = labels != hn_index
        f["split"] = split
        f["classes"] = np.array(out_classes, dtype=object)

    summary = validate_h5(str(out))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB, "
          f"{len(labels)} samples {images.shape[1:]})")
    for split_name, c in summary["counts"].items():
        print(f"  {split_name}: {c['genuine']} genuine, "
              f"{c['hard_negative']} hard negatives")
    print(f"\nnext: python train.py {out} --smart 3 --target-recall 0.9 "
          f"--imbalance-ratio 3 --imbalance-ratio-start 1 --amp")


if __name__ == "__main__":
    main()
