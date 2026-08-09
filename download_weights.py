"""Pre-fetch ImageNet ResNet weights for offline machines.

Run on a connected machine, then copy the output directory to the offline box
and point train.py at the file with --weights-path:

    python download_weights.py --out-dir pretrained
    python download_weights.py --out-dir pretrained --archs resnet18 resnet50

    # offline machine
    python train.py ... --arch resnet50 --weights-path pretrained/resnet50.pth
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from model import ARCHS, weight_url


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out-dir", default="pretrained", help="destination directory")
    parser.add_argument(
        "--archs", nargs="+", choices=sorted(ARCHS), default=sorted(ARCHS),
        help="which architectures to fetch (default: all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for arch in args.archs:
        dst = out_dir / f"{arch}.pth"
        if dst.exists():
            print(f"{dst} already exists, skipping")
            continue
        url = weight_url(arch)
        print(f"{arch}: {url} -> {dst}")
        torch.hub.download_url_to_file(url, str(dst), progress=True)

    print("done")


if __name__ == "__main__":
    main()
