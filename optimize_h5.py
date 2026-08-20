"""Rewrite an h5 dataset into a load-optimized layout.

Any h5 in the h5_format.md layout — whatever wrote it — is repacked with
per-image chunks and your choice of compression, so shuffled per-sample
reads never decompress multi-image slabs (h5py's default chunking under
gzip is pathologically slow to stream: a single-image read can decompress
thousands of images). Labels/gt/split/classes are copied verbatim; the
output is validated and a measured random-read throughput comparison is
printed.

Optional --resize N additionally resizes every image to NxN (the model
input size is 224), batched through the GPU when available. Worthwhile
when sources are LARGER than the target (the file shrinks and the
per-sample CPU resize disappears — training skips its resize op for files
already at model size). When sources are smaller, pre-resizing inflates
the file by the square of the scale factor (32px -> 224px is 49x) and can
push it past the training RAM cache — the script warns before doing it.

    python optimize_h5.py in.h5 out.h5                  # repack only
    python optimize_h5.py in.h5 out.h5 --resize 224     # repack + resize
    python optimize_h5.py in.h5 out.h5 --compression none
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torchvision.transforms import v2
from tqdm import tqdm

from dataset import RESNET_INPUT_SIZE, validate_h5


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("input", help="source .h5 (h5_format.md layout)")
    p.add_argument("output", help="optimized .h5 to write")
    p.add_argument("--resize", type=int, default=None, metavar="N",
                   help=f"resize images to NxN (model input is "
                        f"{RESNET_INPUT_SIZE}); optional — omit to repack "
                        "only")
    p.add_argument("--compression", choices=("gzip", "lzf", "none"),
                   default="gzip")
    p.add_argument("--gzip-level", type=int, default=4, choices=range(0, 10))
    p.add_argument("--batch-size", type=int, default=1024,
                   help="images processed per read/resize/write step")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--no-progress", action="store_true")
    return p.parse_args(argv)


def _read_speed(path, k: int = 512, seed: int = 0) -> float:
    """Random single-sample reads per second (the training access pattern
    when streaming)."""
    with h5py.File(path, "r") as f:
        images = f["images"]
        idx = np.random.default_rng(seed).integers(0, images.shape[0],
                                                   min(k, images.shape[0]))
        t0 = time.perf_counter()
        for i in idx:
            images[int(i)]
        return len(idx) / (time.perf_counter() - t0)


def main(argv=None) -> None:
    args = parse_args(argv)
    validate_h5(args.input)  # refuse to "optimize" a malformed file
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available()
                                          else "cpu"))

    with h5py.File(args.input, "r") as fin:
        n, h, w, c = fin["images"].shape
        th = tw = args.resize or 0
        resizing = bool(args.resize) and (th, tw) != (h, w)
        oh, ow = (th, tw) if resizing else (h, w)

        old_logical = n * h * w * c
        new_logical = n * oh * ow * c
        if new_logical > old_logical:
            print(f"WARNING: resizing {h}x{w} -> {oh}x{ow} INFLATES the data "
                  f"{new_logical / old_logical:.1f}x "
                  f"({old_logical / 1e6:.0f} MB -> {new_logical / 1e6:.0f} MB "
                  "uncompressed). Pre-resizing only pays off when sources are "
                  "larger than the target; for small sources prefer the "
                  "training RAM cache + --workers.")

        resizer = None
        if resizing:
            resizer = v2.Resize((th, tw),
                                interpolation=v2.InterpolationMode.BILINEAR,
                                antialias=True)

        comp = {}
        if args.compression == "gzip":
            comp = dict(compression="gzip", compression_opts=args.gzip_level)
        elif args.compression == "lzf":
            comp = dict(compression="lzf")

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as fout:
            dset = fout.create_dataset("images", shape=(n, oh, ow, c),
                                       dtype=np.uint8,
                                       chunks=(1, oh, ow, c), **comp)
            for name in ("labels", "gt", "split"):
                fout[name] = fin[name][:]
            fout["classes"] = np.array(fin["classes"].asstr()[:], dtype=object)

            steps = range(0, n, args.batch_size)
            for start in tqdm(steps, desc="optimize", unit="batch",
                              disable=args.no_progress):
                block = fin["images"][start:start + args.batch_size]
                if resizer is not None:
                    t = torch.from_numpy(block).permute(0, 3, 1, 2).to(device)
                    t = resizer(t)
                    block = t.permute(0, 2, 3, 1).cpu().numpy()
                dset[start:start + len(block)] = block

    validate_h5(str(out))
    old_mb = Path(args.input).stat().st_size / 1e6
    new_mb = out.stat().st_size / 1e6
    old_speed = _read_speed(args.input)
    new_speed = _read_speed(out)
    print(f"{args.input}: {old_mb:.1f} MB, {old_speed:,.0f} random reads/s")
    print(f"{out}: {new_mb:.1f} MB, {new_speed:,.0f} random reads/s "
          f"({new_speed / max(old_speed, 1e-9):.1f}x)")
    if resizing and (oh, ow) == (RESNET_INPUT_SIZE, RESNET_INPUT_SIZE):
        print("images are at model size: training/evaluation will skip the "
              "resize op for this file")


if __name__ == "__main__":
    main()
