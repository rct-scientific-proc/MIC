"""Rewrite an h5 dataset into a load-optimized layout.

Any h5 in the h5_format.md layout — whatever wrote it — is rewritten with a
CONTIGUOUS, UNCOMPRESSED images dataset (the default): every per-sample
read is then a direct offset into the file, with no chunk lookup and no
decompression, and the OS page cache serves repeats. Compressed sources
(h5py's default chunking under gzip is pathologically slow to stream: a
single-image read can decompress thousands of images) become fast files at
the cost of disk. --compression gzip/lzf is still available when disk
matters more than read speed (written with per-image chunks so reads stay
sane). Labels/gt/split/classes are copied verbatim; the output is
validated and a measured random-read throughput comparison is printed.

Optional --resize N additionally resizes every image to NxN (the model
input size is 224), batched through the GPU when available. Worthwhile
when sources are LARGER than the target (the file shrinks and the
per-sample CPU resize disappears — training skips its resize op for files
already at model size). When sources are smaller, pre-resizing inflates
the file by the square of the scale factor (32px -> 224px is 49x) and can
push it past the training RAM cache — the script warns before doing it.

    python optimize_h5.py in.h5 out.h5                  # contiguous, no compression
    python optimize_h5.py in.h5 out.h5 --resize 224     # + resize to model size
    python optimize_h5.py in.h5 out.h5 --compression gzip   # smaller, slower reads
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
    p.add_argument("--compression", choices=("none", "gzip", "lzf"),
                   default="none",
                   help="default none: contiguous unchunked images, direct "
                        "offset reads; gzip/lzf trade read speed for disk "
                        "(written with per-image chunks)")
    p.add_argument("--gzip-level", type=int, default=4, choices=range(0, 10))
    p.add_argument("--drop-removed", action="store_true",
                   help="permanently purge snippets flagged by curate.py "
                        "(the 'removed' dataset) instead of copying them; "
                        "the output then carries no 'removed' dataset")
    p.add_argument("--sort-split", action="store_true",
                   help="group output rows by split (train | validate | "
                        "test) so each split is one contiguous region - "
                        "large, larger-than-RAM files then read much faster "
                        "(the every-epoch validation pass streams "
                        "sequentially). NOTE: this renames every row index, "
                        "so a resumed run's miner state and curate.py's "
                        "removed references no longer line up with the "
                        "output; sort before training, not after")
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
        keep = np.ones(n, dtype=bool)
        if args.drop_removed and "removed" in fin:
            keep = ~fin["removed"][:].astype(bool)
            print(f"dropping {int((~keep).sum())} removed snippet(s); "
                  f"{int(keep.sum())} remain")
        # Output row order: kept rows, optionally grouped by split. A
        # stable sort keeps each split's internal order (so a split that
        # was already contiguous is unchanged), and the identity order is
        # a plain ascending range so the no-sort path still slices
        # sequentially.
        kept = np.flatnonzero(keep)
        if args.sort_split:
            split_all = fin["split"][:]
            order = kept[np.argsort(split_all[kept], kind="stable")]
            print("grouping rows by split: "
                  + ", ".join(f"{name} {int((split_all[kept] == v).sum())}"
                              for v, name in ((0, "train"), (1, "validate"),
                                              (2, "test"))))
        else:
            order = kept
        n_out = len(order)
        th = tw = args.resize or 0
        resizing = bool(args.resize) and (th, tw) != (h, w)
        oh, ow = (th, tw) if resizing else (h, w)

        bpp = fin["images"].dtype.itemsize
        old_logical = n * h * w * c * bpp
        new_logical = n * oh * ow * c * bpp
        if new_logical > old_logical:
            print(f"WARNING: resizing {h}x{w} -> {oh}x{ow} INFLATES the data "
                  f"{new_logical / old_logical:.1f}x "
                  f"({old_logical / 1e6:.0f} MB -> {new_logical / 1e6:.0f} MB "
                  "uncompressed). Pre-resizing only pays off when sources are "
                  "larger than the target; for small sources keep the source "
                  "size and let training resize (--workers parallelizes it).")

        resizer = None
        if resizing:
            resizer = v2.Resize((th, tw),
                                interpolation=v2.InterpolationMode.BILINEAR,
                                antialias=True)

        # compression requires chunking in HDF5; without it, write a
        # contiguous unchunked dataset so reads are direct offsets
        comp = {}
        if args.compression == "gzip":
            comp = dict(compression="gzip", compression_opts=args.gzip_level,
                        chunks=(1, oh, ow, c))
        elif args.compression == "lzf":
            comp = dict(compression="lzf", chunks=(1, oh, ow, c))

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(out, "w") as fout:
            dset = fout.create_dataset("images", shape=(n_out, oh, ow, c),
                                       dtype=fin["images"].dtype, **comp)
            for name in ("labels", "gt", "split"):
                fout[name] = fin[name][:][order]
            if "removed" in fin and not args.drop_removed:
                # keep is all-True when not dropping, so order covers every
                # row; the mask follows the same permutation as the images
                fout["removed"] = fin["removed"][:][order]
            fout["classes"] = np.array(fin["classes"].asstr()[:], dtype=object)

            steps = range(0, n_out, args.batch_size)
            for out_start in tqdm(steps, desc="optimize", unit="batch",
                                  disable=args.no_progress):
                src = order[out_start:out_start + args.batch_size]
                # h5py point-selects in increasing order; read ascending
                # then restore the batch's output order
                perm = np.argsort(src, kind="stable")
                block = fin["images"][src[perm]]
                block = block[np.argsort(perm, kind="stable")]
                if resizer is not None:
                    src_dtype = block.dtype
                    work = (block if src_dtype == np.uint8
                            else block.astype(np.float32))
                    t = torch.from_numpy(work).permute(0, 3, 1, 2).to(device)
                    t = resizer(t)
                    block = t.permute(0, 2, 3, 1).cpu().numpy()
                    if src_dtype == np.uint16:
                        block = block.round().clip(0, 65535).astype(np.uint16)
                    elif block.dtype != src_dtype:
                        block = block.astype(src_dtype)
                dset[out_start:out_start + len(block)] = block

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
