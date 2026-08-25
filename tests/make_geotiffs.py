"""Generate georeferenced test images with simple shapes, for the full
labelling -> gt.json + h5 -> train -> blind-inference loop.

Each GeoTIFF is a noisy gray background with scattered filled shapes —
circle, square, triangle, ring (the ring makes a natural hard-negative
foil for the circle) — drawn in a shared muted color with per-instance
jitter, so a classifier must learn GEOMETRY, not color. RGB uint8,
EPSG:3857 georeferencing with a 0.01 m pixel and per-image origins laid
out on a grid (same scale as the GeoLabelling example export). Written
uncompressed/stripped so plain PIL readers (inference.py) open them too.

A manifest.json records exactly what was drawn where (shape, center pixel,
size per image) so labels can be cross-checked.

Requires rasterio (not a core pipeline dependency):

    python tests/make_geotiffs.py --out-dir tests/data/geotiffs
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

SHAPES = ("circle", "square", "triangle", "ring")
PIXEL_SIZE = 0.01  # metres per pixel in EPSG:3857, per the example export


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, cx: float, cy: float,
               size: float, color, bg) -> None:
    r = size / 2
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color)
    elif shape == "triangle":
        pts = [(cx + r * math.sin(a), cy - r * math.cos(a))
               for a in (0, 2 * math.pi / 3, 4 * math.pi / 3)]
        draw.polygon(pts, fill=color)
    elif shape == "ring":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
        r2 = r * 0.55
        draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=bg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", default="tests/data/geotiffs")
    p.add_argument("--count", type=int, default=4, help="number of images")
    p.add_argument("--size", type=int, default=1024, help="image width/height")
    p.add_argument("--min-shapes", type=int, default=6)
    p.add_argument("--max-shapes", type=int, default=12)
    p.add_argument("--shape-size", type=int, nargs=2, default=(40, 80),
                   metavar=("MIN", "MAX"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        raise SystemExit("this generator needs rasterio: pip install rasterio")

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = args.size
    bg_value = 128
    manifest = {"crs": "EPSG:3857", "pixel_size_m": PIXEL_SIZE, "images": []}

    for i in range(args.count):
        # noisy background, shapes drawn on top
        base = (bg_value + rng.integers(-18, 19, (s, s, 3))).clip(0, 255)
        pil = Image.fromarray(base.astype(np.uint8))
        draw = ImageDraw.Draw(pil)

        n_shapes = int(rng.integers(args.min_shapes, args.max_shapes + 1))
        placed: list[dict] = []
        attempts = 0
        while len(placed) < n_shapes and attempts < 500:
            attempts += 1
            size = float(rng.integers(args.shape_size[0], args.shape_size[1] + 1))
            margin = size
            cx = float(rng.uniform(margin, s - margin))
            cy = float(rng.uniform(margin, s - margin))
            if any(math.hypot(cx - q["cx"], cy - q["cy"]) <
                   (size + q["size"]) * 0.75 for q in placed):
                continue  # keep shapes separated
            shape = str(rng.choice(SHAPES))
            jitter = rng.integers(-12, 13, 3)
            color = tuple(int(c) for c in
                          np.clip(np.array([62, 70, 92]) + jitter, 0, 255))
            draw_shape(draw, shape, cx, cy, size, color,
                       (bg_value, bg_value, bg_value))
            placed.append({"shape": shape, "cx": round(cx, 1),
                           "cy": round(cy, 1), "size": size})

        # grid of per-image origins so images don't overlap geographically
        col, row = i % 2, i // 2
        x0 = -50.0 + col * (s * PIXEL_SIZE + 20.0)
        y0 = 295.0 - row * (s * PIXEL_SIZE + 20.0)
        path = out_dir / f"shapes_{i:02d}.tif"
        arr = np.asarray(pil)
        with rasterio.open(
                path, "w", driver="GTiff", height=s, width=s, count=3,
                dtype="uint8", crs="EPSG:3857",
                transform=from_origin(x0, y0, PIXEL_SIZE, PIXEL_SIZE)) as dst:
            dst.write(arr.transpose(2, 0, 1))

        Image.open(path).convert("RGB")  # PIL readability (inference.py)
        counts = {}
        for q in placed:
            counts[q["shape"]] = counts.get(q["shape"], 0) + 1
        manifest["images"].append({
            "name": path.stem, "path": str(path), "origin": [x0, y0],
            "shapes": placed})
        print(f"{path}: {len(placed)} shapes "
              + ", ".join(f"{v}x {k}" for k, v in sorted(counts.items())))

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                           encoding="utf-8")
    print(f"\nmanifest: {out_dir / 'manifest.json'}")
    print("next: label these in GeoLabeller, export gt.json, build the h5, "
          "train, then\n  python inference.py <run_dir> "
          f"{out_dir} --window-width 128 --window-height 128 --stride-x 64 "
          "--gt <gt.json>")


if __name__ == "__main__":
    main()
