"""Blind sliding-window inference over full images (Phase 7).

Loads whole images with Pillow, tiles them into sub-windows of a given size
and stride (row-major traversal, final row/column clamped inward so the
right/bottom edges are always covered), runs every window through a trained
checkpoint with the exact training preprocessing, and applies the
checkpoint's STORED operating thresholds — a window is a detection when it
is accepted as a genuine (non-hard-negative) class.

Outputs in --out-dir:
    detections.csv           every accepted window: image, x, y, w, h,
                             class, score
    report_<UTCstamp>.pdf    summary table of all images; one annotated page
                             per image that contains detections (raw window
                             boxes, colored per class)
    assets/                  the annotated overlay PNGs

Example:
    python inference.py runs/exp1 photo1.png scans/ \
        --window-width 128 --window-height 128 --stride-x 64
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from fpdf import FPDF
from PIL import Image, ImageDraw
from tqdm import tqdm

from checkpoints import find_checkpoint, utc_stamp
from dataset import build_transform
from metrics import _per_sample_thresholds, genuineness_scores, non_hn_argmax
from model import build_model
from plots import SERIES

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MAX_TABLE_ROWS = 40  # per-image detection rows shown in the PDF

INK = (11, 11, 11)
INK_2 = (82, 81, 78)
MUTED = (137, 135, 129)
LINE = (225, 224, 217)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint",
                   help="checkpoint from train.py: a .pt file, or a run "
                        "directory (uses its newest best_* checkpoint)")
    p.add_argument("images", nargs="+",
                   help="image files and/or directories (scanned for "
                        + "/".join(sorted(e.lstrip('.') for e in IMAGE_EXTS)) + ")")
    p.add_argument("--window-width", type=int, required=True)
    p.add_argument("--window-height", type=int, required=True)
    p.add_argument("--stride-x", type=int, required=True)
    p.add_argument("--stride-y", type=int, default=None,
                   help="default: same as --stride-x")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--no-amp", action="store_true",
                   help="disable mixed-precision on CUDA (AMP is on by "
                        "default; scores can differ by ~1e-3 from fp32, which "
                        "only matters for windows sitting exactly at a "
                        "threshold)")
    p.add_argument("--grayscale", action="store_true",
                   help="convert images to grayscale before windowing (for "
                        "models trained on grayscale snippets); default RGB")
    p.add_argument("--out-dir", default=None,
                   help="default: inference_<UTCstamp>/")
    p.add_argument("--no-report", action="store_true",
                   help="skip the PDF (detections.csv is always written)")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)
    if args.stride_y is None:
        args.stride_y = args.stride_x
    for name in ("window_width", "window_height", "stride_x", "stride_y"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def gather_images(inputs) -> list[Path]:
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(q for q in p.iterdir()
                                if q.suffix.lower() in IMAGE_EXTS))
        elif p.is_file():
            paths.append(p)
        else:
            raise SystemExit(f"input not found: {p}")
    if not paths:
        raise SystemExit("no image files found in the given inputs")
    return paths


def window_positions(size: int, window: int, stride: int) -> list[int]:
    """Row-major grid offsets along one axis; the final position is clamped
    inward so the edge is covered (windows larger than the image start at 0)."""
    last = max(size - window, 0)
    xs = list(range(0, last + 1, stride))
    if xs[-1] != last:
        xs.append(last)
    return xs


@torch.no_grad()
def infer_image(img: np.ndarray, model, transform, device, args,
                hn_index: int, operating, desc: str) -> tuple[list[dict], int]:
    """Slide over one image (HWC uint8); returns (accepted windows, n_windows).

    Throughput notes: every window in an image has identical dimensions, so
    a whole batch of raw uint8 crops is shipped to the device in one pinned
    non-blocking transfer (~12x less PCIe traffic than fp32 224x224) and the
    resize/scale/normalize transform runs batched on the device; the forward
    pass runs under autocast on CUDA. Decisions are identical to the
    unbatched path — probabilities are always computed in fp32.
    """
    ih, iw = img.shape[:2]
    w = min(args.window_width, iw)
    h = min(args.window_height, ih)
    coords = [(x, y) for y in window_positions(ih, h, args.stride_y)
              for x in window_positions(iw, w, args.stride_x)]  # row-major
    use_cuda = device.type == "cuda"
    amp = use_cuda and not getattr(args, "no_amp", False)

    detections: list[dict] = []
    bar = tqdm(range(0, len(coords), args.batch_size), desc=desc, unit="batch",
               leave=False, disable=args.no_progress)
    for start in bar:
        batch_coords = coords[start:start + args.batch_size]
        batch = np.stack([img[y:y + h, x:x + w] for x, y in batch_coords])
        t = torch.from_numpy(batch)  # (B, H, W, C) uint8, contiguous
        if use_cuda:
            t = t.pin_memory()
        t = t.to(device, non_blocking=True).permute(0, 3, 1, 2)
        if t.shape[1] == 1:
            t = t.expand(-1, 3, -1, -1)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(transform(t))
        probs = torch.softmax(logits.float(), dim=1).cpu().numpy()

        scores = genuineness_scores(probs, hn_index)
        pred = non_hn_argmax(probs, hn_index)
        accepted = scores >= _per_sample_thresholds(operating, pred,
                                                    probs.shape[1])
        for i, (x, y) in enumerate(batch_coords):
            if accepted[i]:
                detections.append({"x": x, "y": y, "w": w, "h": h,
                                   "class_index": int(pred[i]),
                                   "score": float(scores[i])})
    return detections, len(coords)


def draw_overlay(img: np.ndarray, detections: list[dict], classes,
                 hn_index: int, path: Path) -> None:
    """Raw window boxes on the full image, one fixed color per class."""
    pil = Image.fromarray(img.squeeze() if img.shape[-1] == 1 else img)
    pil = pil.convert("RGB")
    draw = ImageDraw.Draw(pil)
    for d in detections:
        color = SERIES[genuine_slot(d["class_index"], hn_index)]
        draw.rectangle([d["x"], d["y"], d["x"] + d["w"] - 1, d["y"] + d["h"] - 1],
                       outline=color, width=2)
    pil.save(path)


def genuine_slot(class_index: int, hn_index: int) -> int:
    """Stable palette slot for a genuine class (hn never gets drawn)."""
    slot = class_index if class_index < hn_index else class_index - 1
    return slot % len(SERIES)


class _Pdf(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"mic blind inference - page {self.page_no()}/{{nb}}",
                  align="C")


def _txt(s) -> str:
    return str(s).encode("latin-1", "replace").decode("latin-1")


def _h1(pdf, text):
    pdf.set_font("helvetica", "B", 15)
    pdf.set_text_color(*INK)
    pdf.cell(0, 9, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2.5)


def _para(pdf, text, size=9.5):
    pdf.set_font("helvetica", "", size)
    pdf.set_text_color(*INK_2)
    pdf.multi_cell(0, 5, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


def _table(pdf, headers, rows, widths):
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(*MUTED)
    for hd, wd in zip(headers, widths):
        pdf.cell(wd, 5.6, _txt(hd), border="B")
    pdf.ln()
    pdf.set_font("helvetica", "", 8.8)
    pdf.set_text_color(*INK)
    pdf.set_draw_color(*LINE)
    for row in rows:
        if pdf.get_y() > pdf.page_break_trigger - 8:
            pdf.add_page()
        for v, wd in zip(row, widths):
            pdf.cell(wd, 5.4, _txt(v), border="B")
        pdf.ln()
    pdf.ln(1.5)


def _image(pdf, path, w=None):
    w = w or pdf.epw
    with Image.open(path) as im:
        hgt = w * im.height / im.width
    if hgt > pdf.eph:  # very tall images: fit to page height instead
        w = w * pdf.eph / hgt
        hgt = pdf.eph
    if pdf.get_y() + hgt > pdf.page_break_trigger:
        pdf.add_page()
    pdf.image(str(path), w=w, x=pdf.l_margin)
    pdf.ln(2)


def build_pdf(out_dir: Path, results: list[dict], classes, hn_index: int,
              ckpt_path, args) -> Path:
    pdf = _Pdf(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    total_det = sum(len(r["detections"]) for r in results)
    flagged = [r for r in results if r["detections"]]
    _h1(pdf, "Blind inference report")
    _para(pdf, f"checkpoint: {ckpt_path}")
    _para(pdf, f"window {args.window_width}x{args.window_height}, stride "
               f"{args.stride_x}x{args.stride_y}, "
               f"{'grayscale' if args.grayscale else 'RGB'} input | "
               f"{len(results)} images, {total_det} detections in "
               f"{len(flagged)} images")

    _table(pdf, ["image", "size", "windows", "detections", "classes found"],
           [[r["path"].name, f"{r['size'][0]}x{r['size'][1]}", r["n_windows"],
             len(r["detections"]),
             ", ".join(sorted({classes[d["class_index"]]
                               for d in r["detections"]})) or "-"]
            for r in results],
           widths=[pdf.epw * w for w in (0.34, 0.14, 0.12, 0.13, 0.27)])

    for r in flagged:
        pdf.add_page()
        _h1(pdf, r["path"].name)
        counts = {}
        for d in r["detections"]:
            counts[d["class_index"]] = counts.get(d["class_index"], 0) + 1
        legend = "  |  ".join(
            f"{classes[c]}: {n} (color {genuine_slot(c, hn_index) + 1})"
            for c, n in sorted(counts.items()))
        _para(pdf, legend)
        _image(pdf, r["overlay"])
        rows = [[d["x"], d["y"], d["w"], d["h"], classes[d["class_index"]],
                 f"{d['score']:.4f}"]
                for d in sorted(r["detections"], key=lambda d: -d["score"])]
        if len(rows) > MAX_TABLE_ROWS:
            _para(pdf, f"top {MAX_TABLE_ROWS} of {len(rows)} detections by "
                       "score (all are in detections.csv):", size=8.5)
            rows = rows[:MAX_TABLE_ROWS]
        _table(pdf, ["x", "y", "w", "h", "class", "score"], rows,
               widths=[pdf.epw * w for w in (0.14, 0.14, 0.12, 0.12, 0.28, 0.2)])

    out_path = out_dir / f"report_{utc_stamp()}.pdf"
    pdf.output(str(out_path))
    return out_path


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = find_checkpoint(args.checkpoint, "best")
    if ckpt_path is None:
        raise SystemExit(f"no checkpoint found at {args.checkpoint}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    hn_index = ckpt["hard_negative_index"]
    per_class = (ckpt.get("threshold_mode") == "per-class"
                 and ckpt.get("class_thresholds"))
    operating = ckpt["class_thresholds"] if per_class else ckpt["threshold"]

    model = build_model(ckpt["arch"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    transform = build_transform(ckpt["imagenet_norm"])

    out_dir = Path(args.out_dir or f"inference_{utc_stamp()}")
    assets = out_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    results = []
    for path in gather_images(args.images):
        pil = Image.open(path).convert("L" if args.grayscale else "RGB")
        img = np.asarray(pil)
        if img.ndim == 2:
            img = img[:, :, None]
        detections, n_windows = infer_image(
            img, model, transform, device, args, hn_index, operating,
            desc=path.name)
        r = {"path": path, "size": pil.size, "n_windows": n_windows,
             "detections": detections, "overlay": None}
        if detections:
            r["overlay"] = assets / f"{path.stem}_overlay.png"
            draw_overlay(img, detections, classes, hn_index, r["overlay"])
        results.append(r)
        print(f"{path.name}: {n_windows} windows, {len(detections)} detections"
              + ("" if not detections else "  <- flagged"))

    with open(out_dir / "detections.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x", "y", "w", "h", "class", "score"])
        for r in results:
            for d in r["detections"]:
                writer.writerow([r["path"], d["x"], d["y"], d["w"], d["h"],
                                 classes[d["class_index"]], f"{d['score']:.6f}"])

    if not args.no_report:
        pdf_path = build_pdf(out_dir, results, classes, hn_index, ckpt_path, args)
        print(f"report: {pdf_path}")
    print(f"outputs written to {out_dir}")


if __name__ == "__main__":
    main()
