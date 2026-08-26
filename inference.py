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
    report_<UTCstamp>.pdf    cover (run facts, per-image table) and, with
                             --gt, a verdict box, per-class summary, score
                             separation + specificity-at-recall, per-class
                             ROC, and snippet grids (top-N, ranked
                             should-have-caught, confusions, rejections)
    assets/                  annotated overlay PNGs (window boxes, GT
                             hit/miss markers) and the report's chart/grid
                             PNGs

Example:
    python inference.py runs/exp1 photo1.png scans/ \
        --window-width 128 --window-height 128 --stride-x 64
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from fpdf import FPDF
from PIL import Image, ImageDraw
from torchmetrics.functional.classification import binary_auroc, binary_roc
from tqdm import tqdm

from checkpoints import find_checkpoint, utc_stamp
from dataset import build_transform
from metrics import _per_sample_thresholds, genuineness_scores, non_hn_argmax
from model import build_model
from plots import (SERIES, plot_confusion_grid, plot_per_class_rocs,
                   plot_sample_grid, plot_score_split)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

INK = (11, 11, 11)
INK_2 = (82, 81, 78)
MUTED = (137, 135, 129)
LINE = (225, 224, 217)
PLANE = (246, 246, 243)
GOOD = (12, 163, 12)
GOOD_WASH = (236, 247, 236)
WARN = (200, 130, 0)
WARN_WASH = (253, 246, 231)
CRIT = (208, 59, 59)
CRIT_WASH = (251, 235, 235)


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
    p.add_argument("--gt", default=None, metavar="GT.json",
                   help="ground-truth JSON (GeoLabelling export: images[] "
                        "with point labels carrying class_name and "
                        "normalized pixel_x/pixel_y). Entries are matched to "
                        "input images by filename; a point counts as HIT "
                        "when an accepted window of the same class contains "
                        "it. Adds gt_results.csv, hit/miss markers on the "
                        "overlays, and GT columns to the report")
    p.add_argument("--top-n", type=int, default=10,
                   help="with --gt: size of the per-class top-N snippet "
                        "grids in the report")
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


def load_gt(path) -> dict[str, dict]:
    """GeoLabelling export -> {lowercased image name/stem/basename: entry}.

    Entry paths are absolute paths from the labelling machine, so matching
    is by filename only. Each key maps to the raw image entry.
    """
    import json
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}
    for entry in data.get("images", []):
        keys = set()
        if entry.get("name"):
            keys.add(str(entry["name"]).lower())
        if entry.get("path"):
            base = Path(str(entry["path"]).replace("\\", "/")).name
            keys.add(base.lower())
            keys.add(Path(base).stem.lower())
        for k in keys:
            lookup.setdefault(k, entry)
    return lookup


def evaluate_gt(entry: dict, iw: int, ih: int, detections: list[dict],
                classes) -> dict:
    """Score one image's detections against its ground-truth points.

    A point is HIT when an accepted window of the same class contains it;
    a detection covering no same-class point counts as a false-positive
    window. Coordinates are normalized fractions (auto-detected: values
    > 1.5 are treated as absolute pixels of original_width/height).
    """
    points = []
    for lab in entry.get("labels", []):
        x_f, y_f = lab.get("pixel_x"), lab.get("pixel_y")
        if x_f is None or y_f is None:
            continue
        if x_f > 1.5 or y_f > 1.5:  # absolute-pixel export
            x_f /= entry.get("original_width") or iw
            y_f /= entry.get("original_height") or ih
        x, y = x_f * iw, y_f * ih
        cname = str(lab.get("class_name", ""))
        known = cname in classes[:-1]  # genuine classes only
        hit, best = False, None
        for d in detections:
            if classes[d["class_index"]] != cname:
                continue
            if d["x"] <= x < d["x"] + d["w"] and d["y"] <= y < d["y"] + d["h"]:
                hit = True
                best = max(best or 0.0, d["score"])
        points.append({"id": lab.get("id"), "class": cname, "x": x, "y": y,
                       "hit": hit, "score": best, "known": known})

    fp_windows = 0
    for d in detections:
        cname = classes[d["class_index"]]
        if not any(p["class"] == cname
                   and d["x"] <= p["x"] < d["x"] + d["w"]
                   and d["y"] <= p["y"] < d["y"] + d["h"] for p in points):
            fp_windows += 1
    scored = [p for p in points if p["known"]]
    return {"points": points, "fp_windows": fp_windows,
            "n_scored": len(scored),
            "n_hit": sum(p["hit"] for p in scored),
            "unknown_classes": sorted({p["class"] for p in points
                                       if not p["known"]})}


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
                desc: str) -> tuple[list[tuple], np.ndarray, int, int]:
    """Slide over one image (HWC uint8); returns (coords, probs, w, h) for
    EVERY window — acceptance is decided by the caller, so ground-truth
    analytics can rank rejected windows too.

    Throughput notes: every window in an image has identical dimensions, so
    a whole batch of raw uint8 crops is shipped to the device in one pinned
    non-blocking transfer (~12x less PCIe traffic than fp32 224x224) and the
    resize/scale/normalize transform runs batched on the device; the forward
    pass runs under autocast on CUDA. Probabilities are computed in fp32.
    """
    ih, iw = img.shape[:2]
    w = min(args.window_width, iw)
    h = min(args.window_height, ih)
    coords = [(x, y) for y in window_positions(ih, h, args.stride_y)
              for x in window_positions(iw, w, args.stride_x)]  # row-major
    use_cuda = device.type == "cuda"
    amp = use_cuda and not getattr(args, "no_amp", False)

    chunks = []
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
        chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    probs = np.concatenate(chunks) if chunks else np.zeros((0, 1))
    return coords, probs, w, h


def detections_from(coords, probs, w: int, h: int, hn_index: int,
                    operating) -> list[dict]:
    """Accepted windows under the stored operating point."""
    if not len(coords):
        return []
    s = genuineness_scores(probs, hn_index)
    pred = non_hn_argmax(probs, hn_index)
    accepted = s >= _per_sample_thresholds(operating, pred, probs.shape[1])
    return [{"x": x, "y": y, "w": w, "h": h, "class_index": int(pred[i]),
             "score": float(s[i])}
            for i, (x, y) in enumerate(coords) if accepted[i]]


SPEC_AT_RECALL_TARGETS = (0.85, 0.90, 0.95, 0.98)


def gt_analytics(results, classes, hn_index: int, operating,
                 top_n: int) -> dict | None:
    """Cross-image ground-truth analytics over EVERY window (accepted or
    not): per-class top-N classifications, should-have-caught windows with
    confidence ranks, per-class ROC data, specificity at fixed recalls, and
    false-positive statistics."""
    K = len(classes)
    rows, fp_counts = [], []
    for ri, r in enumerate(results):
        if r.get("gt") is None or r.get("probs") is None:
            continue
        fp_counts.append(r["gt"]["fp_windows"])
        probs = r["probs"]
        if not len(probs):
            continue
        s = genuineness_scores(probs, hn_index)
        pred = non_hn_argmax(probs, hn_index)
        accepted = s >= _per_sample_thresholds(operating, pred, K)
        pts = [p for p in r["gt"]["points"] if p["known"]]
        w, h = r["wh"]
        for i, (x, y) in enumerate(r["coords"]):
            covers = {p["class"] for p in pts
                      if x <= p["x"] < x + w and y <= p["y"] < y + h}
            rows.append({"img": ri, "x": int(x), "y": int(y), "w": w, "h": h,
                         "s": float(s[i]), "pred": int(pred[i]),
                         "probs": probs[i], "accepted": bool(accepted[i]),
                         "covers": covers})
    if not rows:
        return None

    out = {"fp_counts": fp_counts, "fp_mean": float(np.mean(fp_counts)),
           "fp_std": float(np.std(fp_counts))}

    # per-class competitor pools: windows classified toward c, by P(c) desc
    pools = {}
    for c in range(K):
        if c == hn_index:
            continue
        mine = sorted((r_ for r_ in rows if r_["pred"] == c),
                      key=lambda r_: -float(r_["probs"][c]))
        pools[c] = mine
    out["topn"] = {c: pool[:top_n] for c, pool in pools.items() if pool}

    # should-have-caught: windows on a class-c GT point, ranked by P(c)
    # among the class's competitor pool
    should = {}
    for c in range(K):
        if c == hn_index:
            continue
        cname = classes[c]
        comp = np.array([float(r_["probs"][c]) for r_ in pools[c]])
        entries = []
        for r_ in rows:
            if cname not in r_["covers"]:
                continue
            pc = float(r_["probs"][c])
            entries.append({"row": r_, "pc": pc,
                            "rank": 1 + int((comp > pc).sum()),
                            "total": len(comp),
                            "in_pool": r_["pred"] == c,
                            "correct": r_["accepted"] and r_["pred"] == c})
        entries.sort(key=lambda e: e["rank"])
        if entries:
            should[c] = entries
    out["should"] = should

    # the two failure modes, as explicit lists:
    # accepted with the WRONG genuine class (cross-class confusion), most
    # confident first; and GT-covering windows rejected to hard_negative,
    # nearest the threshold first
    out["wrong_class"] = sorted(
        (r_ for r_ in rows if r_["accepted"] and r_["covers"]
         and classes[r_["pred"]] not in r_["covers"]),
        key=lambda r_: -float(r_["probs"][r_["pred"]]))
    out["rejected_tp"] = sorted(
        (r_ for r_ in rows if r_["covers"] and not r_["accepted"]),
        key=lambda r_: -r_["s"])

    # per-class ROC: score P(c), positive = window covers a class-c point
    rocs = {}
    for c in range(K):
        if c == hn_index:
            continue
        cname = classes[c]
        y_true = np.array([cname in r_["covers"] for r_ in rows])
        if y_true.any() and not y_true.all():
            sc = torch.tensor([float(r_["probs"][c]) for r_ in rows])
            yt = torch.tensor(y_true.astype(np.int64))
            fpr, tpr, _ = binary_roc(sc, yt)
            rocs[cname] = (fpr.numpy(), tpr.numpy(),
                           float(binary_auroc(sc, yt)))
    out["rocs"] = rocs

    # specificity at fixed recalls: positive = covers any known GT point,
    # score = genuineness s (the operating dimension)
    pos = np.array([bool(r_["covers"]) for r_ in rows])
    sarr = np.array([r_["s"] for r_ in rows])
    spec_tbl = []
    if pos.any() and (~pos).any():
        ps = np.sort(sarr[pos])[::-1]
        neg = sarr[~pos]
        for target in SPEC_AT_RECALL_TARGETS:
            k = min(len(ps) - 1, max(0, math.ceil(target * len(ps)) - 1))
            t = float(ps[k])
            spec_tbl.append({"target": target, "threshold": t,
                             "recall": float((sarr[pos] >= t).mean()),
                             "specificity": float((neg < t).mean())})
    out["spec_at_recall"] = spec_tbl
    out["pos_scores"] = sarr[pos]
    out["neg_scores"] = sarr[~pos]

    # window-level confusion matrices at candidate operating points:
    # rows = true class of the covered GT point (+ background), columns =
    # final call (+ hard_negative); a window covering points of two classes
    # counts once per class
    genuine = [c for c in range(K) if c != hn_index]
    gi = {classes[c]: i for i, c in enumerate(genuine)}

    def confusion_at(thr):
        m = np.zeros((len(genuine) + 1, len(genuine) + 1), dtype=int)
        for r_ in rows:
            t = thr[r_["pred"]] if isinstance(thr, dict) else thr
            call = r_["pred"] if r_["s"] >= t else hn_index
            col = len(genuine) if call == hn_index else genuine.index(call)
            if r_["covers"]:
                for cname in r_["covers"]:
                    m[gi[cname], col] += 1
            else:
                m[len(genuine), col] += 1
        return m

    if isinstance(operating, dict):
        stored_label = "stored thresholds (per-class)"
    else:
        stored_label = f"stored threshold {operating:.3f}"
    confusions = [(stored_label, confusion_at(operating))]
    for row in spec_tbl:
        confusions.append((f"{row['target']:.0%} recall "
                           f"(t={row['threshold']:.3f})",
                           confusion_at(row["threshold"])))
    out["confusions"] = confusions
    out["confusion_rows"] = [classes[c] for c in genuine] + ["background"]
    out["confusion_cols"] = [classes[c] for c in genuine] + ["hard_negative"]

    # per-class summary for the cover page
    per_class = []
    for c in range(K):
        if c == hn_index:
            continue
        cname = classes[c]
        pts = [p for r in results if r.get("gt")
               for p in r["gt"]["points"] if p["known"] and p["class"] == cname]
        per_class.append({
            "class": cname, "points": len(pts),
            "hit": sum(1 for p in pts if p["hit"]),
            "classified": len(pools[c]),
            "confused": sum(1 for r_ in out["wrong_class"] if cname in r_["covers"]),
            "rejected": sum(1 for r_ in out["rejected_tp"] if cname in r_["covers"]),
            "auc": rocs[cname][2] if cname in rocs else float("nan"),
        })
    out["per_class"] = per_class
    return out


HIT_COLOR = "#0ca30c"    # status good
MISS_COLOR = "#d03b3b"   # status critical
HIT_RGB = (12, 163, 12)
MISS_RGB = (208, 59, 59)


BORDER_RGB = {
    "good": (12, 163, 12),    # correct class, passed the threshold
    "near": (237, 161, 0),    # correct class, REJECTED by the threshold
    "bad": (208, 59, 59),     # wrong class
}


def _bordered(crop: np.ndarray, state) -> np.ndarray:
    """Copy of a window crop with a colored border: green = correct class
    and accepted, yellow = correct class but rejected by the threshold,
    red = wrong class. (Booleans map to green/red for back-compat.)"""
    if isinstance(state, bool):
        state = "good" if state else "bad"
    arr = np.ascontiguousarray(crop)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=2)
    arr = arr.copy()
    color = np.array(BORDER_RGB[state], dtype=np.uint8)
    bw = max(3, min(arr.shape[:2]) // 30)
    arr[:bw] = color
    arr[-bw:] = color
    arr[:, :bw] = color
    arr[:, -bw:] = color
    return arr


def draw_overlay(img: np.ndarray, detections: list[dict], classes,
                 hn_index: int, path: Path, gt_points=None) -> None:
    """Raw window boxes on the full image, one fixed color per class;
    ground-truth points (when given) as circles - green hit, red miss."""
    pil = Image.fromarray(img.squeeze() if img.shape[-1] == 1 else img)
    pil = pil.convert("RGB")
    draw = ImageDraw.Draw(pil)
    for d in detections:
        color = SERIES[genuine_slot(d["class_index"], hn_index)]
        draw.rectangle([d["x"], d["y"], d["x"] + d["w"] - 1, d["y"] + d["h"] - 1],
                       outline=color, width=2)
    if gt_points:
        r = max(6, min(pil.size) // 80)
        for p in gt_points:
            color = HIT_COLOR if p["hit"] else MISS_COLOR
            x, y = p["x"], p["y"]
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=3)
            draw.line([x - r, y, x + r, y], fill=color, width=1)
            draw.line([x, y - r, x, y + r], fill=color, width=1)
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


def _short(name: str, n: int = 26) -> str:
    """Middle-ellipsis for long names (filenames often differ at both ends)."""
    if len(name) <= n:
        return name
    head = (n - 3) // 2
    return name[:head] + "..." + name[-(n - 3 - head):]


def _fit(pdf, text: str, w: float) -> str:
    """Shrink a table cell's text with a middle ellipsis until it fits the
    column (fpdf cells don't wrap; long filenames would overrun neighbours)."""
    text = _txt(text)
    if pdf.get_string_width(text) <= w - 2:
        return text
    n = len(text)
    while n > 8:
        n -= 2
        candidate = _short(text, n)
        if pdf.get_string_width(candidate) <= w - 2:
            return candidate
    return _short(text, 8)


def _h1(pdf, text):
    pdf.set_font("helvetica", "B", 15)
    pdf.set_text_color(*INK)
    pdf.cell(0, 9, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2.5)


def _h2(pdf, text):
    pdf.ln(1.5)
    pdf.set_font("helvetica", "B", 11)
    pdf.set_text_color(*INK)
    pdf.cell(0, 7, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


def _para(pdf, text, size=9.5):
    pdf.set_font("helvetica", "", size)
    pdf.set_text_color(*INK_2)
    pdf.multi_cell(0, 5, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


def _kv(pdf, pairs, key_w=46):
    pdf.set_font("helvetica", "", 9.5)
    for k, v in pairs:
        pdf.set_text_color(*MUTED)
        pdf.cell(key_w, 5.4, _txt(k))
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 5.4, _txt(v), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(1)


def _box(pdf, lines, fill, edge, title=None):
    """Filled, outlined callout (verdict / warnings)."""
    pdf.set_fill_color(*fill)
    pdf.set_draw_color(*edge)
    start_y = pdf.get_y()
    pdf.set_x(pdf.l_margin)
    if title:
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*edge)
        pdf.multi_cell(0, 6, _txt(title), fill=True, new_x="LMARGIN",
                       new_y="NEXT")
    pdf.set_font("helvetica", "", 9.5)
    pdf.set_text_color(*INK)
    for line in lines:
        pdf.multi_cell(0, 5.2, _txt(line), fill=True, new_x="LMARGIN",
                       new_y="NEXT")
    pdf.rect(pdf.l_margin, start_y, pdf.epw, pdf.get_y() - start_y)
    pdf.ln(2.5)


def _table(pdf, headers, rows, widths):
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(*MUTED)
    for hd, wd in zip(headers, widths):
        pdf.cell(wd, 5.6, _txt(hd), border="B")
    pdf.ln()
    pdf.set_font("helvetica", "", 8.8)
    pdf.set_text_color(*INK)
    pdf.set_draw_color(*LINE)
    pdf.set_fill_color(*PLANE)
    for i, row in enumerate(rows):
        if pdf.get_y() > pdf.page_break_trigger - 8:
            pdf.add_page()
        for v, wd in zip(row, widths):
            pdf.cell(wd, 5.4, _fit(pdf, v, wd), border="B", fill=(i % 2 == 1))
        pdf.ln()
    pdf.ln(1.5)


def _image(pdf, path, w=None):
    w = w or pdf.epw
    with Image.open(path) as im:
        hgt = w * im.height / im.width
    max_h = pdf.eph - 6  # taller than one page: shrink to fit with slack
    if hgt > max_h:
        w = w * max_h / hgt
        hgt = max_h
    if pdf.get_y() + hgt > pdf.page_break_trigger:
        pdf.add_page()
    # suspend auto page break while placing: a tall image that grazes the
    # trigger would otherwise make fpdf insert breaks mid-placement,
    # leaving orphaned blank pages
    pdf.set_auto_page_break(False)
    pdf.image(str(path), w=w, x=pdf.l_margin)
    pdf.set_auto_page_break(True, margin=16)
    pdf.ln(2)


def build_pdf(out_dir: Path, results: list[dict], classes, hn_index: int,
              ckpt_path, args, analytics: dict | None = None,
              operating=None) -> Path:
    pdf = _Pdf(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    total_det = sum(len(r["detections"]) for r in results)
    total_windows = sum(r["n_windows"] for r in results)
    has_gt = any(r["gt"] is not None for r in results)
    if isinstance(operating, dict):
        thr_txt = (f"per-class [{min(operating.values()):.3f} .. "
                   f"{max(operating.values()):.3f}]")
        thr_scalar = None
    else:
        thr_txt = f"{operating:.4f}" if operating is not None else "-"
        thr_scalar = operating

    _h1(pdf, "Blind inference report")
    _kv(pdf, [
        ("generated", utc_stamp()),
        ("checkpoint", f"{Path(ckpt_path).name}  ({Path(ckpt_path).parent})"),
        ("classes", ", ".join(classes)),
        ("stored threshold", thr_txt),
        ("windows", f"{args.window_width}x{args.window_height} px, stride "
                    f"{args.stride_x}x{args.stride_y}, "
                    f"{'grayscale' if args.grayscale else 'RGB'} input"),
        ("images", f"{len(results)} images, {total_windows} windows, "
                   f"{total_det} detections in "
                   f"{sum(1 for r in results if r['detections'])} images"),
    ] + ([("ground truth", f"{args.gt}  "
                          f"({sum(1 for r in results if r['gt'])} of "
                          f"{len(results)} images matched)")] if has_gt else []))

    if has_gt:
        n_scored = sum(r["gt"]["n_scored"] for r in results if r["gt"])
        n_hit = sum(r["gt"]["n_hit"] for r in results if r["gt"])
        n_fp = sum(r["gt"]["fp_windows"] for r in results if r["gt"])
        rate = n_hit / max(n_scored, 1)
        fill, edge = ((GOOD_WASH, GOOD) if rate >= 0.95 else
                      (WARN_WASH, WARN) if rate >= 0.8 else (CRIT_WASH, CRIT))
        lines = [f"GT points hit at the stored operating point: {n_hit}/{n_scored} "
                 f"({rate:.1%})"]
        if analytics:
            lines.append(f"false positives per image: "
                         f"{analytics['fp_mean']:.2f} +/- {analytics['fp_std']:.2f} "
                         f"({n_fp} total across "
                         f"{len(analytics['fp_counts'])} images)")
            s95 = next((row for row in analytics["spec_at_recall"]
                        if abs(row["target"] - 0.95) < 1e-9), None)
            if s95:
                lines.append(f"specificity at 95% window recall: "
                             f"{s95['specificity']:.4f} at threshold "
                             f"{s95['threshold']:.3f}"
                             + (f" (stored threshold {thr_scalar:.3f})"
                                if thr_scalar is not None else ""))
            lines.append(f"failure modes: {len(analytics['wrong_class'])} "
                         "accepted as the wrong class, "
                         f"{len(analytics['rejected_tp'])} GT windows rejected "
                         "to hard_negative")
        _box(pdf, lines, fill, edge, title="Verdict")

        if analytics and analytics.get("per_class"):
            _h2(pdf, "Per class")
            _table(pdf, ["class", "GT points", "hit", "hit rate", "classified",
                         "confused", "rejected", "AUC"],
                   [[row["class"], row["points"], row["hit"],
                     f"{row['hit'] / row['points']:.0%}" if row["points"] else "-",
                     row["classified"], row["confused"], row["rejected"],
                     f"{row['auc']:.3f}" if row["auc"] == row["auc"] else "-"]
                    for row in analytics["per_class"]],
                   widths=[pdf.epw * v for v in
                           (0.2, 0.11, 0.09, 0.11, 0.13, 0.12, 0.12, 0.12)])
            _para(pdf, "classified = windows argmax-routed to the class; "
                       "confused = accepted as another genuine class while on "
                       "this class's point; rejected = on this class's point "
                       "but called hard_negative; AUC = one-vs-rest over all "
                       "windows.", size=8)

    if not has_gt:
        _table(pdf, ["image", "size", "windows", "detections", "classes found"],
               [[r["path"].name, f"{r['size'][0]}x{r['size'][1]}", r["n_windows"],
                 len(r["detections"]),
                 ", ".join(sorted({classes[d["class_index"]]
                                   for d in r["detections"]})) or "-"]
                for r in results],
               widths=[pdf.epw * w for w in (0.34, 0.14, 0.12, 0.13, 0.27)])

    if analytics:
        assets = out_dir / "assets"
        img_cache: dict = {}

        def crop_of(row) -> np.ndarray:
            ri = row["img"]
            if ri not in img_cache:
                pil = Image.open(results[ri]["path"]).convert(
                    "L" if args.grayscale else "RGB")
                arr = np.asarray(pil)
                img_cache[ri] = arr[:, :, None] if arr.ndim == 2 else arr
            a = img_cache[ri]
            return a[row["y"]:row["y"] + row["h"], row["x"]:row["x"] + row["w"]]

        pdf.add_page()
        _h1(pdf, "Operating point")
        _para(pdf, "How well the genuineness score separates windows that sit "
                   "on a labelled point from everything else - and where the "
                   "stored threshold falls relative to the thresholds that "
                   "would achieve fixed recalls.")
        if len(analytics["pos_scores"]) and len(analytics["neg_scores"]):
            split_png = assets / "gt_score_split.png"
            plot_score_split(
                analytics["pos_scores"], analytics["neg_scores"], split_png,
                threshold=thr_scalar,
                marks=[(f"{row['target']:.0%} recall", row["threshold"])
                       for row in analytics["spec_at_recall"]])
            _image(pdf, split_png)
        if analytics["spec_at_recall"]:
            _h2(pdf, "Specificity at fixed recall")
            _table(pdf, ["target recall", "achieved recall", "specificity",
                         "threshold"],
                   [[f"{row['target']:.0%}", f"{row['recall']:.4f}",
                     f"{row['specificity']:.4f}", f"{row['threshold']:.4f}"]
                    for row in analytics["spec_at_recall"]],
                   widths=[pdf.epw * v for v in (0.25, 0.25, 0.25, 0.25)])
            _para(pdf, "positive = window covers a known GT point, score = "
                       "genuineness s; each row's threshold is directly "
                       "comparable to the stored operating threshold.", size=8)
        if analytics["rocs"]:
            pdf.add_page()
            _h1(pdf, "Per-class ROC")
            roc_png = assets / "gt_roc.png"
            dropped = plot_per_class_rocs(analytics["rocs"], roc_png)
            _image(pdf, roc_png, w=pdf.epw * 0.85)
            _para(pdf, "One-vs-rest over all windows: score = P(class), "
                       "positive = window covers a GT point of the class."
                       + (f" Omitted: {', '.join(dropped)}." if dropped else ""),
                  size=8.5)

        if analytics.get("confusions"):
            pdf.add_page()
            _h1(pdf, "Confusion at the stored operating point")
            _para(pdf, "Window-level confusion matrix: rows are the true "
                       "class of the GT point a window covers (plus a "
                       "background row for windows covering none), columns "
                       "are the final call after the threshold. A window "
                       "covering points of two classes counts once per "
                       "class.")
            stored_png = assets / "gt_confusion_stored.png"
            plot_confusion_grid(analytics["confusions"][:1],
                                analytics["confusion_rows"],
                                analytics["confusion_cols"], stored_png,
                                ncols=1, scale=1.6)
            _image(pdf, stored_png)

            if len(analytics["confusions"]) > 1:
                pdf.add_page()
                _h1(pdf, "Confusion at fixed-recall thresholds")
                _para(pdf, "The same matrix re-applied at the thresholds "
                           "that achieve 85/90/95/98% window recall - watch "
                           "the hard_negative column drain into the diagonal "
                           "as the threshold drops. Misses that survive even "
                           "the lowest threshold are classifier confusion, "
                           "not the operating point.")
                recall_png = assets / "gt_confusions_recall.png"
                plot_confusion_grid(analytics["confusions"][1:],
                                    analytics["confusion_rows"],
                                    analytics["confusion_cols"], recall_png,
                                    ncols=2)
                _image(pdf, recall_png)

        def thr_for(c):
            return operating[c] if isinstance(operating, dict) else operating

        def cutoff_txt(c):
            thr = thr_for(c)
            return ("" if thr is None else
                    f" Acceptance cutoff for '{classes[c]}': s >= {thr:.3f}"
                    + (" (per-class)" if isinstance(operating, dict)
                       else " (global)") + ".")

        for c, pool in sorted(analytics["topn"].items()):
            pdf.add_page()
            cname = classes[c]
            _h1(pdf, f"Top {len(pool)} '{cname}' classifications")
            _para(pdf, f"The model's most confident '{cname}' calls across "
                       "all images - a healthy model shows a solid green row."
                       + cutoff_txt(c), size=8.5)
            crops = [_bordered(crop_of(r_),
                               "good" if r_["accepted"] and cname in r_["covers"]
                               else "near" if cname in r_["covers"]
                               else "bad")
                     for r_ in pool]
            caps = [f"{_short(results[r_['img']]['path'].stem)}\n"
                    f"P={float(r_['probs'][c]):.3f} s={r_['s']:.3f}"
                    + (" ACC" if r_["accepted"] else "") for r_ in pool]
            png = assets / f"gt_top_{c}.png"
            plot_sample_grid(crops, caps,
                             f"highest P({cname}) windows across all images",
                             png, ncols=5)
            _image(pdf, png)
            _para(pdf, "green border = accepted AND on a ground-truth "
                       f"{cname} point; yellow = on a {cname} point but "
                       "rejected by the threshold; red = not on a "
                       f"{cname} point.", size=8.5)

        for c, entries in sorted(analytics["should"].items()):
            cname = classes[c]
            total = entries[0]["total"] if entries else 0
            for start in range(0, len(entries), 20):
                chunk = entries[start:start + 20]
                pdf.add_page()
                _h1(pdf, f"'{cname}' windows on GT points"
                         + (f" ({start + 1}-{start + len(chunk)} of "
                            f"{len(entries)})" if len(entries) > 20 else ""))
                if start == 0:
                    _para(pdf, "Every window that should have been called "
                               f"'{cname}', ranked by confidence among all "
                               f"windows classified '{cname}' - red near the "
                               "top of the ranking means confident misses; "
                               "red near the bottom means the threshold, not "
                               "the classifier. Windows whose argmax was "
                               "another class are not in that pool; they "
                               "follow it, labelled with the class they were "
                               "called instead." + cutoff_txt(c)
                               + " Green = s at or above the cutoff AND "
                               f"argmax {cname}.", size=8.5)
                crops = [_bordered(crop_of(e["row"]),
                                   "good" if e["correct"]
                                   else "near" if e["in_pool"]
                                   else "bad")
                         for e in chunk]
                caps = [(f"rank {e['rank']}/{e['total']}"
                         if e["in_pool"] else
                         "not in pool\nargmax "
                         f"{classes[e['row']['pred']]}")
                        + f"\nP={e['pc']:.3f} s={e['row']['s']:.3f}"
                        for e in chunk]
                png = assets / f"gt_should_{c}_{start}.png"
                plot_sample_grid(
                    crops, caps,
                    f"every window covering a {cname} GT point, ranked by "
                    f"P({cname}) among the {total} windows classified {cname}",
                    png, ncols=5)
                _image(pdf, png)
                _para(pdf, "green border = passed the threshold and was "
                           f"classified {cname}; yellow = classified {cname} "
                           "but rejected by the threshold (an operating-point "
                           "miss); red = classified as another class (a "
                           "classifier miss).", size=8.5)

        wrong = analytics["wrong_class"]
        for start in range(0, len(wrong), 16):
            chunk = wrong[start:start + 16]
            pdf.add_page()
            _h1(pdf, "Accepted as the WRONG class"
                     + (f" ({start + 1}-{start + len(chunk)} of {len(wrong)})"
                        if len(wrong) > 16 else ""))
            if start == 0:
                _para(pdf, "Cross-class confusions: accepted windows sitting "
                           "on a GT point of a different genuine class, most "
                           "confident first. Caption: called X, true Y, P(X), "
                           "s (accepted, so s is at or above the cutoff for X).",
                      size=8.5)
            crops = [_bordered(crop_of(r_), False) for r_ in chunk]
            caps = [f"called {classes[r_['pred']]}\n"
                    f"true {'/'.join(sorted(r_['covers']))}\n"
                    f"P={float(r_['probs'][r_['pred']]):.3f} s={r_['s']:.3f}"
                    for r_ in chunk]
            png = assets / f"gt_wrong_{start}.png"
            plot_sample_grid(crops, caps,
                             "cross-class confusions: accepted windows on a "
                             "GT point of a DIFFERENT genuine class, most "
                             "confident first", png, ncols=5)
            _image(pdf, png)

        rej = analytics["rejected_tp"]
        for c in range(len(classes)):
            if c == hn_index:
                continue
            cname = classes[c]
            mine = [r_ for r_ in rej if cname in r_["covers"]]
            for start in range(0, len(mine), 16):
                chunk = mine[start:start + 16]
                pdf.add_page()
                _h1(pdf, f"'{cname}' on a GT point but called hard_negative"
                         + (f" ({start + 1}-{start + len(chunk)} of "
                            f"{len(mine)})" if len(mine) > 16 else ""))
                if start == 0:
                    _para(pdf, f"Windows on a {cname} GT point that the "
                               "threshold rejected, nearest the operating "
                               "point first. Yellow border = argmax was "
                               f"{cname} (an operating-point miss: correct "
                               "class, score just under the cutoff); red = "
                               "argmax was another class (a classifier miss). "
                               "A window covering points of two classes "
                               "appears under each."
                               + cutoff_txt(c), size=8.5)
                crops = [_bordered(crop_of(r_),
                                   "near" if r_["pred"] == c else "bad")
                         for r_ in chunk]
                caps = [f"true {'/'.join(sorted(r_['covers']))}\n"
                        f"argmax {classes[r_['pred']]}\ns={r_['s']:.3f}"
                        for r_ in chunk]
                png = assets / f"gt_rejected_{c}_{start}.png"
                plot_sample_grid(crops, caps,
                                 f"rejected windows covering {cname} points, "
                                 "nearest the operating point first",
                                 png, ncols=5)
                _image(pdf, png)

    if has_gt:
        pdf.add_page()
        _h1(pdf, "Per image")
        if analytics:
            _para(pdf, "Every input image with its window, detection, GT-hit "
                       "and false-positive counts. False positives per "
                       f"image: mean {analytics['fp_mean']:.2f} +/- "
                       f"{analytics['fp_std']:.2f}.")
        _table(pdf, ["image", "size", "windows", "detections", "gt hit", "FP win"],
               [[r["path"].name, f"{r['size'][0]}x{r['size'][1]}",
                 r["n_windows"], len(r["detections"]),
                 (f"{r['gt']['n_hit']}/{r['gt']['n_scored']}"
                  if r["gt"] else "no gt"),
                 r["gt"]["fp_windows"] if r["gt"] else "-"]
                for r in results],
               widths=[pdf.epw * w for w in (0.32, 0.14, 0.13, 0.15, 0.14, 0.12)])

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

    gt_lookup = load_gt(args.gt) if args.gt else None
    gt_matched = 0

    results = []
    for path in gather_images(args.images):
        pil = Image.open(path).convert("L" if args.grayscale else "RGB")
        img = np.asarray(pil)
        if img.ndim == 2:
            img = img[:, :, None]
        coords, probs, w, h = infer_image(img, model, transform, device,
                                          args, desc=path.name)
        detections = detections_from(coords, probs, w, h, hn_index, operating)
        r = {"path": path, "size": pil.size, "n_windows": len(coords),
             "detections": detections, "overlay": None, "gt": None}

        if gt_lookup is not None:
            entry = gt_lookup.get(path.name.lower()) or \
                gt_lookup.get(path.stem.lower())
            if entry is not None:
                r["gt"] = evaluate_gt(entry, pil.size[0], pil.size[1],
                                      detections, classes)
                gt_matched += 1
                # keep every window's probabilities for the GT analytics
                r["coords"], r["probs"], r["wh"] = coords, probs, (w, h)

        misses = (r["gt"] is not None
                  and r["gt"]["n_hit"] < r["gt"]["n_scored"])
        if detections or misses:
            r["overlay"] = assets / f"{path.stem}_overlay.png"
            draw_overlay(img, detections, classes, hn_index, r["overlay"],
                         gt_points=r["gt"]["points"] if r["gt"] else None)
        results.append(r)
        gt_txt = ""
        if r["gt"] is not None:
            gt_txt = (f", gt {r['gt']['n_hit']}/{r['gt']['n_scored']} hit, "
                      f"{r['gt']['fp_windows']} FP windows")
            if r["gt"]["unknown_classes"]:
                gt_txt += (" (unknown classes: "
                           + ", ".join(r["gt"]["unknown_classes"]) + ")")
        print(f"{path.name}: {len(coords)} windows, {len(detections)} detections"
              + gt_txt + ("  <- flagged" if (detections or misses) else ""))

    analytics = None
    if gt_lookup is not None:
        print(f"ground truth: {gt_matched}/{len(results)} input images "
              f"matched in {args.gt}")
        analytics = gt_analytics(results, classes, hn_index, operating,
                                 args.top_n)
        if analytics:
            counts = analytics["fp_counts"]
            if len(counts) > 20:
                nz = sum(1 for v in counts if v)
                detail = (f"{nz} of {len(counts)} images with FPs, "
                          f"max {max(counts)}")
            else:
                detail = f"counts: {counts}"
            print(f"false positives per image: mean {analytics['fp_mean']:.2f}, "
                  f"std {analytics['fp_std']:.2f} ({detail})")
            for row in analytics["spec_at_recall"]:
                print(f"  specificity at {row['target']:.0%} recall: "
                      f"{row['specificity']:.4f} (threshold {row['threshold']:.4f})")

    with open(out_dir / "detections.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x", "y", "w", "h", "class", "score"])
        for r in results:
            for d in r["detections"]:
                writer.writerow([r["path"], d["x"], d["y"], d["w"], d["h"],
                                 classes[d["class_index"]], f"{d['score']:.6f}"])

    if gt_lookup is not None:
        with open(out_dir / "gt_results.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label_id", "class", "x", "y", "hit",
                             "best_score", "known_class"])
            for r in results:
                if r["gt"] is None:
                    continue
                for p in r["gt"]["points"]:
                    writer.writerow([
                        r["path"], p["id"], p["class"], f"{p['x']:.1f}",
                        f"{p['y']:.1f}", int(p["hit"]),
                        "" if p["score"] is None else f"{p['score']:.6f}",
                        int(p["known"])])

    if not args.no_report:
        pdf_path = build_pdf(out_dir, results, classes, hn_index, ckpt_path,
                             args, analytics=analytics, operating=operating)
        print(f"report: {pdf_path}")
    print(f"outputs written to {out_dir}")


if __name__ == "__main__":
    main()
