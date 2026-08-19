"""End-of-training PDF report (fpdf2).

Collects everything meaningful a finished run produced — config, dataset
shape, training history, smart-controller/rescue history, a fresh inference
pass with the stored operating point, calibration, confusion, and thumbnail
grids of the actual problem samples — into <run_dir>/report.pdf, with an
auto-generated warnings section for the conditions worth taking seriously
(classes still under rescue, unreachable targets, pressure ceilings, floor
binding, fallback classes, poor calibration).

train.py generates it automatically at the end of a run (--no-report to
skip; --report-test to use the test split instead of validation). Standalone
regeneration for any past run:

    python report.py runs/exp1 data.h5 [--split 1] [--thumbs 16]
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from fpdf import FPDF
from torch.utils.data import DataLoader

from dataset import SPLIT_NAMES, SPLIT_TRAIN, SPLIT_VAL, H5SnippetDataset, validate_h5
from metrics import (apply_threshold, calibration_bins, collect_probs,
                     final_prediction, genuine_vs_hn_roc, genuineness_scores,
                     non_hn_argmax)
from model import build_model
from plots import (plot_calibration, plot_confusion, plot_controller_timeline,
                   plot_genuine_vs_hn_roc, plot_history,
                   plot_per_class_recall_history, plot_sample_grid)

# Light-theme ink/accent palette shared with plots.py, as RGB tuples.
INK = (11, 11, 11)
INK_2 = (82, 81, 78)
MUTED = (137, 135, 129)
LINE = (225, 224, 217)
ACCENT = (42, 120, 214)
ACCENT_WASH = (238, 244, 252)
WARN = (200, 130, 0)
WARN_WASH = (253, 246, 231)
GOOD = (12, 100, 12)

ECE_WARN = 0.10


def _txt(s) -> str:
    """Core PDF fonts are latin-1; degrade anything else."""
    return str(s).encode("latin-1", "replace").decode("latin-1")


# --------------------------------------------------------------------------
# data gathering
# --------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _reconstruct_cli(config: dict) -> str:
    parts = [f"train.py {config.get('h5', '?')}"]
    skip = {"h5"}
    for k, v in config.items():
        if k in skip or v in (None, False):
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            parts.append(flag)
        elif isinstance(v, list):
            parts.extend(f"{flag} {item}" for item in v)
        else:
            parts.append(f"{flag} {v}")
    return " ".join(parts)


def _collect_warnings(best: dict, last: dict, rows: list[dict], classes,
                      ece: float) -> list[str]:
    """The take-this-seriously list; each entry includes the recommended
    action."""
    warnings: list[str] = []
    cs = last.get("controller_state") or {}
    r_alphas = cs.get("rescue_alphas") or {}
    r_repeats = cs.get("rescue_repeats") or {}
    if r_alphas or r_repeats:
        names = ", ".join(classes[c] for c in sorted(set(r_alphas) | set(r_repeats)))
        warnings.append(
            f"Still under rescue when training ended: {names}. Reweighting has "
            "hit its limit for these classes - collect more data for them; "
            "rescue redistributes gradient, it cannot manufacture information.")

    if rows and not any(r["target_met"] == "1" for r in rows):
        best_ceiling = max(float(r["max_recall"]) for r in rows)
        target = (best.get("config") or {}).get("target_recall")
        warnings.append(
            f"The recall target was never met in any epoch (best achievable "
            f"ceiling: {best_ceiling:.4f} vs target {target}). If the ceiling "
            "sits below the target, no threshold can fix it - lower the target "
            "or address the misclassified classes.")

    if any(r.get("event", "").startswith("ceiling") for r in rows):
        warnings.append(
            "The smart controller hit its pressure ceiling: full hard-negative "
            "pressure was not sustainable at the recall target. The model "
            "trained at reduced pressure; specificity may be below potential.")

    vm = best.get("val_metrics") or {}
    min_thr = best.get("min_threshold", 0.0) or 0.0
    if min_thr > 0 and not vm.get("target_met", True) and \
            vm.get("threshold") == min_thr:
        warnings.append(
            f"The best checkpoint operates AT the min-threshold floor "
            f"({min_thr:.3f}) with the recall target unmet - the floor is "
            "binding. Recall reported is the best achievable above the floor.")

    fb = vm.get("fallback_classes") or []
    if fb:
        warnings.append(
            "Per-class thresholds fell back to the global value for: "
            + ", ".join(classes[c] for c in fb)
            + " (too few predicted validation samples to calibrate - more "
            "validation data would give these classes their own operating "
            "points).")

    if ece > ECE_WARN:
        warnings.append(
            f"Genuineness-score ECE is {ece:.3f} (> {ECE_WARN}): the score is "
            "not calibrated as a probability. Thresholds remain valid (they "
            "are rank-based), but do not read the threshold value as "
            "P(genuine).")
    return warnings


def _hard_sample_grids(h5_path, split, probs, labels, operating, hn_index,
                       classes, best, assets: Path, thumbs: int) -> list[tuple]:
    """Thumbnail grids of the actual problem samples. Returns
    [(png_path, caption), ...]."""
    grids: list[tuple] = []
    ds = H5SnippetDataset(h5_path, split)
    scores = genuineness_scores(probs, hn_index)
    pred = non_hn_argmax(probs, hn_index)
    genuine = labels != hn_index

    if isinstance(operating, dict):
        thr_vec = np.array([operating[int(p)] for p in pred])
    else:
        thr_vec = np.full(len(pred), float(operating))
    recalled = (scores >= thr_vec) & (pred == labels) & genuine

    with h5py.File(h5_path, "r") as f:
        def thumbs_for(positions):
            return [f["images"][int(ds.indices[p])] for p in positions]

        # hardest genuine: not-recalled first (most confidently rejected
        # first), padded with the lowest-scoring recalled ones
        gen_pos = np.flatnonzero(genuine)
        missed = gen_pos[~recalled[gen_pos]]
        missed = missed[np.argsort(scores[missed])]
        rest = gen_pos[recalled[gen_pos]]
        rest = rest[np.argsort(scores[rest])]
        chosen = np.concatenate([missed, rest])[:thumbs]
        if len(chosen):
            caps = [f"{classes[labels[p]]} -> {classes[pred[p]]}\n"
                    f"s={scores[p]:.3f}" + ("" if recalled[p] else "  MISSED")
                    for p in chosen]
            path = assets / "hard_genuine.png"
            plot_sample_grid(thumbs_for(chosen), caps,
                             "Hardest genuine samples (lowest scores; MISSED = "
                             "not recalled at the operating point)", path)
            grids.append((path,
                          "What the recall failures actually look like - the "
                          "candidates for relabeling or collecting more "
                          "examples like them."))

        # most-fooling hard negatives in this split, by genuineness score
        hn_pos = np.flatnonzero(~genuine)
        if len(hn_pos):
            top = hn_pos[np.argsort(-scores[hn_pos])][:thumbs]
            caps = [f"-> {classes[pred[p]]}  s={scores[p]:.3f}" for p in top]
            path = assets / "fooling_hn.png"
            plot_sample_grid(thumbs_for(top), caps,
                             f"Most-fooling hard negatives ({SPLIT_NAMES[split]} "
                             "split, highest genuineness scores)", path)
            grids.append((path,
                          "Hard negatives the model most wants to accept - "
                          "the profile of negatives worth mining more of."))

        # training-split hard negatives the miner found hardest (EMA loss)
        ms = best.get("miner_state")
        if ms is not None:
            train_ds = H5SnippetDataset(h5_path, SPLIT_TRAIN)
            m_scores = np.asarray(ms["scores"])
            seen = np.asarray(ms["seen"], dtype=bool)
            is_hn = train_ds.labels == hn_index
            cand = np.flatnonzero(is_hn & seen)
            if len(cand):
                top = cand[np.argsort(-m_scores[cand])][:thumbs]
                caps = [f"EMA loss {m_scores[p]:.2f}" for p in top]
                imgs = [f["images"][int(train_ds.indices[p])] for p in top]
                path = assets / "miner_hn.png"
                plot_sample_grid(imgs, caps,
                                 "Persistently hard training negatives "
                                 "(miner's EMA loss)", path)
                grids.append((path,
                              "The training hard negatives that stayed "
                              "difficult across epochs, per the mining "
                              "tracker."))
    return grids


# --------------------------------------------------------------------------
# PDF assembly
# --------------------------------------------------------------------------

class _Pdf(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("helvetica", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, f"mic training report - page {self.page_no()}/{{nb}}",
                  align="C")


def _h1(pdf, text):
    pdf.set_font("helvetica", "B", 15)
    pdf.set_text_color(*INK)
    pdf.cell(0, 9, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(*LINE)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2.5)


def _h2(pdf, text):
    pdf.ln(1.5)
    pdf.set_font("helvetica", "B", 11.5)
    pdf.set_text_color(*INK)
    pdf.cell(0, 7, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


def _para(pdf, text, color=INK_2, size=9.5):
    pdf.set_font("helvetica", "", size)
    pdf.set_text_color(*color)
    pdf.multi_cell(0, 5, _txt(text), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(0.5)


def _kv(pdf, pairs, key_w=58):
    pdf.set_font("helvetica", "", 9.5)
    for k, v in pairs:
        pdf.set_text_color(*MUTED)
        pdf.cell(key_w, 5.4, _txt(k))
        pdf.set_text_color(*INK)
        pdf.multi_cell(0, 5.4, _txt(v), new_x="LMARGIN", new_y="NEXT")


def _table(pdf, headers, rows, widths=None):
    epw = pdf.epw
    widths = widths or [epw / len(headers)] * len(headers)
    pdf.set_font("helvetica", "B", 8.5)
    pdf.set_text_color(*MUTED)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5.6, _txt(h), border="B")
    pdf.ln()
    pdf.set_font("helvetica", "", 8.8)
    pdf.set_text_color(*INK)
    pdf.set_draw_color(*LINE)
    for row in rows:
        if pdf.get_y() > pdf.page_break_trigger - 8:
            pdf.add_page()
        for v, w in zip(row, widths):
            pdf.cell(w, 5.4, _txt(v), border="B")
        pdf.ln()
    pdf.ln(1.5)


def _image(pdf, path, w=None):
    from PIL import Image
    w = w or pdf.epw
    with Image.open(path) as im:
        h = w * im.height / im.width
    if pdf.get_y() + h > pdf.page_break_trigger:
        pdf.add_page()
    pdf.image(str(path), w=w, x=pdf.l_margin)
    pdf.ln(2)


def _box(pdf, lines, fill, edge, title=None):
    pdf.set_fill_color(*fill)
    pdf.set_draw_color(*edge)
    start_y = pdf.get_y()
    pdf.set_x(pdf.l_margin)
    if title:
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*edge)
        pdf.multi_cell(0, 6, _txt(title), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9.5)
    pdf.set_text_color(*INK)
    for line in lines:
        pdf.multi_cell(0, 5.2, _txt(line), fill=True, new_x="LMARGIN", new_y="NEXT")
    pdf.rect(pdf.l_margin, start_y, pdf.epw, pdf.get_y() - start_y)
    pdf.ln(2.5)


# --------------------------------------------------------------------------
# report builder
# --------------------------------------------------------------------------

def build_report(run_dir, h5_path, split: int = SPLIT_VAL, thumbs: int = 16,
                 device: str | None = None, progress: bool = True) -> Path:
    run_dir = Path(run_dir)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    best = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    last_path = run_dir / "last.pt"
    last = (torch.load(last_path, map_location=dev, weights_only=False)
            if last_path.exists() else best)
    config = best.get("config") or {}
    classes = best["classes"]
    hn_index = best["hard_negative_index"]
    recall_agg = best.get("recall_agg", "macro")
    per_class_mode = (best.get("threshold_mode") == "per-class"
                      and best.get("class_thresholds"))
    operating = best["class_thresholds"] if per_class_mode else best["threshold"]

    rows = _read_csv(run_dir / "metrics.csv")
    class_rows = _read_csv(run_dir / "class_thresholds.csv")
    summary = validate_h5(str(h5_path))

    # ---- end-of-training inference with the stored operating point --------
    model = build_model(best["arch"], len(classes), pretrained=False).to(dev)
    model.load_state_dict(best["model_state"])
    ds = H5SnippetDataset(str(h5_path), split, imagenet_norm=best["imagenet_norm"])
    loader = DataLoader(ds, batch_size=64, pin_memory=dev.type == "cuda")
    probs, labels = collect_probs(model, loader, dev, desc="report inference",
                                  progress=progress)
    res = apply_threshold(probs, labels, hn_index, operating, agg=recall_agg)
    pred = final_prediction(probs, hn_index, operating)
    cm = np.bincount(labels * len(classes) + pred,
                     minlength=len(classes) ** 2).reshape(len(classes), -1)
    cal = calibration_bins(probs, labels, hn_index)
    try:
        fpr, tpr, auroc = genuine_vs_hn_roc(probs, labels, hn_index)
    except ValueError:
        fpr = tpr = None
        auroc = float("nan")

    # ---- assets ------------------------------------------------------------
    assets = run_dir / "report_assets"
    assets.mkdir(exist_ok=True)
    charts: dict[str, Path] = {}
    if rows:
        charts["history"] = assets / "history.png"
        plot_history(run_dir / "metrics.csv", charts["history"])
        if any(r.get("pressure") for r in rows):
            charts["timeline"] = assets / "timeline.png"
            plot_controller_timeline(run_dir / "metrics.csv", charts["timeline"])
    if class_rows:
        charts["per_class"] = assets / "per_class_recall.png"
        plot_per_class_recall_history(run_dir / "class_thresholds.csv",
                                      charts["per_class"])
    if fpr is not None:
        charts["roc"] = assets / "roc.png"
        plot_genuine_vs_hn_roc(fpr, tpr, auroc, charts["roc"], operating_point=res)
    charts["calibration"] = assets / "calibration.png"
    plot_calibration(cal, charts["calibration"],
                     threshold=None if per_class_mode else best["threshold"])
    charts["confusion"] = assets / "confusion.png"
    plot_confusion(cm, classes, charts["confusion"])
    grids = _hard_sample_grids(str(h5_path), split, probs, labels, operating,
                               hn_index, classes, best, assets, thumbs)

    warnings = _collect_warnings(best, last, rows, classes, cal["ece"])

    # ---- PDF ---------------------------------------------------------------
    pdf = _Pdf(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()

    _h1(pdf, "Training report")
    _kv(pdf, [
        ("generated", datetime.now().strftime("%Y-%m-%d %H:%M")),
        ("run directory", str(run_dir)),
        ("dataset", str(h5_path)),
        ("classes", ", ".join(classes)),
        ("architecture", best["arch"]),
        ("best checkpoint", f"epoch {best['epoch']}"
                            + (f" (last epoch {last['epoch']})"
                               if last is not best else "")),
        ("report split", SPLIT_NAMES[split]),
        ("command", _reconstruct_cli(config)),
    ])
    pdf.ln(2)

    # verdict
    vm = best.get("val_metrics") or {}
    met = bool(vm.get("target_met"))
    verdict = [
        f"{'TARGET MET' if met else 'TARGET NOT MET'} - "
        f"target {config.get('target_recall')} ({recall_agg}) at best epoch "
        f"{best['epoch']}",
        f"{SPLIT_NAMES[split]} split at the stored operating point: "
        f"{recall_agg} recall {res['recall']:.4f}, HN specificity "
        f"{res['specificity']:.4f}, TPR {res['tpr']:.4f}, AUROC {auroc:.4f}, "
        f"ECE {cal['ece']:.4f}",
        ("per-class thresholds "
         f"[{min(operating.values()):.4f} .. {max(operating.values()):.4f}]"
         if per_class_mode else f"global threshold {best['threshold']:.6f}")
        + f", floor {best.get('min_threshold', 0.0):.3f}",
    ]
    target = config.get("target_recall")
    if met and target is not None and res["recall"] < float(target):
        verdict.append(
            "note: this re-inference lands below the target the sweep met "
            "during training - borderline samples near the threshold can flip "
            "between runs; with small validation classes one sample moves the "
            "aggregate visibly.")
    _box(pdf, verdict, ACCENT_WASH, ACCENT if met else WARN, title="Verdict")

    _h2(pdf, "Warnings & recommendations")
    if warnings:
        _box(pdf, [f"- {w}" for w in warnings], WARN_WASH, WARN)
    else:
        pdf.set_text_color(*GOOD)
        pdf.set_font("helvetica", "B", 9.5)
        pdf.cell(0, 6, "No issues detected.", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(1)

    _h2(pdf, "Dataset")
    _table(pdf, ["split", "genuine", "hard negatives", "imbalance"],
           [[name, c["genuine"], c["hard_negative"],
             f"{c['hard_negative'] / max(c['genuine'], 1):.1f}x"]
            for name, c in summary["counts"].items()],
           widths=[pdf.epw * w for w in (0.25, 0.25, 0.25, 0.25)])

    _h2(pdf, f"Final model on the {SPLIT_NAMES[split]} split")
    cs_best = best.get("controller_state") or {}
    rescue_at_best = set(cs_best.get("rescue_alphas") or {}) | \
        set(cs_best.get("rescue_repeats") or {})
    fb = set(vm.get("fallback_classes") or [])
    per_rows = []
    for c, r in sorted(res["per_class_recall"].items()):
        thr_c = operating[c] if per_class_mode else best["threshold"]
        flags = " ".join(filter(None, [
            "[fallback]" if c in fb else "",
            "[rescued]" if c in rescue_at_best else ""]))
        per_rows.append([classes[c], f"{r:.4f}", f"{thr_c:.4f}",
                         int((labels == c).sum()), flags])
    _table(pdf, ["class", "recall", "threshold", "n", "flags"], per_rows,
           widths=[pdf.epw * w for w in (0.3, 0.17, 0.19, 0.12, 0.22)])

    for key in ("roc", "calibration", "confusion"):
        if key in charts:
            _image(pdf, charts[key], w=pdf.epw * 0.72)

    pdf.add_page()
    _h1(pdf, "Training history")
    for key in ("history", "per_class", "timeline"):
        if key in charts:
            _image(pdf, charts[key])

    # smart-mode narrative
    events = [r for r in rows if r.get("event")]
    if events:
        _h2(pdf, "Controller decisions")
        _table(pdf, ["epoch", "cycle", "event", "pressure", "recall", "specificity"],
               [[r["epoch"], r["cycle"], r["event"], r["pressure"],
                 f"{float(r['recall']):.3f}", f"{float(r['specificity']):.3f}"]
                for r in events],
               widths=[pdf.epw * w for w in (0.1, 0.1, 0.36, 0.14, 0.15, 0.15)])
        snaps = (last.get("controller_state") or {}).get("snapshots") or []
        if snaps:
            _para(pdf, "Snapshot archive (top cycle checkpoints): " + "; ".join(
                f"{Path(p).name} (spec {k[1]:.3f}, recall {k[2]:.3f})"
                for k, p in snaps))

    if grids:
        pdf.add_page()
        _h1(pdf, "Problem samples")
        for path, caption in grids:
            _image(pdf, path)
            _para(pdf, caption, size=8.5)

    pdf.add_page()
    _h1(pdf, "Configuration")
    _table(pdf, ["option", "value"],
           sorted((k, str(v)) for k, v in config.items()),
           widths=[pdf.epw * 0.4, pdf.epw * 0.6])

    out_path = run_dir / "report.pdf"
    pdf.output(str(out_path))
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_dir", help="training output directory (with best.pt)")
    p.add_argument("h5", help="dataset .h5 file")
    p.add_argument("--split", type=int, default=SPLIT_VAL,
                   choices=sorted(SPLIT_NAMES),
                   help="split for the inference pass (default: validation)")
    p.add_argument("--thumbs", type=int, default=16,
                   help="thumbnails per problem-sample grid")
    p.add_argument("--device", default=None)
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args()
    path = build_report(args.run_dir, args.h5, split=args.split,
                        thumbs=args.thumbs, device=args.device,
                        progress=not args.no_progress)
    print(f"report written to {path}")


if __name__ == "__main__":
    main()
