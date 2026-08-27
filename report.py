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
import math
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from fpdf import FPDF
from torch.utils.data import DataLoader

from checkpoints import find_checkpoint, utc_stamp
from dataset import SPLIT_NAMES, SPLIT_TRAIN, SPLIT_VAL, H5SnippetDataset, validate_h5
from metrics import (apply_threshold, calibration_bins, collect_probs,
                     final_prediction, genuine_vs_hn_roc, genuineness_scores,
                     non_hn_argmax, per_class_ovr_roc)
from model import build_model
from plots import (plot_calibration, plot_confusion, plot_confusion_grid,
                   plot_controller_timeline, plot_genuine_vs_hn_roc,
                   plot_history, plot_per_class_recall_history,
                   plot_per_class_rocs, plot_sample_grid, plot_score_split)

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
SPEC_AT_RECALL_TARGETS = (0.85, 0.90, 0.95, 0.98)
# annotated confusion grids get unwieldy past this many classes; larger
# label sets keep the compact heatmap instead
MAX_GRID_CLASSES = 15


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


# at most this many per-class sample pages; worst-recall classes win and the
# omission is stated in the PDF (never silent)
MAX_CLASS_PAGES = 12


def _class_sample_pages(h5_path, split, probs, labels, operating, hn_index,
                        classes, res, best, assets: Path, thumbs: int):
    """One page of thumbnails per genuine class: best predictions, worst
    predictions, and impostors (samples of OTHER labels the model routes to
    this class). Returns (pages, omitted_names, miner_grid).

    pages: [{name, stats, grids: [(png_path, note), ...]}, ...]
    """
    ds = H5SnippetDataset(h5_path, split)
    scores = genuineness_scores(probs, hn_index)
    pred = non_hn_argmax(probs, hn_index)
    genuine = labels != hn_index

    if isinstance(operating, dict):
        thr_vec = np.array([operating[int(p)] for p in pred])
    else:
        thr_vec = np.full(len(pred), float(operating))
    accepted = scores >= thr_vec
    recalled = accepted & (pred == labels) & genuine

    best_n = min(6, thumbs)          # one row: the healthy reference
    worst_n = imp_n = min(12, thumbs)  # up to two rows: the problems

    class_ids = sorted(res["per_class_recall"],
                       key=lambda c: res["per_class_recall"][c])
    omitted = [classes[c] for c in class_ids[MAX_CLASS_PAGES:]]
    class_ids = class_ids[:MAX_CLASS_PAGES]

    cs_best = best.get("controller_state") or {}
    rescued = set(cs_best.get("rescue_alphas") or {}) | \
        set(cs_best.get("rescue_repeats") or {})
    fb = set((best.get("val_metrics") or {}).get("fallback_classes") or [])

    pages = []
    with h5py.File(h5_path, "r") as f:
        def thumbs_for(positions):
            return [f["images"][int(ds.indices[p])] for p in positions]

        for c in class_ids:
            name = classes[c]
            thr_c = operating[c] if isinstance(operating, dict) else float(operating)
            mine = labels == c
            grids = []

            # best: correctly classified, highest scores
            good = np.flatnonzero(mine & (pred == c))
            good = good[np.argsort(-scores[good])][:best_n]
            if len(good):
                caps = [f"s={scores[p]:.3f}" for p in good]
                path = assets / f"class_{c}_best.png"
                plot_sample_grid(thumbs_for(good), caps,
                                 f"{name} - best predictions", path, ncols=6)
                grids.append((path, None))

            # worst: not-recalled first (lowest score first), then the
            # weakest recalled ones
            missed = np.flatnonzero(mine & ~recalled)
            missed = missed[np.argsort(scores[missed])]
            weak = np.flatnonzero(mine & recalled)
            weak = weak[np.argsort(scores[weak])]
            worst = np.concatenate([missed, weak])[:worst_n]
            if len(worst):
                caps = [f"-> {classes[pred[p]]}\ns={scores[p]:.3f}"
                        + ("" if recalled[p] else " MISSED") for p in worst]
                path = assets / f"class_{c}_worst.png"
                plot_sample_grid(thumbs_for(worst), caps,
                                 f"{name} - worst predictions (MISSED = not "
                                 "recalled at the operating point)", path, ncols=6)
                grids.append((path, "Candidates for relabeling or collecting "
                                    "more examples like them."))

            # impostors: other labels the model routes to this class, most
            # confident first; ACCEPTED = they cross this class's threshold
            imp = np.flatnonzero(~mine & (pred == c))
            imp = imp[np.argsort(-scores[imp])][:imp_n]
            if len(imp):
                caps = [f"true {classes[labels[p]]}\ns={scores[p]:.3f}"
                        + (" ACCEPTED" if accepted[p] else "") for p in imp]
                path = assets / f"class_{c}_impostors.png"
                plot_sample_grid(thumbs_for(imp), caps,
                                 f"{name} - impostors (other labels predicted "
                                 "as this class)", path, ncols=6)
                grids.append((path, "What fools this class - accepted "
                                    "impostors are its specificity leaks."))

            flags = " ".join(filter(None, ["[fallback]" if c in fb else "",
                                           "[rescued]" if c in rescued else ""]))
            routed = int((pred == c).sum())
            accepted_n = res["accepted_counts"].get(c, 0)
            stats = (f"recall {res['per_class_recall'][c]:.4f} at threshold "
                     f"{thr_c:.4f}  |  {int(mine.sum())} {SPLIT_NAMES[split]} "
                     f"samples  |  argmax routes {routed} here; {accepted_n} "
                     f"accepted as {name} (matches the confusion column), "
                     f"{routed - accepted_n} rejected to hard_negative"
                     + (f"  {flags}" if flags else ""))
            pages.append({"name": name, "stats": stats, "grids": grids})

        # training-split hard negatives the miner found hardest (EMA loss)
        miner_grid = None
        ms = best.get("miner_state")
        if ms is not None:
            train_ds = H5SnippetDataset(h5_path, SPLIT_TRAIN)
            m_scores = np.asarray(ms["scores"])
            seen = np.asarray(ms["seen"], dtype=bool)
            cand = np.flatnonzero((train_ds.labels == hn_index) & seen)
            if len(cand):
                top = cand[np.argsort(-m_scores[cand])][:thumbs]
                caps = [f"EMA loss {m_scores[p]:.2f}" for p in top]
                imgs = [f["images"][int(train_ds.indices[p])] for p in top]
                path = assets / "miner_hn.png"
                plot_sample_grid(imgs, caps,
                                 "Persistently hard training negatives "
                                 "(miner's EMA loss)", path)
                miner_grid = (path, "The training hard negatives that stayed "
                                    "difficult across epochs, per the mining "
                                    "tracker.")
    return pages, omitted, miner_grid


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
                 device: str | None = None, progress: bool = True,
                 out_dir=None, probs: np.ndarray | None = None,
                 labels: np.ndarray | None = None) -> Path:
    """Build report_<UTCstamp>.pdf. Reports accumulate (they are not pruned
    like checkpoints) — the stamp gives each generation its own file.

    out_dir defaults to run_dir; evaluate.py passes its own output directory.
    probs/labels, when supplied, skip the inference pass (the caller already
    ran the model over `split` of `h5_path` with this run's best weights).
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir) if out_dir is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    best_path = find_checkpoint(run_dir, "best")
    if best_path is None:
        raise FileNotFoundError(f"no best checkpoint in {run_dir}")
    best = torch.load(best_path, map_location=dev, weights_only=False)
    last_path = find_checkpoint(run_dir, "last")
    last = (torch.load(last_path, map_location=dev, weights_only=False)
            if last_path is not None else best)
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

    # ---- inference with the stored operating point (skipped when the
    # caller already ran it) -------------------------------------------------
    if probs is None or labels is None:
        model = build_model(best["arch"], len(classes), pretrained=False).to(dev)
        model.load_state_dict(best["model_state"])
        ds = H5SnippetDataset(str(h5_path), split,
                              imagenet_norm=best["imagenet_norm"])
        loader = DataLoader(ds, batch_size=64, pin_memory=dev.type == "cuda")
        probs, labels = collect_probs(model, loader, dev,
                                      desc="report inference",
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

    # sample-level operating-point analytics (mirrors the blind-inference
    # report): positive = genuine-class sample, score = genuineness s
    s_all = genuineness_scores(probs, hn_index)
    pos_s = np.sort(s_all[labels != hn_index])[::-1]
    neg_s = s_all[labels == hn_index]
    spec_tbl = []
    if len(pos_s) and len(neg_s):
        for tgt in SPEC_AT_RECALL_TARGETS:
            k = min(len(pos_s) - 1, max(0, math.ceil(tgt * len(pos_s)) - 1))
            t = float(pos_s[k])
            spec_tbl.append({"target": tgt, "threshold": t,
                             "recall": float((pos_s >= t).mean()),
                             "specificity": float((neg_s < t).mean())})
    rocs = {}
    for c in range(len(classes)):
        if c == hn_index:
            continue
        try:
            rocs[classes[c]] = per_class_ovr_roc(probs, labels, c)
        except ValueError:
            pass  # class absent from this split
    stored_label = ("stored thresholds (per-class)" if per_class_mode
                    else f"stored threshold {best['threshold']:.3f}")
    confusions = [(stored_label, cm)]
    for r_ in spec_tbl:
        pr = final_prediction(probs, hn_index, r_["threshold"])
        confusions.append(
            (f"{r_['target']:.0%} recall (t={r_['threshold']:.3f})",
             np.bincount(labels * len(classes) + pr,
                         minlength=len(classes) ** 2)
             .reshape(len(classes), -1)))

    # ---- assets ------------------------------------------------------------
    assets = out_dir / "report_assets"
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
    if len(classes) > MAX_GRID_CLASSES:
        charts["confusion"] = assets / "confusion.png"
        plot_confusion(cm, classes, charts["confusion"])
    else:
        charts["confusion_stored"] = assets / "confusion_stored.png"
        plot_confusion_grid(confusions[:1], classes, classes,
                            charts["confusion_stored"], ncols=1, scale=1.6)
        if len(confusions) > 1:
            charts["confusions_recall"] = assets / "confusions_recall.png"
            plot_confusion_grid(confusions[1:], classes, classes,
                                charts["confusions_recall"], ncols=2)
    if len(pos_s) and len(neg_s):
        charts["score_split"] = assets / "score_split.png"
        plot_score_split(pos_s, neg_s, charts["score_split"],
                         threshold=None if per_class_mode
                         else best["threshold"],
                         marks=[(f"{r_['target']:.0%} recall",
                                 r_["threshold"]) for r_ in spec_tbl],
                         pos_label="genuine samples",
                         neg_label="hard_negative samples", unit="samples",
                         title="Score separation: genuine vs hard_negative")
    rocs_dropped: list[str] = []
    if rocs:
        charts["roc_per_class"] = assets / "roc_per_class.png"
        rocs_dropped = plot_per_class_rocs(rocs, charts["roc_per_class"])
    class_pages, omitted_classes, miner_grid = _class_sample_pages(
        str(h5_path), split, probs, labels, operating, hn_index, classes, res,
        best, assets, thumbs)

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
        ("best checkpoint", f"{best_path.name} (epoch {best['epoch']})"
                            + (f", last epoch {last['epoch']}"
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
    s95 = next((r_ for r_ in spec_tbl
                if abs(r_["target"] - 0.95) < 1e-9), None)
    if s95:
        verdict.append(
            f"specificity at 95% acceptance recall: "
            f"{s95['specificity']:.4f} at threshold {s95['threshold']:.3f}")
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

    # ---- operating point (mirrors the blind-inference report) --------------
    if "score_split" in charts or spec_tbl:
        pdf.add_page()
        _h1(pdf, "Operating point")
        _para(pdf, "How well the genuineness score s = 1 - P(hard_negative) "
                   "separates genuine samples from hard negatives - and "
                   "where the stored threshold falls relative to the "
                   "thresholds that would achieve fixed recalls.")
        if "score_split" in charts:
            _image(pdf, charts["score_split"])
        if spec_tbl:
            _h2(pdf, "Specificity at fixed recall")
            _table(pdf, ["target recall", "achieved recall", "specificity",
                         "threshold"],
                   [[f"{r_['target']:.0%}", f"{r_['recall']:.4f}",
                     f"{r_['specificity']:.4f}", f"{r_['threshold']:.4f}"]
                    for r_ in spec_tbl],
                   widths=[pdf.epw * v for v in (0.25, 0.25, 0.25, 0.25)])
            _para(pdf, "positive = genuine-class sample, score = genuineness "
                       "s; recall here counts acceptance only (not class "
                       "correctness), so each row's threshold is directly "
                       "comparable to the stored operating threshold.",
                  size=8)
    if "roc_per_class" in charts:
        pdf.add_page()
        _h1(pdf, "Per-class ROC")
        _image(pdf, charts["roc_per_class"], w=pdf.epw * 0.85)
        _para(pdf, "One-vs-rest over the split: score = P(class)."
                   + (f" Omitted: {', '.join(rocs_dropped)}."
                      if rocs_dropped else ""), size=8.5)
    if "confusion_stored" in charts:
        pdf.add_page()
        _h1(pdf, "Confusion at the stored operating point")
        _para(pdf, "Rows are the true class, columns the final call after "
                   "the stored threshold - samples it rejects land in the "
                   "hard_negative column.")
        _image(pdf, charts["confusion_stored"])
        if "confusions_recall" in charts:
            pdf.add_page()
            _h1(pdf, "Confusion at fixed-recall thresholds")
            _para(pdf, "The same matrix re-applied at the thresholds that "
                       "achieve 85/90/95/98% acceptance recall - watch the "
                       "hard_negative column drain into the diagonal as the "
                       "threshold drops. Misses that survive even the "
                       "lowest threshold are classifier confusion, not the "
                       "operating point.")
            _image(pdf, charts["confusions_recall"])

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

    for page in class_pages:
        pdf.add_page()
        _h1(pdf, f"Class: {page['name']}")
        _para(pdf, page["stats"])
        for path, note in page["grids"]:
            _image(pdf, path)
            if note:
                _para(pdf, note, size=8.5)
    if omitted_classes:
        _para(pdf, "Per-class sample pages shown for the "
                   f"{len(class_pages)} worst-recall classes; omitted: "
                   + ", ".join(omitted_classes))
    if miner_grid:
        pdf.add_page()
        _h1(pdf, "Training hard negatives (miner)")
        _image(pdf, miner_grid[0])
        _para(pdf, miner_grid[1], size=8.5)

    pdf.add_page()
    _h1(pdf, "Configuration")
    _table(pdf, ["option", "value"],
           sorted((k, str(v)) for k, v in config.items()),
           widths=[pdf.epw * 0.4, pdf.epw * 0.6])

    out_path = out_dir / f"report_{utc_stamp()}.pdf"
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
