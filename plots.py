"""Matplotlib PNG plots for training and evaluation (offline-friendly).

Styling follows a small fixed system: a light chart surface, recessive
hairline grid, ink-colored text, and a fixed-order categorical palette
(colors follow the entity — class i always gets slot i mod 8 — never
re-assigned when series are dropped). One value axis per chart; loss lives
in its own panel, never on a twin axis.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Fixed categorical order (validated reference palette, light mode).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

MAX_ROC_SERIES = 8  # past this, keep the worst-AUC classes and say what was dropped


def _new_axes(figsize=(6.4, 4.8)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    _style_axes(ax)
    return fig, ax


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def _legend(ax) -> None:
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
                    facecolor=SURFACE, edgecolor=GRID)
    for text in leg.get_texts():
        text.set_color(INK_2)


def _save(fig, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_genuine_vs_hn_roc(fpr, tpr, auc: float, path,
                           operating_point: dict | None = None) -> None:
    """Binary genuine-vs-hard-negative ROC; x is the fraction of hard
    negatives accepted (1 - specificity). Marks the stored operating point."""
    fig, ax = _new_axes((5.2, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    ax.plot(fpr, tpr, color=SERIES[0], linewidth=2)
    ax.text(0.98, 0.06, f"AUROC {auc:.4f}", transform=ax.transAxes,
            ha="right", fontsize=9, color=INK_2)

    if operating_point is not None:
        # The point's y is the genuine acceptance rate (TPR), not recall
        # (which also demands correct class). With one global threshold it
        # lies exactly on the binary curve; per-class thresholds are not a
        # single cut of s, so their point generally sits off the curve.
        x = 1.0 - operating_point["specificity"]
        y = operating_point["tpr"]
        ax.plot([x], [y], marker="o", markersize=9, color=SERIES[1],
                markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
        thr = operating_point.get("threshold")
        thr_txt = (f"thr {thr:.4f}" if thr is not None
                   else "per-class thresholds (off-curve)")
        # Keep the label inside the axes wherever the point lands.
        va = "top" if y > 0.75 else "bottom"
        dy = -10 if va == "top" else 10
        ax.annotate(
            f"operating point\n{thr_txt}\n"
            f"{operating_point.get('recall_agg', 'macro')} recall "
            f"{operating_point['recall']:.3f}, "
            f"spec {operating_point['specificity']:.3f}",
            (x, y), xytext=(14, dy), textcoords="offset points",
            fontsize=8, color=INK_2, va=va,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("hard negatives accepted (1 - specificity)")
    ax.set_ylabel("genuine samples accepted (TPR)")
    ax.set_title("Genuine vs hard-negative ROC")
    _save(fig, path)


def plot_per_class_rocs(rocs: dict[str, tuple], path) -> list[str]:
    """One-vs-rest ROCs, one line per class in fixed palette order.

    rocs: {class_name: (fpr, tpr, auc)} in class-index order. If there are
    more than MAX_ROC_SERIES classes, the plot keeps the worst-AUC classes
    (the interesting ones) and returns the names of those dropped — the
    caller should report them, never drop silently.
    """
    names = list(rocs)
    dropped: list[str] = []
    if len(names) > MAX_ROC_SERIES:
        by_auc = sorted(names, key=lambda n: rocs[n][2])
        keep = set(by_auc[:MAX_ROC_SERIES])
        dropped = [n for n in names if n not in keep]
        names = [n for n in names if n in keep]

    fig, ax = _new_axes((5.6, 5.2))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    for i, name in enumerate(names):
        fpr, tpr, auc = rocs[name]
        ax.plot(fpr, tpr, color=SERIES[i % len(SERIES)], linewidth=2,
                label=f"{name} (AUC {auc:.3f})")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    title = "Per-class one-vs-rest ROC"
    if dropped:
        title += f" (worst {len(names)} of {len(rocs)} classes)"
    ax.set_title(title)
    _legend(ax)
    _save(fig, path)
    return dropped


def plot_history(csv_path, path) -> None:
    """Metric-vs-epoch curves from train.py's metrics.csv.

    Top panel: the [0, 1] metrics (macro recall, specificity, AUROC,
    threshold) on one shared axis. Bottom panel: training loss on its own
    axis — separate panel, not a twin axis.
    """
    rows = list(csv.DictReader(open(csv_path, newline="")))
    if not rows:
        raise ValueError(f"{csv_path}: no rows to plot")
    epochs = [int(r["epoch"]) for r in rows]

    def col(name):
        return [float(r[name]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), facecolor=SURFACE, sharex=True,
        height_ratios=[2.2, 1], constrained_layout=True,
    )
    _style_axes(ax1)
    _style_axes(ax2)

    # Old CSVs (pre recall-agg) used macro_recall; keep them plottable.
    if "recall" in rows[0]:
        recall_col = "recall"
        recall_label = f"{rows[0].get('recall_agg', 'macro')} recall (genuine)"
    else:
        recall_col, recall_label = "macro_recall", "macro recall (genuine)"
    unit_series = [  # (csv column, label, palette slot)
        (recall_col, recall_label, 0),
        ("specificity", "HN specificity", 1),
        ("auroc", "genuine-vs-HN AUROC", 2),
        ("threshold", "operating threshold", 3),
    ]
    for name, label, slot in unit_series:
        ax1.plot(epochs, col(name), color=SERIES[slot], linewidth=2, label=label)
    # Per-class threshold spread as a band around the global threshold line
    # (collapses to nothing in global mode, where thr_min == thr_max).
    if "thr_min" in rows[0] and "thr_max" in rows[0]:
        ax1.fill_between(epochs, col("thr_min"), col("thr_max"),
                         color=SERIES[3], alpha=0.18, linewidth=0)
    ax1.set_ylim(-0.04, 1.04)
    ax1.set_ylabel("value")
    ax1.set_title("Validation metrics per epoch")
    _legend(ax1)

    ax2.plot(epochs, col("train_loss"), color=SERIES[6], linewidth=2)
    ax2.set_ylabel("train loss")
    ax2.set_xlabel("epoch")
    ax2.set_title("Training loss", fontsize=10)

    _save(fig, path)


def plot_per_class_recall_history(class_csv_path, path) -> list[str]:
    """Per-class validation recall over epochs, from class_thresholds.csv.

    One line per class in fixed palette order (by first appearance). Past
    MAX_ROC_SERIES classes, keeps the worst-final-recall ones and returns the
    dropped names — callers should report them, never drop silently.
    """
    rows = list(csv.DictReader(open(class_csv_path, newline="")))
    if not rows:
        raise ValueError(f"{class_csv_path}: no rows to plot")

    series: dict[str, tuple[list, list]] = {}
    for r in rows:
        xs, ys = series.setdefault(r["class"], ([], []))
        xs.append(int(r["epoch"]))
        ys.append(float(r["recall"]))

    order = list(series)  # palette follows the entity: first-appearance order
    dropped: list[str] = []
    if len(order) > MAX_ROC_SERIES:
        by_final = sorted(order, key=lambda n: series[n][1][-1])
        keep = set(by_final[:MAX_ROC_SERIES])
        dropped = [n for n in order if n not in keep]
        order = [n for n in order if n in keep]

    fig, ax = _new_axes((7.2, 4.4))
    for i, name in enumerate(order):
        xs, ys = series[name]
        ax.plot(xs, ys, color=SERIES[i % len(SERIES)], linewidth=2, label=name)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation recall")
    title = "Per-class recall per epoch"
    if dropped:
        title += f" (worst {len(order)} of {len(series)} classes)"
    ax.set_title(title)
    _legend(ax)
    _save(fig, path)
    return dropped


EVENT_MARKS = {  # base event -> (marker, palette slot)
    "raise": ("^", 2), "hold": ("o", 0), "rewind": ("v", 7), "ceiling": ("s", 3),
}


def plot_controller_timeline(csv_path, path) -> None:
    """Smart-mode controller history: pressure with boundary-event markers on
    top, cycled learning rate (log scale) below. Skips rows from non-smart
    epochs (empty pressure column)."""
    rows = [r for r in csv.DictReader(open(csv_path, newline=""))
            if r.get("pressure")]
    if not rows:
        raise ValueError(f"{csv_path}: no smart-mode rows to plot")
    epochs = [int(r["epoch"]) for r in rows]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.2), facecolor=SURFACE, sharex=True,
        height_ratios=[1.6, 1], constrained_layout=True,
    )
    _style_axes(ax1)
    _style_axes(ax2)

    ax1.step(epochs, [float(r["pressure"]) for r in rows], where="post",
             color=SERIES[0], linewidth=2)
    seen_events = set()
    for r in rows:
        if not r["event"]:
            continue
        base = r["event"].split()[0]
        mark, slot = EVENT_MARKS.get(base, ("o", 0))
        label = base if base not in seen_events else None
        seen_events.add(base)
        ax1.plot([int(r["epoch"])], [float(r["pressure"])], marker=mark,
                 markersize=8, color=SERIES[slot], markeredgecolor=SURFACE,
                 markeredgewidth=1.2, linestyle="none", label=label, zorder=5)
    ax1.set_ylim(-0.05, 1.1)
    ax1.set_ylabel("hard-negative pressure")
    ax1.set_title("Smart controller timeline")
    if seen_events:
        _legend(ax1)

    ax2.plot(epochs, [float(r["lr"]) for r in rows], color=SERIES[6], linewidth=2)
    ax2.set_yscale("log")
    ax2.set_ylabel("learning rate")
    ax2.set_xlabel("epoch")
    _save(fig, path)


GRID_WIDTH = 9.5  # inches: the widest a sample grid renders in the reports


def _caption_cell_width(captions) -> float:
    """Cell width (inches) needed by the longest caption line at the grid's
    fontsize-7 caption font (DejaVu Sans averages ~0.058 in per char)."""
    chars = max((len(ln) for c in captions for ln in str(c).split("\n")),
                default=0)
    return max(1.9, chars * 0.058 + 0.3)


def grid_ncols(captions, ncols: int = 5) -> int:
    """Columns (at most ncols) that fit GRID_WIDTH with cells wide enough
    for the longest caption line — long filenames get fewer, wider columns
    instead of overlapping their neighbours. Callers that paginate a pool
    use this to size their per-page chunks to whole rows."""
    return max(1, min(ncols, int(GRID_WIDTH // _caption_cell_width(captions))))


def plot_sample_grid(images: list[np.ndarray], captions: list[str], title: str,
                     path, ncols: int = 4) -> None:
    """Grid of raw dataset thumbnails (uint8 HWC) with per-sample captions —
    used by the report to show the actual problem samples. Columns drop (and
    cells widen) when the captions need the room, so entire filenames stay
    readable."""
    n = len(images)
    ncols = min(grid_ncols(captions, ncols), max(n, 1))
    nrows = (n + ncols - 1) // ncols
    # row height grows with the deepest caption so multi-line labels never
    # collide with the row above
    cap_lines = max((c.count(chr(10)) + 1 for c in captions), default=1)
    row_h = 2.3 + 0.28 * max(0, cap_lines - 2)
    cell_w = min(_caption_cell_width(captions), GRID_WIDTH)
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell_w * ncols, row_h * nrows),
                             facecolor=SURFACE, squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, img, cap in zip(axes.flat, images, captions):
        ax.imshow(img.squeeze() if img.shape[-1] == 1 else img,
                  cmap="gray" if img.shape[-1] == 1 else None,
                  vmin=0, vmax=255)
        ax.set_title(cap, fontsize=7, color=INK_2)
    fig.suptitle(title, fontsize=11, color=INK)
    _save(fig, path)


def plot_score_split(pos_scores, neg_scores, path, threshold=None,
                     marks=None, pos_label="windows on GT points",
                     neg_label="background windows", unit="windows",
                     title="Score separation: GT windows vs background") -> None:
    """Genuineness-score distributions of the positive population vs the
    negative one, with the stored operating threshold and optional labelled
    marks (e.g. the thresholds that achieve fixed recalls). Log-scaled
    counts: negatives typically outnumber positives by orders of magnitude.
    Default labels fit the inference report (windows on GT points vs
    background); the training/evaluate report relabels for samples."""
    pos = np.asarray(pos_scores)
    neg = np.asarray(neg_scores)
    fig, ax = _new_axes((7.2, 4.0))
    bins = np.linspace(0.0, 1.0, 41)
    ax.hist(neg, bins, color=SERIES[1], alpha=0.55,
            label=f"{neg_label} (n={len(neg)})")
    ax.hist(pos, bins, color=SERIES[2], alpha=0.8,
            label=f"{pos_label} (n={len(pos)})")
    ax.set_yscale("log")
    if threshold is not None:
        ax.axvline(threshold, color=INK, linewidth=1.4, linestyle=(0, (4, 3)))
        ax.text(threshold, ax.get_ylim()[1], f" stored thr {threshold:.3f}",
                fontsize=8, color=INK, va="top", ha="left")
    y_lo, y_hi = ax.get_ylim()
    for i, (label, x) in enumerate(marks or []):
        ax.axvline(x, color=MUTED, linewidth=1, linestyle=(0, (1, 3)))
        # stagger alternate labels so neighbouring thresholds stay legible
        y_text = y_lo * (1.6 if i % 2 == 0 else 1.6 * (y_hi / y_lo) ** 0.35)
        ax.text(x, y_text, f" {label}", fontsize=7, color=INK_2,
                rotation=90, va="bottom", ha="left")
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("genuineness score s")
    ax.set_ylabel(f"{unit} (log)")
    ax.set_title(title)
    leg = ax.legend(loc="upper center", fontsize=8, framealpha=0.9,
                    facecolor=SURFACE, edgecolor=GRID)
    for text in leg.get_texts():
        text.set_color(INK_2)
    _save(fig, path)


def plot_confusion_grid(entries, row_names, col_names, path,
                        ncols: int = 2, title: str | None = None,
                        scale: float = 1.0) -> None:
    """Confusion matrices (one per candidate operating point):
    row-normalized shading, raw counts annotated. entries =
    [(title, matrix), ...] sharing the same row/col labels; scale > 1
    renders larger cells (e.g. a single full-width matrix)."""
    n = len(entries)
    ncols = min(ncols, max(n, 1))
    nrows = (n + ncols - 1) // ncols
    k = max(len(row_names), len(col_names))
    cell = (0.52 * k + 1.7) * scale
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell, nrows * (cell + 0.4)),
                             facecolor=SURFACE, squeeze=False)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", [SURFACE, "#cde2fb", "#3987e5", "#0d366b"])
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, (sub_title, cm) in zip(axes.flat, entries):
        ax.set_axis_on()
        frac = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)
        fs = 7 * (scale ** 0.5)
        ax.set_xticks(range(len(col_names)), col_names, rotation=45,
                      ha="right", fontsize=fs)
        ax.set_yticks(range(len(row_names)), row_names, fontsize=fs)
        ax.tick_params(colors=MUTED, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for i in range(len(row_names)):
            for j in range(len(col_names)):
                if cm[i, j]:
                    color = SURFACE if frac[i, j] > 0.55 else INK
                    ax.text(j, i, str(int(cm[i, j])), ha="center",
                            va="center", fontsize=6.5 * (scale ** 0.5),
                            color=color)
        ax.set_title(sub_title, fontsize=9 * (scale ** 0.5), color=INK)
        ax.set_xlabel("final call", color=INK_2, fontsize=7.5 * (scale ** 0.5))
        ax.set_ylabel("true (GT)", color=INK_2, fontsize=7.5 * (scale ** 0.5))
    if title:
        fig.suptitle(title, fontsize=11, color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        fig.tight_layout()
    _save(fig, path)


def plot_calibration(cal: dict, path, threshold: float | None = None) -> None:
    """Reliability diagram for the genuineness score.

    Top panel: observed genuine fraction vs mean predicted score per bin,
    against the perfect-calibration diagonal. Bottom panel: per-bin sample
    counts on their own axis (never a twin axis) — with heavy imbalance the
    tail bins dominate, so counts use a log scale when they span decades.
    """
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.6, 6.4), facecolor=SURFACE, sharex=True,
        height_ratios=[2.6, 1], constrained_layout=True,
    )
    _style_axes(ax1)
    _style_axes(ax2)

    ax1.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=(0, (4, 4)))
    # Connect only adjacent occupied bins — a line across empty bins would
    # imply observations that don't exist. Isolated bins render as markers.
    breaks = np.flatnonzero(np.diff(cal["bins"]) > 1) + 1
    for seg in np.split(np.arange(len(cal["bins"])), breaks):
        ax1.plot(cal["mean_pred"][seg], cal["frac_genuine"][seg],
                 color=SERIES[0], linewidth=2, marker="o", markersize=6,
                 markeredgecolor=SURFACE, markeredgewidth=1.5)
    if threshold is not None:
        ax1.axvline(threshold, color=SERIES[1], linewidth=1.2,
                    linestyle=(0, (2, 3)))
        ha = "right" if threshold > 0.5 else "left"
        pad = -0.01 if ha == "right" else 0.01
        ax1.text(threshold + pad, 0.97, f"thr {threshold:.3f}", fontsize=8,
                 color=SERIES[1], ha=ha, va="top")
    ax1.text(0.98, 0.04, f"ECE {cal['ece']:.4f}", transform=ax1.transAxes,
             ha="right", fontsize=9, color=INK_2)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)
    ax1.set_ylabel("observed genuine fraction")
    ax1.set_title("Calibration of the genuineness score")

    width = 0.9 / cal["n_bins"]
    ax2.bar(cal["mean_pred"], cal["counts"], width=width, color=SERIES[0],
            edgecolor=SURFACE, linewidth=0.5)
    counts = cal["counts"]
    if counts.max() > 50 * max(counts.min(), 1):
        ax2.set_yscale("log")
    ax2.set_ylabel("samples")
    ax2.set_xlabel("predicted genuineness score s (bin mean)")

    _save(fig, path)


def plot_confusion(cm: np.ndarray, class_names: list[str], path) -> None:
    """Row-normalized confusion heatmap (rows = true class), single-hue
    sequential ramp, annotated with counts when the grid is small enough."""
    k = len(class_names)
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    frac = cm / row_sums

    size = max(4.6, 0.55 * k + 2.0)
    fig, ax = plt.subplots(figsize=(size + 1.2, size), facecolor=SURFACE)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "seq_blue", [SURFACE, "#cde2fb", "#3987e5", "#0d366b"])
    im = ax.imshow(frac, cmap=cmap, vmin=0, vmax=1)

    ax.set_xticks(range(k), class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(k), class_names, fontsize=8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("predicted", color=INK_2)
    ax.set_ylabel("true", color=INK_2)
    ax.set_title("Confusion matrix (row-normalized shading, raw counts)",
                 color=INK, fontsize=10)

    if k <= 20:
        for i in range(k):
            for j in range(k):
                if cm[i, j]:
                    color = SURFACE if frac[i, j] > 0.55 else INK
                    ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                            fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045)
    cbar.ax.tick_params(colors=MUTED, labelsize=7)
    cbar.outline.set_edgecolor(GRID)
    _save(fig, path)
