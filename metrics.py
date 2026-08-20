"""Operating-point metrics: threshold sweep, macro recall, HN specificity, ROC.

Score and decision rule (the operating point every metric hangs off):

    s = P(not hard_negative) = 1 - softmax(logits)[hard_negative]
    accept sample as genuine  iff  s >= threshold
    predicted class of an accepted sample = argmax over non-HN classes

A genuine sample counts as *recalled* only if it is accepted AND classified as
its true class. Per-class recalls over the genuine classes present in the
split are combined by a configurable aggregate:

    macro     arithmetic mean — can hide one collapsed class behind several
              perfect ones ((1,1,1,1,0.1) -> 0.82)
    harmonic  harmonic mean — dominated by the worst classes ((1,1,1,1,0.1)
              -> 0.36; any class at 0 -> 0), so a recall target forces every
              class to perform
    min       worst single class — strictest, noisy when classes are small

Hard-negative specificity is the fraction of hard negatives rejected
(s < threshold).

sweep_threshold picks the LARGEST threshold whose aggregated recall still
meets the target — recall is monotone non-increasing in the threshold, so
this is the operating point with maximum specificity subject to the recall
constraint. A configurable floor (min_threshold) is never crossed: if the
target is only reachable below it, the sweep operates exactly at the floor
and reports target_met=False.

sweep_class_thresholds generalizes this to one threshold per genuine class:
samples are partitioned by predicted class, and each partition is swept
independently (class-c recall depends only on true-c samples predicted c, so
the per-partition sweeps decompose exactly). Classes with too few predicted
validation samples fall back to the global threshold.

The chosen threshold(s) are first-class outputs: stored in checkpoints and
required at inference/evaluation time.
"""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.functional.classification import binary_auroc, binary_roc
from tqdm import tqdm


@torch.no_grad()
def collect_probs(model, loader, device, desc: str | None = None,
                  progress: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; return (probs (N, K) float32, labels (N,)).

    Shows a transient batch progress bar when `progress` (labelled `desc`);
    the bar clears on completion so summary lines stay the persistent log.
    """
    model.eval()
    probs, labels = [], []
    bar = tqdm(loader, desc=desc, unit="batch", leave=False, disable=not progress)
    for imgs, labs, _ in bar:
        logits = model(imgs.to(device, non_blocking=True))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        labels.append(labs)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


RECALL_AGGREGATES = ("macro", "harmonic", "min")


def aggregate_recall(recalls: np.ndarray, agg: str, axis: int = 0) -> np.ndarray:
    """Combine per-class recalls along `axis` (values or curves).

    harmonic returns 0 wherever any class recall is 0 — the limit of the
    harmonic mean, and the behavior that makes a dead class unmissable.
    """
    recalls = np.asarray(recalls, dtype=np.float64)
    if agg == "macro":
        return recalls.mean(axis=axis)
    if agg == "harmonic":
        with np.errstate(divide="ignore"):
            inv = np.where(recalls > 0, 1.0 / np.where(recalls > 0, recalls, 1.0),
                           np.inf)
        hm = recalls.shape[axis] / inv.sum(axis=axis)
        return np.where(np.isfinite(hm), hm, 0.0)
    if agg == "min":
        return recalls.min(axis=axis)
    raise ValueError(f"agg must be one of {RECALL_AGGREGATES}, got '{agg}'")


def genuineness_scores(probs: np.ndarray, hn_index: int) -> np.ndarray:
    return 1.0 - probs[:, hn_index]


def non_hn_argmax(probs: np.ndarray, hn_index: int) -> np.ndarray:
    """Predicted genuine class (ignoring the hard_negative column)."""
    non_hn = np.delete(np.arange(probs.shape[1]), hn_index)
    return non_hn[np.argmax(probs[:, non_hn], axis=1)]


def _per_sample_thresholds(threshold, pred: np.ndarray, num_classes: int):
    """Resolve a scalar or {class: t} threshold to one value per sample
    (per-class thresholds apply the PREDICTED class's threshold)."""
    if isinstance(threshold, dict):
        vec = np.zeros(num_classes)
        for c, t in threshold.items():
            vec[c] = t
        return vec[pred]
    return float(threshold)


def _choose_cut(s_sorted: np.ndarray, curve: np.ndarray, target: float,
                floor: float) -> tuple[int, bool]:
    """Pick an operating cut on descending-sorted scores.

    Valid cuts are the last index of each distinct score value (accepting
    s >= t always takes whole tie groups), restricted to thresholds >= floor.
    Returns (k, True) for the highest-threshold cut whose curve value meets
    the target, else (k_floor, False) where k_floor accepts exactly the
    samples with s >= floor (k_floor == -1 means accept nothing).
    """
    n = len(s_sorted)
    cuts = np.append(np.flatnonzero(np.diff(s_sorted) != 0), n - 1)
    eligible = cuts[s_sorted[cuts] >= floor]
    if eligible.size:
        meets = curve[eligible] >= target
        if meets.any():
            return int(eligible[int(np.argmax(meets))]), True
    return int((s_sorted >= floor).sum()) - 1, False


def sweep_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    hn_index: int,
    target_recall: float,
    agg: str = "macro",
    min_threshold: float = 0.0,
) -> dict:
    """Choose the global operating threshold on (typically validation) data.

    Every per-class recall is monotone non-decreasing as the threshold falls,
    so each aggregate (macro/harmonic/min) is too — the first cut meeting the
    target is still the maximum-specificity operating point. The threshold
    never goes below min_threshold: if the target is only reachable beneath
    the floor, the operating point IS the floor and target_met is False.

    Returns a dict with:
      threshold        chosen operating point (accept if s >= threshold)
      target_met       whether the target was met at a threshold >= the floor
      recall           aggregated recall over genuine classes at the threshold
      recall_agg       the aggregate used ('macro' | 'harmonic' | 'min')
      per_class_recall {class_index: recall} at the threshold
      specificity      fraction of hard negatives rejected (nan if none present)
      max_recall       aggregated recall with everything accepted (threshold ~ 0),
                       the model's ceiling ignoring the floor
      predicted_counts {class_index: samples argmax-assigned to the class}
      min_threshold    the floor in effect
    """
    scores = genuineness_scores(probs, hn_index)
    pred = non_hn_argmax(probs, hn_index)
    genuine = labels != hn_index
    if not genuine.any():
        raise ValueError("no genuine samples to sweep a threshold on")

    order = np.argsort(-scores, kind="stable")
    s_sorted = scores[order]
    labels_s = labels[order]
    correct_s = (pred[order] == labels_s) & genuine[order]

    class_ids = [int(c) for c in np.unique(labels[genuine])]
    class_totals = {c: int((labels == c).sum()) for c in class_ids}

    # Cumulative per-class recall as the accepted set grows (threshold falls).
    cum_recall = {
        c: np.cumsum(correct_s & (labels_s == c)) / class_totals[c] for c in class_ids
    }
    agg_curve = aggregate_recall(np.stack([cum_recall[c] for c in class_ids]), agg)

    n_hn = int((~genuine).sum())
    cum_hn_accepted = np.cumsum(~genuine[order])

    k, target_met = _choose_cut(s_sorted, agg_curve, target_recall, min_threshold)
    threshold = float(s_sorted[k]) if target_met else float(min_threshold)

    if k >= 0:
        recall = float(agg_curve[k])
        per_class = {c: float(cum_recall[c][k]) for c in class_ids}
        spec = float(1.0 - cum_hn_accepted[k] / n_hn) if n_hn else float("nan")
    else:  # floor above every score: nothing accepted
        recall = 0.0
        per_class = {c: 0.0 for c in class_ids}
        spec = 1.0 if n_hn else float("nan")

    accepted_mask = scores >= threshold
    return {
        "threshold": threshold,
        "target_met": target_met,
        "recall": recall,
        "recall_agg": agg,
        "per_class_recall": per_class,
        "specificity": spec,
        "max_recall": float(agg_curve[-1]),
        # argmax routing (pre-threshold) vs final predictions (post-threshold,
        # matches the confusion-matrix column)
        "predicted_counts": {int(c): int((pred == c).sum()) for c in
                             range(probs.shape[1]) if c != hn_index},
        "accepted_counts": {int(c): int((accepted_mask & (pred == c)).sum())
                            for c in range(probs.shape[1]) if c != hn_index},
        "min_threshold": float(min_threshold),
    }


def sweep_class_thresholds(
    probs: np.ndarray,
    labels: np.ndarray,
    hn_index: int,
    target_recall: float,
    agg: str = "macro",
    min_threshold: float = 0.0,
    min_count: int = 20,
) -> dict:
    """Choose one operating threshold per genuine class.

    A sample's threshold is that of its PREDICTED class, so the sweep
    decomposes: class-c recall depends only on true-c samples predicted c,
    and each predicted-class partition is swept independently for the
    highest threshold >= min_threshold meeting the recall target. Classes
    with fewer than min_count predicted validation samples — or with no
    genuine validation samples at all — fall back to the global threshold
    (which is always computed, and also serves as the scalar 'threshold'
    field for logging).

    target_met requires EVERY class with genuine validation samples to meet
    the target (per-class mode implicitly gates like --recall-agg min; the
    agg still shapes the single reported recall number).
    """
    global_op = sweep_threshold(probs, labels, hn_index, target_recall, agg,
                                min_threshold)
    scores = genuineness_scores(probs, hn_index)
    pred = non_hn_argmax(probs, hn_index)

    thresholds: dict[int, float] = {}
    met: dict[int, bool] = {}
    fallback: dict[int, bool] = {}
    for c in range(probs.shape[1]):
        if c == hn_index:
            continue
        part = np.flatnonzero(pred == c)
        n_true = int((labels == c).sum())
        if n_true == 0 or len(part) < min_count:
            thresholds[c] = global_op["threshold"]
            fallback[c] = True
            continue
        fallback[c] = False
        order = np.argsort(-scores[part], kind="stable")
        s_sorted = scores[part][order]
        curve = np.cumsum(labels[part][order] == c) / n_true
        k, ok = _choose_cut(s_sorted, curve, target_recall, min_threshold)
        thresholds[c] = float(s_sorted[k]) if ok else float(min_threshold)
        met[c] = ok

    res = apply_threshold(probs, labels, hn_index, thresholds, agg=agg)
    for c, r in res["per_class_recall"].items():
        if fallback.get(c):  # judged at the fallback threshold it actually uses
            met[c] = r >= target_recall

    return {
        "threshold": global_op["threshold"],  # global fallback, for logging
        "class_thresholds": thresholds,
        "target_met": all(met.get(c, False) for c in res["per_class_recall"]),
        "recall": res["recall"],
        "recall_agg": agg,
        "per_class_recall": res["per_class_recall"],
        "specificity": res["specificity"],
        "tpr": res["tpr"],
        "max_recall": global_op["max_recall"],
        "predicted_counts": global_op["predicted_counts"],
        "accepted_counts": res["accepted_counts"],
        "fallback_classes": sorted(c for c, f in fallback.items() if f),
        "min_threshold": float(min_threshold),
    }


def apply_threshold(
    probs: np.ndarray, labels: np.ndarray, hn_index: int,
    threshold: "float | dict[int, float]", agg: str = "macro",
) -> dict:
    """Evaluate a FIXED operating point (e.g. the stored one, on the test
    split). `threshold` is either the global scalar or a {class: t} dict of
    per-class thresholds, applied by predicted class."""
    scores = genuineness_scores(probs, hn_index)
    pred = non_hn_argmax(probs, hn_index)
    genuine = labels != hn_index
    accepted = scores >= _per_sample_thresholds(threshold, pred, probs.shape[1])
    correct = accepted & (pred == labels) & genuine

    class_ids = [int(c) for c in np.unique(labels[genuine])]
    per_class = {c: float(correct[labels == c].mean()) for c in class_ids}

    n_hn = int((~genuine).sum())
    recall = (float(aggregate_recall(np.array(list(per_class.values())), agg))
              if per_class else float("nan"))
    per_class_mode = isinstance(threshold, dict)
    return {
        "threshold": None if per_class_mode else float(threshold),
        "class_thresholds": dict(threshold) if per_class_mode else None,
        "recall": recall,
        "recall_agg": agg,
        "per_class_recall": per_class,
        "specificity": float((~accepted[~genuine]).mean()) if n_hn else float("nan"),
        "tpr": float(accepted[genuine].mean()),  # genuine acceptance rate
        # final predictions per class (accepted & argmax == c) — matches the
        # confusion-matrix column; rejected samples land on hard_negative
        "accepted_counts": {int(c): int((accepted & (pred == c)).sum())
                            for c in range(probs.shape[1]) if c != hn_index},
    }


def final_prediction(probs: np.ndarray, hn_index: int,
                     threshold: "float | dict[int, float]") -> np.ndarray:
    """Full decision rule -> class indices (hn_index for rejected samples)."""
    pred = non_hn_argmax(probs, hn_index)
    scores = genuineness_scores(probs, hn_index)
    rejected = scores < _per_sample_thresholds(threshold, pred, probs.shape[1])
    out = pred.copy()
    out[rejected] = hn_index
    return out


def calibration_bins(
    probs: np.ndarray, labels: np.ndarray, hn_index: int, n_bins: int = 10
) -> dict:
    """Reliability data for the genuineness score s = P(not hard_negative).

    Equal-width bins over [0, 1]; empty bins are dropped. Returns per-bin
    mean predicted score, observed genuine fraction, and counts, plus the
    expected calibration error (ECE): the count-weighted mean |observed -
    predicted|. A perfectly calibrated score has observed == predicted in
    every bin and ECE 0.
    """
    scores = genuineness_scores(probs, hn_index)
    genuine = labels != hn_index

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, edges[1:-1]), 0, n_bins - 1)

    mean_pred, frac_genuine, counts, bins = [], [], [], []
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n == 0:
            continue
        mean_pred.append(float(scores[mask].mean()))
        frac_genuine.append(float(genuine[mask].mean()))
        counts.append(n)
        bins.append(b)

    mean_pred = np.array(mean_pred)
    frac_genuine = np.array(frac_genuine)
    counts = np.array(counts)
    ece = float(np.sum(counts / counts.sum() * np.abs(frac_genuine - mean_pred)))

    return {"mean_pred": mean_pred, "frac_genuine": frac_genuine,
            "counts": counts, "bins": np.array(bins), "ece": ece,
            "n_bins": n_bins}


def genuine_vs_hn_roc(
    probs: np.ndarray, labels: np.ndarray, hn_index: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Binary genuine-vs-hard-negative ROC on the genuineness score.

    Returns (fpr, tpr, auc). fpr is the fraction of hard negatives accepted,
    so HN specificity = 1 - fpr. Requires both populations present.
    """
    scores = torch.from_numpy(genuineness_scores(probs, hn_index))
    genuine = torch.from_numpy((labels != hn_index).astype(np.int64))
    if genuine.min() == genuine.max():
        raise ValueError("ROC needs both genuine and hard-negative samples")
    fpr, tpr, _ = binary_roc(scores, genuine)
    auc = binary_auroc(scores, genuine)
    return fpr.numpy(), tpr.numpy(), float(auc)


def per_class_ovr_roc(
    probs: np.ndarray, labels: np.ndarray, class_index: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """One-vs-rest ROC for a single class using its softmax probability."""
    scores = torch.from_numpy(probs[:, class_index].copy())
    positive = torch.from_numpy((labels == class_index).astype(np.int64))
    if positive.min() == positive.max():
        raise ValueError(f"class {class_index}: need positives and negatives for ROC")
    fpr, tpr, _ = binary_roc(scores, positive)
    auc = binary_auroc(scores, positive)
    return fpr.numpy(), tpr.numpy(), float(auc)
