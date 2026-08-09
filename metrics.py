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

sweep_threshold picks the LARGEST threshold whose macro recall still meets the
target — recall is monotone non-increasing in the threshold, so this is the
operating point with maximum specificity subject to the recall constraint.
The chosen threshold is a first-class output: it is stored in checkpoints and
required at inference/evaluation time.
"""

from __future__ import annotations

import numpy as np
import torch
from torchmetrics.functional.classification import binary_auroc, binary_roc


@torch.no_grad()
def collect_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a loader; return (probs (N, K) float32, labels (N,))."""
    model.eval()
    probs, labels = [], []
    for imgs, labs, _ in loader:
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


def sweep_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    hn_index: int,
    target_recall: float,
    agg: str = "macro",
) -> dict:
    """Choose the operating threshold on (typically validation) data.

    Every per-class recall is monotone non-decreasing as the threshold falls,
    so each aggregate (macro/harmonic/min) is too — the first cut meeting the
    target is still the maximum-specificity operating point.

    Returns a dict with:
      threshold        chosen operating point (accept if s >= threshold)
      target_met       whether the target aggregated recall is achievable
      recall           aggregated recall over genuine classes at the threshold
      recall_agg       the aggregate used ('macro' | 'harmonic' | 'min')
      per_class_recall {class_index: recall} at the threshold
      specificity      fraction of hard negatives rejected (nan if none present)
      max_recall       aggregated recall with everything accepted (threshold ~ 0)
    """
    n = len(labels)
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

    # Valid cut points: last index of each distinct score value (accepting
    # s >= t always takes whole tie groups).
    cuts = np.append(np.flatnonzero(np.diff(s_sorted) != 0), n - 1)

    meets = agg_curve[cuts] >= target_recall
    if meets.any():
        k = cuts[int(np.argmax(meets))]  # first (highest-threshold) cut meeting target
        threshold = float(s_sorted[k])
        target_met = True
    else:
        k = cuts[-1]  # accept everything; best we can do
        threshold = 0.0
        target_met = False

    return {
        "threshold": threshold,
        "target_met": target_met,
        "recall": float(agg_curve[k]),
        "recall_agg": agg,
        "per_class_recall": {c: float(cum_recall[c][k]) for c in class_ids},
        "specificity": float(1.0 - cum_hn_accepted[k] / n_hn) if n_hn else float("nan"),
        "max_recall": float(agg_curve[-1]),
    }


def apply_threshold(
    probs: np.ndarray, labels: np.ndarray, hn_index: int, threshold: float,
    agg: str = "macro",
) -> dict:
    """Evaluate a FIXED threshold (e.g. the stored one, on the test split)."""
    scores = genuineness_scores(probs, hn_index)
    pred = non_hn_argmax(probs, hn_index)
    genuine = labels != hn_index
    accepted = scores >= threshold
    correct = accepted & (pred == labels) & genuine

    class_ids = [int(c) for c in np.unique(labels[genuine])]
    per_class = {c: float(correct[labels == c].mean()) for c in class_ids}

    n_hn = int((~genuine).sum())
    recall = (float(aggregate_recall(np.array(list(per_class.values())), agg))
              if per_class else float("nan"))
    return {
        "threshold": float(threshold),
        "recall": recall,
        "recall_agg": agg,
        "per_class_recall": per_class,
        "specificity": float((~accepted[~genuine]).mean()) if n_hn else float("nan"),
        "tpr": float(accepted[genuine].mean()),  # genuine acceptance rate
    }


def final_prediction(probs: np.ndarray, hn_index: int, threshold: float) -> np.ndarray:
    """Full decision rule -> class indices (hn_index for rejected samples)."""
    pred = non_hn_argmax(probs, hn_index)
    rejected = genuineness_scores(probs, hn_index) < threshold
    pred[rejected] = hn_index
    return pred


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
