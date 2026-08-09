"""Operating-point metrics: threshold sweep, macro recall, HN specificity, ROC.

Score and decision rule (the operating point every metric hangs off):

    s = P(not hard_negative) = 1 - softmax(logits)[hard_negative]
    accept sample as genuine  iff  s >= threshold
    predicted class of an accepted sample = argmax over non-HN classes

A genuine sample counts as *recalled* only if it is accepted AND classified as
its true class. Macro recall averages per-class recall over the genuine
classes present in the split. Hard-negative specificity is the fraction of
hard negatives rejected (s < threshold).

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
) -> dict:
    """Choose the operating threshold on (typically validation) data.

    Returns a dict with:
      threshold        chosen operating point (accept if s >= threshold)
      target_met       whether the target macro recall is achievable at all
      macro_recall     macro recall over genuine classes at the threshold
      per_class_recall {class_index: recall} at the threshold
      specificity      fraction of hard negatives rejected (nan if none present)
      max_macro_recall macro recall with everything accepted (threshold ~ 0)
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
    macro = np.mean([cum_recall[c] for c in class_ids], axis=0)

    n_hn = int((~genuine).sum())
    cum_hn_accepted = np.cumsum(~genuine[order])

    # Valid cut points: last index of each distinct score value (accepting
    # s >= t always takes whole tie groups).
    cuts = np.append(np.flatnonzero(np.diff(s_sorted) != 0), n - 1)

    meets = macro[cuts] >= target_recall
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
        "macro_recall": float(macro[k]),
        "per_class_recall": {c: float(cum_recall[c][k]) for c in class_ids},
        "specificity": float(1.0 - cum_hn_accepted[k] / n_hn) if n_hn else float("nan"),
        "max_macro_recall": float(macro[-1]),
    }


def apply_threshold(
    probs: np.ndarray, labels: np.ndarray, hn_index: int, threshold: float
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
    return {
        "threshold": float(threshold),
        "macro_recall": float(np.mean(list(per_class.values()))) if per_class else float("nan"),
        "per_class_recall": per_class,
        "specificity": float((~accepted[~genuine]).mean()) if n_hn else float("nan"),
    }


def final_prediction(probs: np.ndarray, hn_index: int, threshold: float) -> np.ndarray:
    """Full decision rule -> class indices (hn_index for rejected samples)."""
    pred = non_hn_argmax(probs, hn_index)
    rejected = genuineness_scores(probs, hn_index) < threshold
    pred[rejected] = hn_index
    return pred


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
