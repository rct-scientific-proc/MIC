"""Multiclass focal loss with per-class alpha weighting, optionally
asymmetric for the hard-negative class.

FL(p_y) = -alpha[y] * (1 - p_y)^gamma[y] * log(p_y)

The hard_negative class gets its own alpha (typically < 1 early in training,
optionally ramped upward) so the loss stays recall-focused on genuine classes
while hard-negative pressure is applied gradually.

Asymmetric focusing (Ridnik et al., "Asymmetric Loss for Multi-Label
Classification"): when negatives outnumber positives by orders of
magnitude, the sum of many small easy-negative losses still dominates the
gradient at the symmetric gamma. hn_gamma gives the hard-negative class its
own, larger exponent (4 is the paper's choice for negatives, 0-1 for
positives), and hn_margin m shifts a hard negative's probability
p_y -> min(p_y + m, 1) before the loss: a negative the model already
rejects with genuineness s = 1 - p_y <= m contributes exactly zero loss and
zero gradient, so training capacity goes to the negatives that still look
genuine - the ones that matter at the operating threshold.

forward() returns per-sample losses (reduction='none') because the training
loop feeds them to the hard-negative miner; call .mean() for the batch loss.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from dataset import CONFIDENCE_MAX, CONFIDENCE_UNRATED


def parse_confidence_curve(spec: str):
    """--confidence-curve -> f mapping t in [0, 1] (the rating rescaled so 1
    -> 0 and CONFIDENCE_MAX -> 1) to [0, 1], vectorised over numpy arrays.

    linear      t
    power:P     t ** P. P > 1 is convex: mid ratings stay near the minimum
                weight and only the surest labels count fully; P < 1 is
                concave: mid ratings sit near full weight and only the
                shakiest labels lose much.
    step:K      0 below rating K, 1 from K up - a hard cut.
    """
    name, _, arg = str(spec).strip().lower().partition(":")
    if name == "linear":
        if arg:
            raise ValueError("linear takes no parameter")
        return lambda t: np.asarray(t, dtype=np.float64)
    if name == "power":
        try:
            p = float(arg)
        except ValueError:
            raise ValueError("power needs an exponent, e.g. power:2") from None
        if not p > 0:
            raise ValueError(f"power exponent must be > 0, got {arg}")
        return lambda t: np.asarray(t, dtype=np.float64) ** p
    if name == "step":
        try:
            k = int(arg)
        except ValueError:
            raise ValueError("step needs a rating, e.g. step:3") from None
        if not 2 <= k <= CONFIDENCE_MAX:
            raise ValueError(f"step rating must be 2..{CONFIDENCE_MAX}, got {arg}")
        t_k = (k - 1) / (CONFIDENCE_MAX - 1)
        return lambda t: (np.asarray(t, dtype=np.float64) >= t_k - 1e-9).astype(np.float64)
    raise ValueError(f"unknown curve '{spec}' (linear, power:P, step:K)")


def confidence_weights(confidence, labels, hn_index: int, min_weight: float,
                       curve: str = "linear") -> np.ndarray:
    """Per-sample loss weights from the labeller's confidence ratings.

    A rated genuine chip weighs min_weight at rating 1 rising to 1.0 at
    rating CONFIDENCE_MAX along the curve; unrated chips (0) and every hard
    negative (always 0 in the file) weigh 1.0. A sure label the model gets
    wrong costs the full loss, a doubtful one costs little - the labels
    most likely to be wrong are the ones the model is allowed to disagree
    with cheaply."""
    if not 0.0 <= min_weight <= 1.0:
        raise ValueError(f"min_weight must be in [0, 1], got {min_weight}")
    conf = np.asarray(confidence).astype(np.int64)
    labels = np.asarray(labels)
    if conf.size and (conf.min() < CONFIDENCE_UNRATED or conf.max() > CONFIDENCE_MAX):
        raise ValueError(f"confidence values must be {CONFIDENCE_UNRATED}..{CONFIDENCE_MAX}")
    f = parse_confidence_curve(curve)
    t = (np.clip(conf, 1, CONFIDENCE_MAX) - 1) / (CONFIDENCE_MAX - 1)
    w = min_weight + (1.0 - min_weight) * f(t)
    rated = (conf > CONFIDENCE_UNRATED) & (labels != hn_index)
    return np.where(rated, w, 1.0).astype(np.float32)


class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, hard_negative_index: int,
                 gamma: float = 2.0, hn_alpha: float = 0.25,
                 hn_gamma: float | None = None, hn_margin: float = 0.0):
        super().__init__()
        if not 0.0 <= hn_margin < 1.0:
            raise ValueError(f"hn_margin must be in [0, 1), got {hn_margin}")
        self.gamma = gamma
        self.hn_gamma = gamma if hn_gamma is None else hn_gamma
        self.hn_margin = float(hn_margin)
        self.hard_negative_index = hard_negative_index
        alpha = torch.ones(num_classes)
        alpha[hard_negative_index] = hn_alpha
        self.register_buffer("alpha", alpha)
        gammas = torch.full((num_classes,), float(gamma))
        gammas[hard_negative_index] = self.hn_gamma
        self.register_buffer("gammas", gammas)
        # logit adjustment (Menon et al. 2021): tau * log(prior) added to the
        # logits inside the loss only, so the network's raw output learns
        # the prior-corrected scores that validation and deployment use
        self.register_buffer("logit_offset", torch.zeros(num_classes))

    def set_logit_offset(self, offset) -> None:
        """Per-epoch hook: the additive logit offset (None or zeros = off)."""
        if offset is None:
            self.logit_offset.zero_()
        else:
            self.logit_offset.copy_(torch.as_tensor(offset, dtype=self.logit_offset.dtype))

    def set_hn_alpha(self, hn_alpha: float) -> None:
        """Ramp hook: adjust the hard-negative class weight between epochs."""
        self.alpha[self.hard_negative_index] = hn_alpha

    def set_class_alphas(self, alphas: dict[int, float]) -> None:
        """Rescue hook: set genuine-class weights (the hard_negative entry is
        owned by set_hn_alpha and ignored here). Pass a complete mapping —
        classes absent from `alphas` keep their current value."""
        for c, a in alphas.items():
            if c != self.hard_negative_index:
                self.alpha[c] = a

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits + self.logit_offset, dim=1)
        log_p_y = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_y = log_p_y.exp()
        if self.hn_margin > 0:
            # ASL probability shift for hard negatives: p_y + m, capped at
            # 1, so a negative rejected with margin to spare drops out.
            # Its log is taken from the shifted probability (>= m > 0, so
            # it is safe without the log-softmax path). The clamp below is
            # a no-op for hard negatives; it protects the genuine samples,
            # whose branch torch.where discards but still differentiates:
            # a genuine p_y that underflowed to 0 would turn log(0) into a
            # NaN gradient for the whole batch.
            is_hn = targets == self.hard_negative_index
            p_y = torch.where(is_hn, (p_y + self.hn_margin).clamp(max=1.0), p_y)
            log_p_y = torch.where(is_hn, torch.log(p_y.clamp(min=self.hn_margin)),
                                  log_p_y)
        return -self.alpha[targets] * (1 - p_y).pow(self.gammas[targets]) * log_p_y
