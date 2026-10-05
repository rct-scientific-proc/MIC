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

import torch
from torch import nn
from torch.nn import functional as F


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
            # it is safe without the log-softmax path).
            is_hn = targets == self.hard_negative_index
            p_y = torch.where(is_hn, (p_y + self.hn_margin).clamp(max=1.0), p_y)
            log_p_y = torch.where(is_hn, torch.log(p_y), log_p_y)
        return -self.alpha[targets] * (1 - p_y).pow(self.gammas[targets]) * log_p_y
