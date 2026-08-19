"""Multiclass focal loss with per-class alpha weighting.

FL(p_y) = -alpha[y] * (1 - p_y)^gamma * log(p_y)

The hard_negative class gets its own alpha (typically < 1 early in training,
optionally ramped upward) so the loss stays recall-focused on genuine classes
while hard-negative pressure is applied gradually.

forward() returns per-sample losses (reduction='none') because the training
loop feeds them to the hard-negative miner; call .mean() for the batch loss.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, num_classes: int, hard_negative_index: int,
                 gamma: float = 2.0, hn_alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.hard_negative_index = hard_negative_index
        alpha = torch.ones(num_classes)
        alpha[hard_negative_index] = hn_alpha
        self.register_buffer("alpha", alpha)

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
        log_p = F.log_softmax(logits, dim=1)
        log_p_y = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_y = log_p_y.exp()
        return -self.alpha[targets] * (1 - p_y).pow(self.gamma) * log_p_y
