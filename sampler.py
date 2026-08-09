"""Epoch sampling under a class-imbalance cap, with hard-negative mining hooks.

The imbalance ratio (1..inf) caps how many hard negatives the model sees per
epoch: at most `ratio * n_genuine` hard negatives are drawn each epoch (all
genuine samples are always included). When the cap forces subsampling, the
HardNegativeMiner supplies per-sample error scores so that high-error hard
negatives are drawn preferentially, mixed with a configurable uniformly-random
fraction so scores don't go stale.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Sampler

# Fresh, never-seen hard negatives start with a high score so they are drawn
# before well-classified ones. Focal losses are typically << 10.
INITIAL_SCORE = 10.0


class HardNegativeMiner:
    """Tracks a per-sample EMA of training error for hard negatives.

    Indexed by position within the training split (matching the `index`
    returned by H5SnippetDataset.__getitem__).
    """

    def __init__(self, labels: np.ndarray, hard_negative_index: int, ema_decay: float = 0.7):
        self.is_hn = labels == hard_negative_index
        self.ema_decay = ema_decay
        self.scores = np.full(len(labels), INITIAL_SCORE, dtype=np.float64)
        self.seen = np.zeros(len(labels), dtype=bool)

    def update(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        """Record per-sample losses for a training batch (any subset; only
        hard-negative entries are tracked)."""
        idx = indices.detach().cpu().numpy()
        loss = losses.detach().cpu().numpy().astype(np.float64)

        mask = self.is_hn[idx]
        idx, loss = idx[mask], loss[mask]

        first = ~self.seen[idx]
        self.scores[idx[first]] = loss[first]  # first observation replaces the prior
        rest = idx[~first]
        self.scores[rest] = self.ema_decay * self.scores[rest] + (1 - self.ema_decay) * loss[~first]
        self.seen[idx] = True

    def state_dict(self) -> dict:
        return {"scores": self.scores.copy(), "seen": self.seen.copy(), "ema_decay": self.ema_decay}

    def load_state_dict(self, state: dict) -> None:
        self.scores = np.asarray(state["scores"], dtype=np.float64).copy()
        self.seen = np.asarray(state["seen"], dtype=bool).copy()
        self.ema_decay = float(state["ema_decay"])


class ImbalanceCapSampler(Sampler[int]):
    """Yields one epoch of dataset positions: every genuine sample plus at most
    `ratio * n_genuine` hard negatives.

    Hard-negative selection: a `random_frac` share of the budget is drawn
    uniformly; the remainder is drawn by miner score (highest error first,
    via weighted sampling without replacement). Without a miner, all draws
    are uniform.

    Call set_epoch(e) before each epoch for a deterministic-but-different
    draw and shuffle order per epoch.
    """

    def __init__(
        self,
        labels: np.ndarray,
        hard_negative_index: int,
        ratio: float,
        miner: HardNegativeMiner | None = None,
        random_frac: float = 0.2,
        seed: int = 0,
    ):
        if ratio < 1:
            raise ValueError(f"imbalance ratio must be >= 1, got {ratio}")
        if not 0.0 <= random_frac <= 1.0:
            raise ValueError(f"random_frac must be in [0, 1], got {random_frac}")

        self.genuine_pos = np.flatnonzero(labels != hard_negative_index)
        self.hn_pos = np.flatnonzero(labels == hard_negative_index)
        if len(self.genuine_pos) == 0:
            raise ValueError("training split contains no genuine samples")

        self.ratio = ratio
        self.miner = miner
        self.random_frac = random_frac
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_ratio(self, ratio: float) -> None:
        """Ramp hook: adjust the hard-negative budget between epochs."""
        if ratio < 1:
            raise ValueError(f"imbalance ratio must be >= 1, got {ratio}")
        self.ratio = ratio

    @property
    def hn_budget(self) -> int:
        if np.isinf(self.ratio):
            return len(self.hn_pos)
        return min(len(self.hn_pos), int(round(self.ratio * len(self.genuine_pos))))

    def _select_hard_negatives(self, gen: torch.Generator) -> np.ndarray:
        budget = self.hn_budget
        if budget >= len(self.hn_pos):
            return self.hn_pos

        if self.miner is None:
            perm = torch.randperm(len(self.hn_pos), generator=gen).numpy()
            return self.hn_pos[perm[:budget]]

        n_random = int(round(self.random_frac * budget))
        n_mined = budget - n_random

        weights = torch.from_numpy(self.miner.scores[self.hn_pos] + 1e-8)
        mined_local = torch.multinomial(weights, n_mined, replacement=False, generator=gen).numpy()

        # Uniform draw from the hard negatives not already mined.
        remaining = np.setdiff1d(np.arange(len(self.hn_pos)), mined_local, assume_unique=False)
        perm = torch.randperm(len(remaining), generator=gen).numpy()
        random_local = remaining[perm[:n_random]]

        return self.hn_pos[np.concatenate([mined_local, random_local])]

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed * 100_003 + self.epoch)

        chosen_hn = self._select_hard_negatives(gen)
        epoch_pos = np.concatenate([self.genuine_pos, chosen_hn])
        shuffle = torch.randperm(len(epoch_pos), generator=gen).numpy()
        return iter(epoch_pos[shuffle].tolist())

    def __len__(self) -> int:
        return len(self.genuine_pos) + self.hn_budget
