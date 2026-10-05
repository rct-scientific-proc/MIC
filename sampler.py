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

# Mining scores are difficulties in [0, 1] (1 - p of the true class, see
# train_one_epoch). Fresh, never-seen hard negatives start above that range
# so they are drawn before anything already examined.
INITIAL_SCORE = 2.0


class HardNegativeMiner:
    """Tracks a per-sample EMA of training difficulty for hard negatives.

    Indexed by position within the training split (matching the `index`
    returned by H5SnippetDataset.__getitem__).
    """

    def __init__(self, labels: np.ndarray, hard_negative_index: int, ema_decay: float = 0.7):
        self.is_hn = labels == hard_negative_index
        self.ema_decay = ema_decay
        self.scores = np.full(len(labels), INITIAL_SCORE, dtype=np.float64)
        self.seen = np.zeros(len(labels), dtype=bool)

    def update(self, indices: torch.Tensor, losses: torch.Tensor) -> None:
        """Record per-sample difficulties for a training batch (any subset;
        only hard-negative entries are tracked)."""
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

    def hardest(self, n: int) -> np.ndarray:
        """Training-split positions of the n hardest hard negatives seen so
        far (highest difficulty EMA first)."""
        cand = np.flatnonzero(self.is_hn & self.seen)
        order = np.argsort(-self.scores[cand], kind="stable")
        return cand[order[:n]]

    def load_state_dict(self, state: dict) -> None:
        self.scores = np.asarray(state["scores"], dtype=np.float64).copy()
        self.seen = np.asarray(state["seen"], dtype=bool).copy()
        self.ema_decay = float(state["ema_decay"])


MINED_CSV = "mined_hard_negatives.csv"
MINED_FIELDS = ["rank", "h5_row", "train_index", "difficulty", "seen"]


def write_mined_csv(path, scores, seen, labels, h5_rows, hard_negative_index,
                    n: int) -> int:
    """mined_hard_negatives.csv: the n training hard negatives the miner
    found hardest, hardest first - the audit list for label noise. A hard
    negative that stays difficult across epochs is either a genuinely
    confusing background or an unlabelled positive; both deserve a look,
    and curate.py --rows opens exactly these snippets. Columns: rank
    (1 = hardest), h5_row (the row in the file - what curate.py shows as
    #row), train_index (position within the training split), difficulty
    (EMA of 1 - p(hard_negative), 0..1), seen (how the score was formed:
    always 1 here). Returns the number of rows written."""
    import csv
    scores = np.asarray(scores, dtype=np.float64)
    seen = np.asarray(seen, dtype=bool)
    labels = np.asarray(labels)
    cand = np.flatnonzero((labels == hard_negative_index) & seen)
    order = cand[np.argsort(-scores[cand], kind="stable")][:max(int(n), 0)]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MINED_FIELDS)
        w.writeheader()
        for rank, p in enumerate(order, start=1):
            w.writerow({"rank": rank, "h5_row": int(h5_rows[p]),
                        "train_index": int(p),
                        "difficulty": f"{scores[p]:.6f}", "seen": 1})
    return len(order)


class ClassBalancedSampler(Sampler[int]):
    """num_samples positions per epoch, each drawn by picking a class
    uniformly among those present and then a sample uniformly within it
    (with replacement): every class, hard_negative included, gets the same
    share of the epoch. The classifier re-training draw (train.py
    --crt-epochs): with the features frozen, a balanced draw re-fits the
    head's decision boundary free of the training prior."""

    def __init__(self, labels, num_samples: int, seed: int = 0):
        self.labels = np.asarray(labels)
        self.by_class = [np.flatnonzero(self.labels == c)
                         for c in np.unique(self.labels)]
        self.num_samples = int(num_samples)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed * 7_919 + self.epoch)
        cls = torch.randint(len(self.by_class), (self.num_samples,), generator=gen).numpy()
        out = np.empty(self.num_samples, dtype=np.int64)
        for k, pos in enumerate(self.by_class):
            m = cls == k
            out[m] = pos[torch.randint(len(pos), (int(m.sum()),), generator=gen).numpy()]
        return iter(out.tolist())

    def __len__(self) -> int:
        return self.num_samples


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

        self.labels = np.asarray(labels)
        self.hard_negative_index = hard_negative_index
        self.genuine_pos = np.flatnonzero(labels != hard_negative_index)
        self.hn_pos = np.flatnonzero(labels == hard_negative_index)
        if len(self.genuine_pos) == 0:
            raise ValueError("training split contains no genuine samples")

        self.ratio = ratio
        self.miner = miner
        self.random_frac = random_frac
        self.seed = seed
        self.epoch = 0
        self.genuine_repeats: dict[int, int] = {}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def set_ratio(self, ratio: float) -> None:
        """Ramp hook: adjust the hard-negative budget between epochs."""
        if ratio < 1:
            raise ValueError(f"imbalance ratio must be >= 1, got {ratio}")
        self.ratio = ratio

    def set_genuine_repeats(self, repeats: dict[int, int] | None) -> None:
        """Rescue hook: per-class oversampling for genuine classes. A class
        with repeat r appears r times per epoch (its samples are duplicated,
        not resampled). The hard-negative budget stays keyed to the UNIQUE
        genuine count, so oversampling never inflates hard-negative volume."""
        repeats = {c: int(r) for c, r in (repeats or {}).items() if int(r) > 1}
        for r in repeats.values():
            if r < 1:
                raise ValueError("repeat factors must be >= 1")
        self.genuine_repeats = repeats

    def _genuine_epoch_positions(self) -> np.ndarray:
        parts = [self.genuine_pos]
        for c, r in sorted(self.genuine_repeats.items()):
            pos_c = self.genuine_pos[self.labels[self.genuine_pos] == c]
            parts.extend([pos_c] * (r - 1))
        return np.concatenate(parts) if len(parts) > 1 else self.genuine_pos

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
        epoch_pos = np.concatenate([self._genuine_epoch_positions(), chosen_hn])
        shuffle = torch.randperm(len(epoch_pos), generator=gen).numpy()
        return iter(epoch_pos[shuffle].tolist())

    def __len__(self) -> int:
        return len(self._genuine_epoch_positions()) + self.hn_budget

    def epoch_class_counts(self, num_classes: int) -> np.ndarray:
        """How many samples of each class one epoch draws under the current
        ratio and rescue repeats: the genuine counts (times their repeat
        factors) and the hard-negative budget. This is the class prior the
        model is actually trained under - what the classifier bias is
        initialised to (train.py prior init) and what logit adjustment
        corrects for."""
        counts = np.bincount(self.labels[self._genuine_epoch_positions()],
                             minlength=num_classes).astype(np.float64)
        counts[self.hard_negative_index] = self.hn_budget
        return counts
