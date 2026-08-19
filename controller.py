"""Smart training controller: cyclic LR + adaptive hard-negative pressure.

Training is structured as LR cycles (half-cosine from lr_max down to lr_min,
then restart). Control decisions happen only at cycle boundaries, where the
model sits in a settled low-LR state:

  raise    the cycle met the recall target and matched-or-beat the last
           milestone -> the cycle's best checkpoint becomes the new
           milestone, pressure steps up, next cycle restarts at lr_max
           (the high LR helps absorb the pressure increase)
  hold     the cycle met the target but regressed vs the milestone (or
           there is nothing to raise) -> another cycle at the same pressure
  rewind   the cycle never met the target -> reload the milestone weights
           with a FRESH optimizer (stale Adam moments would re-poison the
           weights), halve the pressure step, and retry from the stable
           pressure plus the smaller step. Miner scores survive: they are
           what makes the retry smarter than the first attempt.
  ceiling  more than max_rewinds failures at one level -> accept the stable
           pressure as this run's ceiling and keep training there

Pressure p in [0, 1] maps through the same endpoint flags as the ramp
(--hn-alpha -> --hn-alpha-end, --imbalance-ratio-start -> --imbalance-ratio).

The controller also keeps a top-K snapshot archive (snapshots/) of the best
cycle checkpoints across the run, each carrying its own thresholds.

Class rescue (optional): at each trough the per-class validation recalls are
EMA-smoothed; classes lagging the target by more than a small deadband get a
loss-weight boost (up to rescue_alpha_max) and an oversampling repeat factor
(up to rescue_oversample_max) for the next cycle, both scaled by the deficit
and recomputed statelessly so a recovered class sheds its boost. A cycle
trained under active rescue never counts as stable enough for a pressure
raise — pressure and rescue never change in the same step.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path


class SmartController:
    # A class is rescued only when its EMA recall sits below the target by
    # more than this fraction of the target — keeps hairline deficits from
    # blocking pressure raises.
    RESCUE_DEADBAND = 0.02

    def __init__(self, lr_max: float, lr_min: float, cycle_epochs: int,
                 pressure_step: float, max_rewinds: int, keep_top_k: int,
                 rescue: bool = False, rescue_alpha_max: float = 3.0,
                 rescue_oversample_max: int = 3, rescue_ema: float = 0.5,
                 target_recall: float | None = None):
        if cycle_epochs < 1:
            raise ValueError("cycle_epochs must be >= 1")
        if rescue and target_recall is None:
            raise ValueError("rescue requires target_recall")
        self.lr_max = lr_max
        self.lr_min = lr_min
        self.cycle_epochs = cycle_epochs
        self.max_rewinds = max_rewinds
        self.keep_top_k = keep_top_k

        self.rescue = rescue
        self.rescue_alpha_max = rescue_alpha_max
        self.rescue_oversample_max = rescue_oversample_max
        self.rescue_ema = rescue_ema
        self.target_recall = target_recall
        self.ema_recall: dict[int, float] = {}
        self.rescue_alphas: dict[int, float] = {}   # boosts for the NEXT cycle
        self.rescue_repeats: dict[int, int] = {}
        self.rescue_active = False                  # a class was boosted this cycle

        self.step = pressure_step
        self.p_stable = 0.0   # pressure of the current milestone
        self.p_try = 0.0      # pressure being trained at
        self.cycle = 0
        self.epoch_in_cycle = 0
        self.rewinds = 0      # failures at the current level
        self.ceiling = False
        self.milestone_key = None    # selection key of milestone.pt
        self.cycle_best_key = None   # selection key of cycle_best.pt
        self.cycles_since_best = 0   # cycle-granular patience counter
        self.snapshots: list[list] = []  # [[key, path], ...]

    # --- per-epoch -------------------------------------------------------

    def lr_at(self) -> float:
        """Half-cosine within the cycle: lr_max at epoch 0, lr_min at the
        trough (last epoch of the cycle)."""
        t = self.epoch_in_cycle
        denom = max(self.cycle_epochs - 1, 1)
        return self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
            1 + math.cos(math.pi * min(t, denom) / denom))

    def observe(self, key: tuple) -> bool:
        """Record an epoch's validation key; True if it is a new cycle best
        (caller then writes cycle_best.pt)."""
        if self.cycle_best_key is None or key > tuple(self.cycle_best_key):
            self.cycle_best_key = list(key)
            return True
        return False

    def at_boundary(self) -> bool:
        """Call after incrementing epoch_in_cycle."""
        return self.epoch_in_cycle >= self.cycle_epochs

    # --- cycle boundary --------------------------------------------------

    def end_cycle(self, out_dir, per_class_recall: dict | None = None) -> str:
        """Decide at the trough; returns the event. Events 'rewind' and
        'ceiling' require the caller to reload milestone.pt into the model
        and build a fresh optimizer.

        `per_class_recall` (the trough epoch's per-class validation recalls)
        feeds the rescue logic: lagging classes get loss/exposure boosts for
        the next cycle, and a cycle trained under active rescue never counts
        as stable enough for a pressure raise."""
        out_dir = Path(out_dir)
        cb = tuple(self.cycle_best_key) if self.cycle_best_key is not None else None
        met = cb is not None and bool(cb[0])

        if met and (self.milestone_key is None or cb >= tuple(self.milestone_key)):
            shutil.copyfile(out_dir / "cycle_best.pt", out_dir / "milestone.pt")
            self.milestone_key = list(cb)
            self.p_stable = self.p_try
            self.rewinds = 0
            if not self.ceiling and self.p_try < 1.0 and not self.rescue_active:
                self.p_try = min(1.0, self.p_try + self.step)
                event = "raise"
            else:
                event = "hold"
        elif met or self.milestone_key is None or self.ceiling:
            event = "hold"  # regressed-but-met, nothing to rewind to, or capped
        else:
            self.rewinds += 1
            if self.rewinds > self.max_rewinds:
                self.p_try = self.p_stable
                self.ceiling = True
                event = "ceiling"
            else:
                self.step /= 2
                self.p_try = self.p_stable + self.step
                event = "rewind"

        self._snapshot(out_dir)
        if self.rescue and per_class_recall:
            self._update_rescue(per_class_recall)
        self.cycle += 1
        self.epoch_in_cycle = 0
        self.cycle_best_key = None
        return event

    def _update_rescue(self, per_class_recall: dict) -> None:
        """EMA the trough recalls, then recompute next cycle's boosts
        statelessly from the current deficits — a recovered class loses its
        boost automatically, so there is no see-saw bookkeeping."""
        for c, r in per_class_recall.items():
            prev = self.ema_recall.get(c)
            self.ema_recall[c] = (r if prev is None else
                                  self.rescue_ema * prev + (1 - self.rescue_ema) * r)

        alphas, repeats = {}, {}
        for c, ema in self.ema_recall.items():
            deficit = (self.target_recall - ema) / max(self.target_recall, 1e-9)
            deficit = min(1.0, deficit)
            if deficit < self.RESCUE_DEADBAND:
                continue
            alphas[c] = 1.0 + (self.rescue_alpha_max - 1.0) * deficit
            rep = 1 + round((self.rescue_oversample_max - 1) * deficit)
            if rep > 1:
                repeats[c] = rep
        self.rescue_alphas = alphas
        self.rescue_repeats = repeats
        self.rescue_active = bool(alphas or repeats)

    def _snapshot(self, out_dir: Path) -> None:
        """Keep the top-K cycle-best checkpoints across the run."""
        src = out_dir / "cycle_best.pt"
        if self.cycle_best_key is None or not src.exists():
            return
        key = tuple(self.cycle_best_key)
        if len(self.snapshots) >= self.keep_top_k:
            worst = min(self.snapshots, key=lambda s: tuple(s[0]))
            if tuple(worst[0]) >= key:
                return
            self.snapshots.remove(worst)
            Path(worst[1]).unlink(missing_ok=True)
        dst = out_dir / "snapshots" / f"cycle{self.cycle:03d}.pt"
        dst.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, dst)
        self.snapshots.append([list(key), str(dst)])

    # --- persistence -----------------------------------------------------

    _STATE_FIELDS = ("lr_max", "lr_min", "cycle_epochs", "max_rewinds",
                     "keep_top_k", "step", "p_stable", "p_try", "cycle",
                     "epoch_in_cycle", "rewinds", "ceiling", "milestone_key",
                     "cycle_best_key", "cycles_since_best", "snapshots",
                     "rescue", "rescue_alpha_max", "rescue_oversample_max",
                     "rescue_ema", "target_recall", "ema_recall",
                     "rescue_alphas", "rescue_repeats", "rescue_active")

    def state_dict(self) -> dict:
        return {f: getattr(self, f) for f in self._STATE_FIELDS}

    def load_state_dict(self, state: dict, out_dir=None) -> None:
        for f in self._STATE_FIELDS:
            if f in state:
                setattr(self, f, state[f])
        if out_dir is not None:
            # Resuming into a fresh directory: forget files that don't exist.
            out_dir = Path(out_dir)
            if self.milestone_key is not None and not (out_dir / "milestone.pt").exists():
                self.milestone_key = None
            self.snapshots = [s for s in self.snapshots if Path(s[1]).exists()]
