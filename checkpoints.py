"""Checkpoint naming and discovery.

User-facing checkpoints carry their role, epoch, headline metrics, and a UTC
timestamp in the filename:

    best_e0041_rec0.9211_spec1.0000_20260819T153042Z.pt
    last_e0051_rec0.8824_spec1.0000_20260819T153055Z.pt

Exactly one file per role is kept: each save writes the new file, then prunes
older files of the same role (including legacy fixed-name best.pt/last.pt
from older runs). Controller-internal files (cycle_best.pt, milestone.pt)
keep fixed names — they are mechanism, not deliverables.

find_checkpoint resolves what users pass on the command line: an explicit
.pt file is used as-is; a run directory resolves to its newest checkpoint of
the requested role (legacy fixed names accepted).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path


def checkpoint_name(role: str, epoch: int, op: dict) -> str:
    spec = op.get("specificity")
    spec_txt = ("nan" if spec is None or math.isnan(spec) else f"{spec:.4f}")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{role}_e{epoch:04d}_rec{op['recall']:.4f}_spec{spec_txt}_{stamp}.pt"


def find_checkpoint(path, role: str) -> Path | None:
    """Resolve a checkpoint argument.

    A .pt file path is returned as-is; a directory is searched for the newest
    '{role}_*.pt' (falling back to the legacy fixed name '{role}.pt').
    Returns None when nothing matches.
    """
    p = Path(path)
    if p.is_file():
        return p
    if p.is_dir():
        cands = sorted(p.glob(f"{role}_*.pt"), key=lambda q: q.stat().st_mtime)
        if cands:
            return cands[-1]
        legacy = p / f"{role}.pt"
        if legacy.exists():
            return legacy
    return None


def prune_role(run_dir, role: str, keep: Path) -> None:
    """Delete older checkpoints of this role so exactly one file remains."""
    run_dir = Path(run_dir)
    for q in run_dir.glob(f"{role}_*.pt"):
        if q != keep:
            q.unlink(missing_ok=True)
    legacy = run_dir / f"{role}.pt"
    if legacy.exists() and legacy != keep:
        legacy.unlink()
