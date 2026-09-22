"""Checkpoint naming, discovery, and storage-tolerant file I/O.

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

Storage tolerance: slow or networked disks (a saturated HDD, an SMB/NFS
share, a scanned or synced folder) can refuse to reopen or delete a file
the process wrote moments ago, surfacing as EACCES/EPERM/EBUSY. Every
checkpoint write goes through a temp file + os.replace (readers never see
a partial file), every load/copy/delete retries transient errors with
backoff for about a minute, and housekeeping deletes that still fail are
warnings rather than a dead run.
"""

from __future__ import annotations

import errno
import math
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# errno values slow/networked storage returns while a just-written file is
# still settling (Windows sharing violations arrive as PermissionError too)
TRANSIENT_ERRNOS = {errno.EACCES, errno.EPERM, errno.EBUSY, errno.EAGAIN,
                    errno.ETXTBSY}
RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 16.0, 16.0)  # ~1 min total


def utc_stamp() -> str:
    """Filename-safe UTC timestamp (no colons, Windows-friendly)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def checkpoint_name(role: str, epoch: int, op: dict) -> str:
    spec = op.get("specificity")
    spec_txt = ("nan" if spec is None or math.isnan(spec) else f"{spec:.4f}")
    return f"{role}_e{epoch:04d}_rec{op['recall']:.4f}_spec{spec_txt}_{utc_stamp()}.pt"


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


# ---- storage-tolerant primitives -------------------------------------------

def _transient(e: OSError) -> bool:
    return isinstance(e, PermissionError) or e.errno in TRANSIENT_ERRNOS


def retry_io(fn, what: str):
    """Run fn(); on a transient OS error wait and retry with backoff,
    printing each attempt so a stalled disk shows as waiting rather than a
    silent hang. Non-transient errors (missing file, corrupt data) raise
    immediately; the last attempt's error surfaces if every retry fails."""
    for i, delay in enumerate(RETRY_DELAYS, 1):
        try:
            return fn()
        except OSError as e:
            if not _transient(e):
                raise
            print(f"[io] {what}: {e.strerror or e} - retry {i}/"
                  f"{len(RETRY_DELAYS)} in {delay:g}s", flush=True)
            time.sleep(delay)
    return fn()


def atomic_save(obj, path) -> None:
    """torch.save through a temp file in the same directory, then one
    os.replace: readers never see a half-written checkpoint, and storage
    that is still settling the previous file gets a rename instead of a
    truncate-in-place."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    retry_io(lambda: torch.save(obj, tmp), f"write {tmp.name}")
    retry_io(lambda: os.replace(tmp, path), f"replace {path.name}")


def atomic_copy(src, dst) -> None:
    """Copy via a temp file + os.replace, retrying transient errors."""
    src, dst = Path(src), Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    retry_io(lambda: shutil.copyfile(src, tmp), f"copy {src.name}")
    retry_io(lambda: os.replace(tmp, dst), f"replace {dst.name}")


def load_checkpoint(path, device):
    """torch.load (full checkpoint dict) with transient-error retries."""
    return retry_io(
        lambda: torch.load(path, map_location=device, weights_only=False),
        f"load {Path(path).name}")


def remove_quietly(path) -> None:
    """Delete a file, retrying transient errors; a delete that still fails
    is a warning, never a dead run - a stale extra checkpoint is harmless
    because discovery picks the newest file."""
    path = Path(path)
    try:
        retry_io(lambda: path.unlink(missing_ok=True), f"delete {path.name}")
    except OSError as e:
        print(f"WARNING: could not delete {path} ({e.strerror or e}); "
              "leaving it in place", flush=True)


def prune_role(run_dir, role: str, keep: Path) -> None:
    """Delete older checkpoints of this role (and any leftover temp files)
    so exactly one file remains."""
    run_dir = Path(run_dir)
    for q in run_dir.glob(f"{role}_*.pt"):
        if q != keep:
            remove_quietly(q)
    for q in run_dir.glob(f"{role}_*.pt.tmp"):
        remove_quietly(q)
    legacy = run_dir / f"{role}.pt"
    if legacy.exists() and legacy != keep:
        remove_quietly(legacy)
