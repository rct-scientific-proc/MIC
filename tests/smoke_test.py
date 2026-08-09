"""End-to-end smoke test: synthetic data -> train -> resume -> evaluate.

Runs the real CLI entry points in subprocesses (a couple of minutes on CPU,
faster with CUDA). Exits non-zero on any failure.

All outputs (dataset, checkpoints, metrics.csv, eval report and plots) are
kept in tests/runs/ for inspection; the directory is wiped and rebuilt at the
start of each run, and is gitignored.

    python tests/smoke_test.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(__file__).resolve().parent / "runs"
sys.path.insert(0, str(REPO))

from make_synthetic_h5 import make_dataset  # noqa: E402


def run(*argv) -> None:
    cmd = [sys.executable, *map(str, argv)]
    print("::", " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    # Fresh output tree each run — a stale metrics.csv or checkpoint would
    # otherwise leak into the new run via CSV append / best-key comparison.
    if OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True)
    h5 = OUT_ROOT / "smoke.h5"
    out = OUT_ROOT / "run"

    make_dataset(str(h5), num_genuine_classes=5, genuine_per_class=30,
                 hn_factor=25.0, seed=0, channels=3, image_hw=(128, 128))

    common = [
        REPO / "train.py", h5, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.8",
        "--imbalance-ratio", "3.0", "--imbalance-ratio-start", "1.0",
        "--ramp-epochs", "2", "--hn-alpha", "0.25", "--hn-alpha-end", "1.0",
        "--out-dir", out, "--patience", "0", "--seed", "1",
    ]
    # short first leg with multiprocess loading (the Windows-sensitive path)
    run(*common, "--epochs", "3", "--workers", "2")
    for name in ("last.pt", "best.pt", "metrics.csv"):
        assert (out / name).exists(), f"missing {out / name}"

    # resume and train long enough for BN stats to settle and the model to
    # learn (random init needs ~15 epochs on this data); workers=0 is much
    # faster at this size
    run(*common, "--epochs", "25", "--workers", "0",
        "--resume", out / "last.pt")

    run(REPO / "evaluate.py", out / "best.pt", h5, "--out-dir", out / "eval")
    for name in ("report.txt", "confusion.csv", "confusion.png",
                 "roc_genuine_vs_hn.png", "roc_per_class.png",
                 "calibration.png", "history.png"):
        assert (out / "eval" / name).exists(), f"missing eval output {name}"

    report = (out / "eval" / "report.txt").read_text(encoding="utf-8")
    print("\n--- report.txt ---\n" + report)

    print(f"\noutputs kept in {OUT_ROOT}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
