"""End-to-end smoke test: synthetic data -> train -> resume -> evaluate.

Runs the real CLI entry points in subprocesses (a couple of minutes on CPU,
faster with CUDA). Exits non-zero on any failure.

All outputs (dataset, checkpoints, metrics.csv, eval report and plots) are
kept in tests/runs/ for inspection; the directory is wiped and rebuilt at the
start of each run, and is gitignored.

    python tests/smoke_test.py
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = Path(__file__).resolve().parent / "runs"
sys.path.insert(0, str(REPO))

from make_synthetic_h5 import make_dataset  # noqa: E402

# Use the GPU explicitly when present (train legs also get AMP).
GPU = ["--device", "cuda"] if torch.cuda.is_available() else []
GPU_TRAIN = GPU + (["--amp"] if GPU else [])


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
        "--batch-size", "32", "--target-recall", "0.98",
        "--imbalance-ratio", "3.0", "--imbalance-ratio-start", "1.0",
        "--ramp-epochs", "2", "--hn-alpha", "0.25", "--hn-alpha-end", "1.0",
        "--out-dir", out, "--patience", "0", "--seed", "1", *GPU_TRAIN,
    ]
    # short first leg with multiprocess loading (the Windows-sensitive path)
    run(*common, "--epochs", "3", "--workers", "2")
    for name in ("last.pt", "best.pt", "metrics.csv", "class_thresholds.csv"):
        assert (out / name).exists(), f"missing {out / name}"

    # resume and train long enough for BN stats to settle and the model to
    # learn (random init needs ~15 epochs on this data); workers=0 is much
    # faster at this size
    run(*common, "--epochs", "40", "--workers", "0",
        "--resume", out / "last.pt")

    run(REPO / "evaluate.py", out / "best.pt", h5, "--out-dir", out / "eval", *GPU)
    for name in ("report.txt", "confusion.csv", "confusion.png",
                 "roc_genuine_vs_hn.png", "roc_per_class.png",
                 "calibration.png", "history.png"):
        assert (out / "eval" / name).exists(), f"missing eval output {name}"
    assert (out / "report.pdf").exists(), "missing end-of-training PDF report"

    report = (out / "eval" / "report.txt").read_text(encoding="utf-8")
    print("\n--- report.txt ---\n" + report)

    # short per-class-threshold leg with a floor, warm-started from the
    # trained weights (resume restores the epoch counter, so extend past it)
    out_pc = OUT_ROOT / "run_per_class"
    run(REPO / "train.py", h5, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.98",
        "--threshold-mode", "per-class", "--per-class-min-count", "5",
        "--min-threshold", "0.05", "--out-dir", out_pc, "--patience", "0",
        "--seed", "1", "--epochs", "43", "--workers", "0", *GPU_TRAIN,
        "--resume", out / "last.pt")
    assert (out_pc / "class_thresholds.csv").exists()

    run(REPO / "evaluate.py", out_pc / "best.pt", h5, "--out-dir", out_pc / "eval", *GPU)
    pc_report = (out_pc / "eval" / "report.txt").read_text(encoding="utf-8")
    assert "per-class" in pc_report, "per-class mode not reflected in report"
    print("\n--- per-class report.txt ---\n" + pc_report)

    # smart-mode leg: cyclic LR + pressure controller, warm-started so the
    # target is reachable and raise/milestone events actually fire
    out_sm = OUT_ROOT / "run_smart"
    run(REPO / "train.py", h5, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.95",
        "--smart", "--lr-cycle-epochs", "3", "--pressure-step", "0.5",
        "--rescue", "--rescue-ema", "0.3",
        "--imbalance-ratio", "3.0", "--imbalance-ratio-start", "1.0",
        "--hn-alpha", "0.25", "--hn-alpha-end", "1.0",
        "--out-dir", out_sm, "--patience", "0", "--seed", "1",
        "--epochs", "52", "--workers", "0", *GPU_TRAIN,
        "--resume", out / "last.pt")
    for name in ("best.pt", "last.pt", "cycle_best.pt", "metrics.csv"):
        assert (out_sm / name).exists(), f"missing smart output {name}"
    events = [r["event"] for r in csv.DictReader(open(out_sm / "metrics.csv"))
              if r["event"]]
    assert events, "smart run produced no cycle-boundary events"
    print("smart events:", events)
    class_rows = list(csv.DictReader(open(out_sm / "class_thresholds.csv")))
    assert {"alpha", "repeat"} <= set(class_rows[0]), "missing rescue columns"
    boosts = sorted({(r["class"], r["alpha"], r["repeat"]) for r in class_rows
                     if float(r["alpha"]) > 1.0 or int(r["repeat"]) > 1})
    print("rescue boosts seen:", boosts or "none (all classes at target)")
    assert (out_sm / "report.pdf").exists(), "missing smart-run PDF report"
    assert (out_sm / "report_assets" / "timeline.png").exists()

    run(REPO / "evaluate.py", out_sm / "best.pt", h5, "--out-dir", out_sm / "eval", *GPU)
    assert (out_sm / "eval" / "report.txt").exists()

    print(f"\noutputs kept in {OUT_ROOT}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
