"""End-to-end smoke test: synthetic data -> train -> resume -> evaluate.

Runs the real CLI entry points in subprocesses (a couple of minutes on CPU,
faster with CUDA). Exits non-zero on any failure.

    python tests/smoke_test.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from make_synthetic_h5 import make_dataset  # noqa: E402


def run(*argv) -> None:
    cmd = [sys.executable, *map(str, argv)]
    print("::", " ".join(cmd[1:]))
    subprocess.run(cmd, cwd=REPO, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mic_smoke_") as tmp:
        tmp = Path(tmp)
        h5 = tmp / "smoke.h5"
        out = tmp / "run"

        make_dataset(str(h5), num_genuine_classes=2, genuine_per_class=30,
                     hn_factor=5.0, seed=0)

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
                     "roc_genuine_vs_hn.png", "roc_per_class.png", "history.png"):
            assert (out / "eval" / name).exists(), f"missing eval output {name}"

        report = (out / "eval" / "report.txt").read_text(encoding="utf-8")
        print("\n--- report.txt ---\n" + report)

    print("\nSMOKE TEST PASSED")


if __name__ == "__main__":
    main()
