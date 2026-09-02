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
import importlib.util
import json
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


def ck(run_dir: Path, role: str) -> Path:
    """The single role checkpoint (metric-stamped filename)."""
    hits = list(run_dir.glob(f"{role}_*.pt"))
    assert len(hits) == 1, f"expected one {role} checkpoint in {run_dir}: {hits}"
    return hits[0]


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
        # label-safe for brightness bands; parameterized specs + post-resize
        "--augment", "rotation:p=0.6,degrees=10", "gaussianblur",
        "erasing:p=0.25,scale=0.02-0.1",
        "--out-dir", out, "--patience", "0", "--seed", "1", *GPU_TRAIN,
    ]
    # short first leg with multiprocess loading (the Windows-sensitive path)
    run(*common, "--epochs", "3", "--workers", "2")
    for name in ("metrics.csv", "class_thresholds.csv"):
        assert (out / name).exists(), f"missing {out / name}"
    print("checkpoints:", ck(out, "best").name, "/", ck(out, "last").name)

    # resume and train long enough for BN stats to settle and the model to
    # learn (random init needs ~15 epochs on this data); workers=0 is much
    # faster at this size. Passing the run DIRECTORY exercises checkpoint
    # discovery (newest last_*).
    run(*common, "--epochs", "40", "--workers", "0", "--resume", out)

    # evaluate also accepts the run directory (newest best_*)
    run(REPO / "evaluate.py", out, h5, "--out-dir", out / "eval", *GPU)
    for name in ("report.txt", "confusion.csv", "confusion.png",
                 "roc_genuine_vs_hn.png", "roc_per_class.png",
                 "calibration.png", "history.png"):
        assert (out / "eval" / name).exists(), f"missing eval output {name}"
    assert list(out.glob("report_*.pdf")), "missing end-of-training PDF report"
    assert list((out / "eval").glob("report_*.pdf")), "missing evaluation PDF report"

    report = (out / "eval" / "report.txt").read_text(encoding="utf-8")
    print("\n--- report.txt ---\n" + report)

    # the emitted config.json is itself a valid --config: one extra epoch
    # driven entirely by it (CLI overrides only out-dir/epochs/resume)
    assert (out / "config.json").exists(), "missing resolved config.json"
    out_cfg = OUT_ROOT / "run_config"
    run(REPO / "train.py", "--config", out / "config.json",
        "--out-dir", out_cfg, "--epochs", "41", "--resume", out,
        "--no-report")
    assert (out_cfg / "metrics.csv").exists()
    assert (out_cfg / "config.json").exists()

    # short per-class-threshold leg with a floor, warm-started from the
    # trained weights (resume restores the epoch counter, so extend past it)
    out_pc = OUT_ROOT / "run_per_class"
    run(REPO / "train.py", h5, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.98",
        "--threshold-mode", "per-class", "--per-class-min-count", "5",
        "--min-threshold", "0.05", "--out-dir", out_pc, "--patience", "0",
        "--seed", "1", "--epochs", "43", "--workers", "0", *GPU_TRAIN,
        "--resume", out)
    assert (out_pc / "class_thresholds.csv").exists()

    run(REPO / "evaluate.py", out_pc, h5, "--out-dir", out_pc / "eval", *GPU)
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
        "--resume", out)
    for name in ("cycle_best.pt", "metrics.csv"):
        assert (out_sm / name).exists(), f"missing smart output {name}"
    ck(out_sm, "best")
    ck(out_sm, "last")
    events = [r["event"] for r in csv.DictReader(open(out_sm / "metrics.csv"))
              if r["event"]]
    assert events, "smart run produced no cycle-boundary events"
    print("smart events:", events)
    class_rows = list(csv.DictReader(open(out_sm / "class_thresholds.csv")))
    assert {"alpha", "repeat"} <= set(class_rows[0]), "missing rescue columns"
    boosts = sorted({(r["class"], r["alpha"], r["repeat"]) for r in class_rows
                     if float(r["alpha"]) > 1.0 or int(r["repeat"]) > 1})
    print("rescue boosts seen:", boosts or "none (all classes at target)")
    assert list(out_sm.glob("report_*.pdf")), "missing smart-run PDF report"
    assert (out_sm / "report_assets" / "timeline.png").exists()

    run(REPO / "evaluate.py", out_sm, h5, "--out-dir", out_sm / "eval", *GPU)
    assert (out_sm / "eval" / "report.txt").exists()

    # blind sliding-window inference: composite scene with two planted band
    # patches on a hard-negative background (brightness values match
    # make_synthetic_h5's bands)
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(7)

    def noisy(base, hgt, wid):
        return (base + rng.integers(-20, 21, (hgt, wid, 3))).clip(0, 255).astype(np.uint8)

    scene = noisy(152, 512, 512)                    # midpoint = hard negative
    scene[64:192, 64:192] = noisy(30, 128, 128)     # band0
    scene[250:378, 300:428] = noisy(225, 128, 128)  # band4
    scenes = OUT_ROOT / "scenes"
    scenes.mkdir()
    Image.fromarray(scene).save(scenes / "scene.png")
    Image.fromarray(noisy(152, 400, 400)).save(scenes / "clean.png")

    # ground truth in the GeoLabelling export schema: patch centres should
    # hit, a background point and an unknown class should not
    import json
    gt = {
        "version": "3.2", "classes": ["band0", "band4", "mystery"],
        "images": [{
            "path": "D:\\labeller\\scene.png", "name": "scene",
            "group": "t", "original_width": 512, "original_height": 512,
            "labels": [
                {"id": 1, "class_name": "band0", "pixel_x": 128 / 512,
                 "pixel_y": 128 / 512},
                {"id": 2, "class_name": "band4", "pixel_x": 364 / 512,
                 "pixel_y": 314 / 512},
                {"id": 3, "class_name": "band0", "pixel_x": 480 / 512,
                 "pixel_y": 40 / 512},
                {"id": 4, "class_name": "mystery", "pixel_x": 0.02,
                 "pixel_y": 0.02},
            ]}],
        "_next_id": 5,
    }
    gt_path = OUT_ROOT / "gt.json"
    gt_path.write_text(json.dumps(gt))

    out_inf = OUT_ROOT / "inference"
    run(REPO / "inference.py", out_sm, scenes, "--window-width", "128",
        "--window-height", "128", "--stride-x", "64", "--gt", gt_path,
        "--out-dir", out_inf, *GPU)
    det = list(csv.DictReader(open(out_inf / "detections.csv")))
    assert det, "no detections on the planted scene"
    assert all("clean.png" not in r["image"] for r in det), \
        "false positives on the clean scene"
    assert {r["class"] for r in det} <= {"band0", "band4"}, det
    assert list(out_inf.glob("report_*.pdf")), "missing inference PDF report"
    print("inference detections:",
          [(r["class"], r["x"], r["y"]) for r in det])

    gt_rows = {int(r["label_id"]): r
               for r in csv.DictReader(open(out_inf / "gt_results.csv"))}
    assert len(gt_rows) == 4
    assert any(gt_rows[i]["hit"] == "1" for i in (1, 2)), \
        "no planted gt point was hit"
    assert gt_rows[3]["hit"] == "0", "background gt point counted as hit"
    assert gt_rows[4]["known_class"] == "0", "unknown class not flagged"
    print("gt hits:", {i: gt_rows[i]["hit"] for i in sorted(gt_rows)})

    # optimize_h5: repack + pre-resize to model size, then train an epoch on
    # the resized file (exercises the skip-resize dataset path)
    out_opt = OUT_ROOT / "smoke_opt.h5"
    run(REPO / "optimize_h5.py", h5, out_opt, "--resize", "224", "--no-progress")
    run(REPO / "train.py", out_opt, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.5", "--epochs", "1",
        "--out-dir", OUT_ROOT / "run_opt", "--no-report", "--patience", "0",
        "--seed", "1", *GPU_TRAIN)
    assert (OUT_ROOT / "run_opt" / "metrics.csv").exists()

    # --optuna: a two-trial study on the tiny dataset, then the best trial's
    # config.json reproduces a plain run (optuna is an optional dependency)
    if importlib.util.find_spec("optuna") is not None:
        out_hp = OUT_ROOT / "run_optuna"
        run(REPO / "train.py", h5, "--arch", "resnet18", "--no-pretrained",
            "--batch-size", "32", "--target-recall", "0.5", "--epochs", "2",
            "--out-dir", out_hp, "--optuna", "2", "--optuna-prune-warmup", "0",
            "--patience", "0", "--seed", "1", "--no-progress", *GPU_TRAIN)
        assert (out_hp / "trials.csv").exists(), "trials.csv missing"
        best = json.loads((out_hp / "best_trial.json").read_text(encoding="utf-8"))
        assert Path(best["config"]).exists(), "best trial config.json missing"
        assert len(list(out_hp.glob("trial_*"))) == 2
        run(REPO / "train.py", "--config", best["config"], "--epochs", "1",
            "--out-dir", OUT_ROOT / "run_optuna_repro", "--no-report",
            "--no-progress", *GPU_TRAIN)
        assert (OUT_ROOT / "run_optuna_repro" / "metrics.csv").exists()
        assert not (OUT_ROOT / "run_optuna_repro" / "trials.csv").exists(), \
            "reproducing a trial config relaunched a study"
        print("optuna study: best trial", best["trial"], "value", best["value"])

        # pinning + comment keys: an explicit CLI flag removes its dimension
        # from a custom space (single note), "_"-keys are comments, and the
        # trial's recorded values match its config.json (effective values)
        space = OUT_ROOT / "space.json"
        space.write_text(json.dumps({
            "_note": "smoke space",
            "lr": {"type": "float", "low": 1e-4, "high": 1e-3, "log": True},
            "batch-size": {"type": "categorical", "choices": [16, 64]},
        }), encoding="utf-8")
        out_pin = OUT_ROOT / "run_optuna_pin"
        run(REPO / "train.py", h5, "--arch", "resnet18", "--no-pretrained",
            "--batch-size", "32", "--target-recall", "0.5", "--epochs", "1",
            "--out-dir", out_pin, "--optuna", "1", "--optuna-space", space,
            "--patience", "0", "--seed", "1", "--no-progress", *GPU_TRAIN)
        header, row = (out_pin / "trials.csv").read_text(
            encoding="utf-8").splitlines()[:2]
        cols = dict(zip(header.split(","), row.split(",")))
        cfg0 = json.loads((out_pin / "trial_000" / "config.json")
                          .read_text(encoding="utf-8"))
        assert cfg0["batch_size"] == 32, "CLI pin did not hold in the trial"
        assert cols["batch_size"] == "32", \
            "pinned value missing from trials.csv effective columns"
        assert float(cols["lr"]) == cfg0["lr"], \
            "trials.csv lr does not match the trial's config.json"
    else:
        print("optuna not installed - search leg skipped")

    # custom augmentations shipped as a plugin file: the example plugin's
    # entries (incl. a post-resize one) must train an epoch end to end
    run(REPO / "train.py", h5, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.5", "--epochs", "1",
        "--out-dir", OUT_ROOT / "run_plugin", "--no-report", "--patience",
        "0", "--class-alpha-auto", "0.99",
        "--augment-plugin", REPO / "example_augment_plugin.py",
        "--augment", "hflip:p=1.0", "gaussnoise:p=1.0,sigma=5",
        "gridmask:p=1.0", "coldrop:p=1.0,frac=0.2", "--seed", "1", "--no-progress", *GPU_TRAIN)
    assert (OUT_ROOT / "run_plugin" / "metrics.csv").exists()

    # non-uint8 storage: uint16 grayscale trains end to end; a float32
    # [0,1] file trains and survives an optimize round trip byte-true
    h5_u16 = OUT_ROOT / "smoke_u16.h5"
    make_dataset(str(h5_u16), num_genuine_classes=3, genuine_per_class=20,
                 hn_factor=10.0, seed=3, channels=1, image_hw=(64, 64),
                 dtype="uint16")
    run(REPO / "train.py", h5_u16, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.5", "--epochs", "1",
        "--out-dir", OUT_ROOT / "run_u16", "--no-report", "--patience", "0",
        "--seed", "1", "--no-progress", *GPU_TRAIN)
    assert (OUT_ROOT / "run_u16" / "metrics.csv").exists()

    import h5py
    import numpy as np
    h5_f32 = OUT_ROOT / "smoke_f32.h5"
    with h5py.File(h5, "r") as a, h5py.File(h5_f32, "w") as b:
        b["images"] = a["images"][:].astype(np.float32) / 255.0
        for k in ("labels", "gt", "split"):
            b[k] = a[k][:]
        b["classes"] = np.array(a["classes"].asstr()[:], dtype=object)
    run(REPO / "train.py", h5_f32, "--arch", "resnet18", "--no-pretrained",
        "--batch-size", "32", "--target-recall", "0.5", "--epochs", "1",
        "--out-dir", OUT_ROOT / "run_f32", "--no-report", "--patience", "0",
        "--seed", "1", "--no-progress", *GPU_TRAIN)
    run(REPO / "optimize_h5.py", h5_f32, OUT_ROOT / "smoke_f32_opt.h5",
        "--no-progress")
    with h5py.File(OUT_ROOT / "smoke_f32_opt.h5", "r") as f, \
            h5py.File(h5_f32, "r") as s:
        assert f["images"].dtype == np.float32, f["images"].dtype
        assert np.array_equal(f["images"][:8], s["images"][:8]), \
            "optimize changed float pixel values"

    # curate: snippet-removal GUI - self-test does a remove/save/restore
    # round-trip on a copy (removed mask honored by loaders, file left clean)
    h5_cur = OUT_ROOT / "curate_smoke.h5"
    shutil.copyfile(h5_f32, h5_cur)
    if importlib.util.find_spec("PyQt5") is not None:
        run(REPO / "curate.py", h5_cur, "--self-test", OUT_ROOT / "gui_curate")
        assert (OUT_ROOT / "gui_curate" / "removed_view.png").exists()
    else:
        r = subprocess.run([sys.executable, str(REPO / "curate.py"), str(h5_cur)],
                           capture_output=True, text=True, cwd=REPO)
        assert "pip install PyQt5" in r.stdout + r.stderr

    # gui: optional PyQt5 front-end - the self-test renders offscreen and
    # asserts the commands each tab builds (without PyQt5, the entry point
    # must print the install hint instead of a traceback)
    if importlib.util.find_spec("PyQt5") is not None:
        run(REPO / "gui.py", "--self-test", OUT_ROOT / "gui")
        assert (OUT_ROOT / "gui" / "train_tab.png").exists()
    else:
        r = subprocess.run([sys.executable, str(REPO / "gui.py")],
                           capture_output=True, text=True, cwd=REPO)
        assert "pip install PyQt5" in r.stdout + r.stderr, \
            "missing-PyQt5 hint not printed"
        print("PyQt5 not installed - gui install hint verified")

    print(f"\noutputs kept in {OUT_ROOT}")
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
