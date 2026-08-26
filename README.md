# mic — imbalanced snippet classifier trainer

Trains ResNet-18/34/50 classifiers on `.h5` snippet datasets (layout in
`h5_format.md`) that are heavily imbalanced and contain a dedicated
`hard_negative` class. Fully CLI-driven; all outputs are files (checkpoints,
CSV logs, PNG plots), so it works on offline machines.

## Objective & operating point

1. Reach a configurable **recall target** over the genuine (non-hard-negative)
   classes. Per-class recalls are combined by `--recall-agg`: `harmonic`
   (default — dominated by the worst classes, so one collapsed class can't
   hide behind strong ones), `macro` (arithmetic mean), or `min` (worst
   single class).
2. Subject to that, maximize **hard-negative specificity**.

Recall/specificity are threshold-dependent, so the recall target is enforced
post-hoc: every epoch, the genuineness score `s = 1 - P(hard_negative)` is
swept on the **validation** split and the largest threshold meeting the target
is chosen. A sample is accepted as genuine iff `s >= threshold`; its class is
the argmax over non-HN classes. **The threshold is stored in every checkpoint**
— inference without it is incomplete — and `evaluate.py` reuses the stored
value, never re-tuning on test data.

## Install

```
pip install -r requirements.txt
```

(For CUDA builds of torch/torchvision, install per pytorch.org first.)

## Train

```
python train.py data.h5 --arch resnet18 --epochs 50 --target-recall 0.95 \
    --imbalance-ratio 3.0 --out-dir runs/exp1
```

All options can live in a JSON config instead of the command line —
`python train.py --config exp.json` (keys = long option names, `"h5"` for
the dataset; explicit CLI flags override the file; see
`example_config.json` for a ready-to-edit starting point). Every run writes its
resolved options to `<out-dir>/config.json`, which is itself a valid
`--config`, so any run can be reproduced or branched from its output
directory.

Key options (see `python train.py --help` for all):

- `--target-recall` — recall target the threshold sweep must meet.
- `--recall-agg` — how per-class recalls aggregate for that target
  (`harmonic` default / `macro` / `min`). Example: recalls (1, 1, 1, 1, 0.1)
  give macro 0.82 but harmonic 0.36 — harmonic makes the target honest.
- `--min-threshold` — a floor no operating threshold may cross; if the target
  is only reachable beneath it, the run operates at the floor with
  `target_met=0`.
- `--threshold-mode per-class` (with `--per-class-min-count`) — one threshold
  per genuine class, applied by predicted class: easy classes keep high
  thresholds while hard ones get slack, instead of one global threshold being
  dragged down by the weakest class. Small/absent classes fall back to the
  global threshold. Per-class thresholds and per-class recall are logged each
  epoch to `class_thresholds.csv` (written in both modes).
- `--imbalance-ratio R` — at most `R x genuine` hard negatives per epoch
  (1..inf). When this forces subsampling, hard negatives are **mined**: a
  per-sample EMA of training error drives the draw, with a
  `--mining-random-frac` uniform share (`--no-mining` for uniform draws).
- `--ramp-epochs N` with `--imbalance-ratio-start` / `--hn-alpha-end` —
  start recall-focused (few hard negatives, low focal alpha on the HN class)
  and step toward full pressure; progress advances only on epochs whose
  validation meets the recall target.
- `--smart [1-5]` — adaptive alternative to the fixed ramp: cyclic
  (half-cosine) learning rate with hard-negative pressure raised/held/
  rewound at cycle troughs based on validation metrics. The value is an
  **effort level**: 1 = minimal/fast (30 epochs, short cycles, aggressive
  pressure steps), 5 = marathon (300 epochs, long cycles, tiny steps, deep
  LR anneals, many retries — slowly reaches the goal over a long horizon).
  Bare `--smart` = level 3 (balanced, rescue on). Levels preset epochs,
  `--lr-cycle-epochs`, `--lr-min`, `--pressure-step`, `--max-rewinds`,
  rescue, patience, and `--keep-top-k`; any flag passed explicitly
  overrides its preset, and the resolved config is printed at startup.
  Rewinds reload the last stable `milestone.pt` with a fresh optimizer; the
  top-K cycle checkpoints are archived in `snapshots/`. Patience counts
  cycles, not epochs, in this mode.
- `--rescue` (smart mode) — class rescue: genuine classes lagging the recall
  target at a cycle trough get a deficit-scaled focal-alpha boost
  (`--rescue-alpha-max`) and per-epoch oversampling
  (`--rescue-oversample-max`), EMA-smoothed across troughs (`--rescue-ema`)
  and dropped automatically once the class recovers. Pressure raises are
  blocked while rescue is active; boosts are logged per class in
  `class_thresholds.csv` and the eval report flags classes still under
  rescue at the end. `--class-alpha NAME=VALUE` is the manual, any-mode
  equivalent.
- `--focal-gamma`, `--hn-alpha` — focal loss shape; genuine classes have
  alpha 1.
- `--imagenet-norm` — ImageNet mean/std normalization (default: images are
  only scaled to [0,1]). Applies to eval too (recorded in the checkpoint).
- `--amp` — mixed precision on CUDA.
- `--resume runs/exp1/last.pt` — full resume (optimizer, miner state, ramp
  progress).

Outputs: metric-stamped checkpoints — one `best_*` and one `last_*` file,
named like `best_e0041_rec0.9211_spec1.0000_20260819T153042Z.pt` (role,
epoch, recall, specificity, UTC time; best is ranked by *target met →
specificity → recall*; `--resume` and `evaluate.py`/`report.py` accept the
run directory and resolve the right file) — plus `metrics.csv`,
`class_thresholds.csv`, and a timestamped `report_<UTCstamp>.pdf` (reports
accumulate rather than overwrite; `evaluate.py` also writes one for the
split it evaluates) — an
end-of-training PDF (verdict, auto-generated warnings with recommended
actions, training/controller history charts, a fresh inference pass at the
stored operating point, and a thumbnail page per class: best predictions,
worst predictions, and impostors — other labels the model routes to that
class). `--no-report` skips it; `--report-test`
uses the test split (opt-in, to keep test honest); regenerate for any past
run with `python report.py <run_dir> <data.h5>`.

Augmentation is **off by default** and opt-in per transform, with
per-entry parameters: `--augment rotation:p=0.7,degrees=30
erasing:p=0.5,scale=0.02-0.2` (train split only; validation/eval/inference
are never augmented; in a `--config` file, JSON objects like
`{"name": "erasing", "p": 0.5}` also work). Catalog: rotation,
perspective, gaussianblur, sharpness, erasing (Random Erasing/Cutout —
random black rectangle, applied post-resize), colorjitter, grayscale,
invert, posterize, solarize, autocontrast, equalize, plus the policy-based
autoaugment / randaugment / trivialaugment. Caution: the photometric ops
alter intensity — if your classes are intensity-coded they can destroy the
label; rotation, perspective, gaussianblur, sharpness, and erasing are the
safer subset there.

## Offline pretrained weights

```
# connected machine
python download_weights.py --out-dir pretrained
# offline machine (after copying the folder)
python train.py data.h5 --arch resnet50 --weights-path pretrained/resnet50.pth
```

Without `--weights-path`, torchvision's normal cache is used; `--no-pretrained`
trains from scratch.

## Evaluate

```
python evaluate.py runs/exp1/best.pt data.h5            # test split
python evaluate.py runs/exp1/best.pt data.h5 --split 1  # validation split
```

Writes `report.txt`, `confusion.csv`/`confusion.png`, `roc_genuine_vs_hn.png`
(with the operating point marked), `roc_per_class.png`, `calibration.png`
(reliability diagram of the genuineness score, with ECE), and `history.png`
(metric-vs-epoch curves from `metrics.csv`).

## Blind inference (full images)

```
python inference.py runs/exp1 photo.png scans/ \
    --window-width 128 --window-height 128 --stride-x 64
```

Slides sub-windows over whole images (Pillow; row-major, final row/column
clamped so edges are covered), classifies each window with the checkpoint's
stored thresholds, and writes `detections.csv` plus a
`report_<UTCstamp>.pdf` — a summary of all images and an annotated page
(raw window boxes, colored per class) for each image containing
non-hard-negative detections. `--grayscale` for models trained on grayscale
snippets; `--stride-y` defaults to `--stride-x`. `--gt labels.json` scores
detections against a GeoLabelling point-label export (matched by filename):
a point is hit when a same-class accepted window contains it — adds
`gt_results.csv`, green/red hit/miss markers on the overlay PNGs, and a
GT report: a color-coded verdict box and per-class table on the cover, a
score-separation histogram with the stored and fixed-recall thresholds
marked, specificity at 85/90/95/98% recall, false positives per image
with mean/std, per-class ROC curves, and per-class snippet grids: top-N highest-confidence windows
(`--top-n`), every should-have-caught window ranked by confidence
("rank 2/29"), cross-class confusions (accepted as the wrong class), and
GT windows rejected to hard_negative — borders triage the failure mode:
green = correct and accepted, yellow = right class but under the
threshold, red = wrong class. Whole-image overlays live in `assets/` as PNGs, not report pages.

## Real public datasets

```
python tests/make_real_h5.py fashionmnist fashion.h5 --hn-classes 4
python tests/make_real_h5.py cifar100 cifar.h5 --hn-classes 80 --subset-per-class 200
```

Downloads a torchvision dataset (mnist / fashionmnist / cifar10 / cifar100,
cached after the first fetch) and converts it to the h5 format, relabeling a
random `--hn-classes` subset of the original classes (or an explicit
`--hn-names` list) as `hard_negative`. More HN classes = closer to this
pipeline's heavy-imbalance regime.

## Georeferenced test scenes

```
python tests/make_geotiffs.py    # requires rasterio (test tooling only)
```

Generates GeoTIFF scenes (EPSG:3857, 0.01 m pixel) of simple shapes —
circle, square, triangle, ring — in one muted color family so geometry is
the class signal, with a `manifest.json` of every drawn shape. Label them
in GeoLabeller, export gt.json, build the h5, train, then run
`inference.py --gt` over the same images to exercise the full loop.

## Optimizing an h5 for loading

```
python optimize_h5.py in.h5 out.h5                # contiguous, uncompressed
python optimize_h5.py big_src.h5 out.h5 --resize 224   # + pre-resize
```

Rewrites any h5 with a contiguous, uncompressed images dataset so every
per-sample read is a direct offset into the file (~240x faster random reads
than h5py's default chunking under gzip; pixels byte-identical). All
project generators now write this layout natively; use this tool on files
from other sources or older versions. `--resize N` is optional and pays off
only when sources are larger than the target — for smaller sources it
inflates the file quadratically (the script warns); files already at model
size skip the resize op during training/evaluation automatically.
`--compression gzip|lzf` remains available when disk matters more than
read speed.

## Smoke test

```
python tests/smoke_test.py
```

Generates a small synthetic dataset (`tests/make_synthetic_h5.py`, which also
serves as an h5-format example), trains, resumes, and evaluates through the
real CLIs.

## Notes / gotchas

- **Windows + `--workers > 0`**: supported — the dataset opens the h5 file
  lazily per worker. Worker startup costs a few seconds per epoch, so for
  small datasets `--workers 0` is often faster.
- **Training from scratch (`--no-pretrained`)**: with few batches per epoch,
  BatchNorm running stats lag batch stats for the first ~10-15 epochs and
  validation metrics can look collapsed while train loss is near zero. They
  recover; pretrained weights (the default) largely avoid this.
- `gt` in the h5 is redundant with `labels == hard_negative`; it is checked
  for consistency at load time.
