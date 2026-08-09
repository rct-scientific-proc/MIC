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

Key options (see `python train.py --help` for all):

- `--target-recall` — recall target the threshold sweep must meet.
- `--recall-agg` — how per-class recalls aggregate for that target
  (`harmonic` default / `macro` / `min`). Example: recalls (1, 1, 1, 1, 0.1)
  give macro 0.82 but harmonic 0.36 — harmonic makes the target honest.
- `--imbalance-ratio R` — at most `R x genuine` hard negatives per epoch
  (1..inf). When this forces subsampling, hard negatives are **mined**: a
  per-sample EMA of training error drives the draw, with a
  `--mining-random-frac` uniform share (`--no-mining` for uniform draws).
- `--ramp-epochs N` with `--imbalance-ratio-start` / `--hn-alpha-end` —
  start recall-focused (few hard negatives, low focal alpha on the HN class)
  and step toward full pressure; progress advances only on epochs whose
  validation meets the recall target.
- `--focal-gamma`, `--hn-alpha` — focal loss shape; genuine classes have
  alpha 1.
- `--imagenet-norm` — ImageNet mean/std normalization (default: images are
  only scaled to [0,1]). Applies to eval too (recorded in the checkpoint).
- `--amp` — mixed precision on CUDA.
- `--resume runs/exp1/last.pt` — full resume (optimizer, miner state, ramp
  progress).

Outputs: `last.pt`, `best.pt` (best by *target met → specificity → macro
recall*), `metrics.csv`. No augmentations are applied by design.

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
