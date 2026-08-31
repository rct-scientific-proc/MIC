"""Train a ResNet on an h5 snippet dataset (see h5_format.md, TODO.md).

Objective: reach the target macro recall over genuine classes, then maximize
hard-negative specificity at that recall. The recall target is enforced by a
per-epoch threshold sweep on the validation split; the chosen threshold is
stored in every checkpoint alongside the weights.

Outputs in --out-dir:
    last.pt       most recent checkpoint (resumable)
    best.pt       best by (target met, specificity, macro recall)
    metrics.csv   per-epoch log

Example:
    python train.py data.h5 --arch resnet18 --epochs 50 --target-recall 0.95 \
        --imbalance-ratio 2.0 --out-dir runs/exp1
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from checkpoints import checkpoint_name, find_checkpoint, prune_role
from controller import SmartController
from dataset import (AUGMENTATIONS, SPLIT_TRAIN, SPLIT_VAL, H5SnippetDataset,
                     load_augmentation_plugins, validate_h5)
from losses import FocalLoss
from metrics import (RECALL_AGGREGATES, collect_probs, genuine_vs_hn_roc,
                     sweep_class_thresholds, sweep_threshold)
from model import ARCHS, build_model
from sampler import HardNegativeMiner, ImbalanceCapSampler

# --smart level presets: 1 = minimal/fast, 5 = marathon (slowly reach the
# goal over a long horizon). Explicit flags always override their preset.
SMART_PRESETS = {
    1: dict(epochs=30, lr_cycle_epochs=3, lr_min_div=10, pressure_step=0.50,
            max_rewinds=1, keep_top_k=2, rescue=False, rescue_ema=0.5,
            patience=3),
    2: dict(epochs=50, lr_cycle_epochs=6, lr_min_div=20, pressure_step=0.35,
            max_rewinds=2, keep_top_k=3, rescue=False, rescue_ema=0.5,
            patience=4),
    3: dict(epochs=80, lr_cycle_epochs=10, lr_min_div=25, pressure_step=0.25,
            max_rewinds=3, keep_top_k=3, rescue=True, rescue_ema=0.5,
            patience=5),
    4: dict(epochs=150, lr_cycle_epochs=15, lr_min_div=50, pressure_step=0.15,
            max_rewinds=4, keep_top_k=5, rescue=True, rescue_ema=0.6,
            patience=8),
    5: dict(epochs=300, lr_cycle_epochs=20, lr_min_div=100, pressure_step=0.10,
            max_rewinds=6, keep_top_k=8, rescue=True, rescue_ema=0.7,
            patience=12),
}

# --recall-first level presets: how hard training initially chases the
# recall target before hard-negative pressure (specificity) is allowed to
# matter. 1 = brief soft-start, 5 = long recall-only opening that starts
# with almost no hard-negative influence. Values preset the pressure
# endpoints; in fixed-ramp mode the ramp length too (smart mode paces
# pressure itself at cycle troughs). Explicit flags always override.
RECALL_FIRST_PRESETS = {
    1: dict(ramp_epochs=3, hn_alpha=0.15, hn_alpha_end=0.25,
            imbalance_ratio_start=None),
    2: dict(ramp_epochs=6, hn_alpha=0.10, hn_alpha_end=0.25,
            imbalance_ratio_start=2.0),
    3: dict(ramp_epochs=12, hn_alpha=0.05, hn_alpha_end=0.25,
            imbalance_ratio_start=1.0),
    4: dict(ramp_epochs=20, hn_alpha=0.02, hn_alpha_end=0.25,
            imbalance_ratio_start=1.0),
    5: dict(ramp_epochs=35, hn_alpha=0.01, hn_alpha_end=0.25,
            imbalance_ratio_start=1.0),
}

# --optuna default search space, in the --optuna-space JSON format:
# {option: spec} where spec is {"type": "float"|"int"|"categorical", "low",
# "high", "log", "step", "choices"}. Most train.py options can be searched;
# a null choice leaves the option at its config/preset value, and an option
# passed explicitly on the study command line pins its value (the dimension
# leaves the space).
OPTUNA_DEFAULT_SPACE = {
    "lr": {"type": "float", "low": 1e-5, "high": 1e-2, "log": True},
    "weight_decay": {"type": "float", "low": 1e-6, "high": 1e-2, "log": True},
    "optimizer": {"type": "categorical", "choices": ["adamw", "sgd"]},
    "focal_gamma": {"type": "float", "low": 0.5, "high": 4.0},
    "hn_alpha_end": {"type": "float", "low": 0.1, "high": 1.0},
    "recall_first": {"type": "categorical", "choices": [None, 1, 2, 3, 4, 5]},
}
OPTUNA_KEYS = ("optuna", "optuna_space", "optuna_storage",
               "optuna_prune_warmup", "optuna_no_prune")
# options the study itself manages per trial - never searchable
OPTUNA_RESERVED = {"h5", "config", "out_dir", "no_report", "resume"}

CSV_FIELDS = [
    "epoch", "train_loss", "threshold", "threshold_mode", "thr_min", "thr_max",
    "target_met", "recall", "recall_agg", "specificity", "max_recall", "auroc",
    "hn_alpha", "imbalance_ratio", "ramp_progress", "pressure", "cycle",
    "event", "lr", "epoch_time_s",
]
CLASS_CSV_FIELDS = ["epoch", "class", "threshold", "recall", "predicted_n",
                    "accepted_n", "fallback", "alpha", "repeat"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("h5", nargs="?", default=None,
                   help="dataset .h5 file (h5_format.md); may instead be "
                        "given in --config as \"h5\"")
    p.add_argument("--config", default=None, metavar="FILE.json",
                   help="JSON file of options (keys = long option names, "
                        "dashes or underscores; keys starting with _ are "
                        "comments). Precedence: explicit CLI "
                        "flags > config file > --smart level presets > "
                        "defaults. Every run writes its resolved options to "
                        "<out-dir>/config.json, which is itself a valid "
                        "--config file")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: runs/<timestamp>)")

    m = p.add_argument_group("model")
    m.add_argument("--arch", choices=sorted(ARCHS), default="resnet18")
    m.add_argument("--no-pretrained", action="store_true",
                   help="random init instead of ImageNet weights")
    m.add_argument("--weights-path", default=None,
                   help="local ImageNet .pth (from download_weights.py) for offline use")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=None,
                   help="training epochs (default: 50, or per --smart level)")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw",
                   help="adamw (default: decoupled weight decay, forgiving of "
                        "the LR choice) or sgd with momentum - the classic "
                        "from-scratch CNN choice, which typically wants a "
                        "~10x higher --lr than adamw's default")
    t.add_argument("--momentum", type=float, default=0.9,
                   help="sgd only: momentum (nesterov when > 0)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    t.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    t.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    t.add_argument("--patience", type=int, default=None,
                   help="early-stop after N epochs (smart mode: N cycles) "
                        "without improvement; 0 = off (default: 10, or per "
                        "--smart level)")
    t.add_argument("--no-progress", action="store_true",
                   help="disable per-batch progress bars (for logged runs)")

    rp = p.add_argument_group("report")
    rp.add_argument("--no-report", action="store_true",
                    help="skip the end-of-training PDF report")
    rp.add_argument("--report-test", action="store_true",
                    help="report's inference pass uses the TEST split instead "
                         "of validation (opt-in: keeps routine runs from "
                         "quietly turning test into a second validation set)")
    rp.add_argument("--report-thumbs", type=int, default=16,
                    help="thumbnails per problem-sample grid in the report")
    t.add_argument("--resume", default=None,
                   help="checkpoint to resume from: a .pt file, or a run "
                        "directory (uses its newest last_* checkpoint)")

    o = p.add_argument_group("objective")
    o.add_argument("--target-recall", type=float, default=0.95,
                   help="target aggregated recall over genuine classes")
    o.add_argument("--recall-agg", choices=RECALL_AGGREGATES, default="harmonic",
                   help="how per-class recalls combine for the target: macro "
                        "(arithmetic mean; one collapsed class can hide behind "
                        "strong ones), harmonic (dominated by the worst classes; "
                        "default), or min (strictest, worst single class)")
    o.add_argument("--min-threshold", type=float, default=0.0,
                   help="floor no operating threshold may go below; if the "
                        "recall target is only reachable beneath it, the epoch "
                        "operates AT the floor with target_met=0")
    o.add_argument("--threshold-mode", choices=("global", "per-class"),
                   default="global",
                   help="'global': one threshold for all classes; 'per-class': "
                        "each genuine class gets its own threshold (applied by "
                        "predicted class), so easy classes keep high specificity "
                        "while hard ones get the slack they need")
    o.add_argument("--per-class-min-count", type=int, default=20,
                   help="per-class mode: classes with fewer predicted validation "
                        "samples than this fall back to the global threshold")
    o.add_argument("--class-alpha", action="append", metavar="NAME=VALUE",
                   default=None,
                   help="manual focal alpha for a genuine class, e.g. "
                        "--class-alpha band3=2.0 (repeatable; works in any "
                        "mode; rescue boosts apply on top as max(manual, boost))")
    o.add_argument("--imbalance-ratio", type=float, default=math.inf,
                   help="max hard negatives per epoch = ratio * genuine count (1..inf)")
    o.add_argument("--focal-gamma", type=float, default=2.0)
    o.add_argument("--hn-alpha", type=float, default=0.25,
                   help="focal alpha for the hard_negative class (genuine classes = 1)")

    r = p.add_argument_group(
        "hard-negative pressure ramp",
        "optional schedule: start recall-focused, then increase hard-negative "
        "pressure. Progress advances one step per epoch whose validation meets "
        "the recall target, and holds otherwise.",
    )
    r.add_argument("--ramp-epochs", type=int, default=0,
                   help="number of ramp steps (0 = no ramp; constant values)")
    r.add_argument("--hn-alpha-end", type=float, default=None,
                   help="hn alpha at full ramp (default: same as --hn-alpha)")
    r.add_argument("--imbalance-ratio-start", type=float, default=None,
                   help="starting imbalance ratio (default: same as --imbalance-ratio; "
                        "set lower, e.g. 1.0, to begin with few hard negatives)")
    r.add_argument("--recall-first", type=int, nargs="?", const=3,
                   default=None, choices=sorted(RECALL_FIRST_PRESETS),
                   help="preset how hard training initially chases the recall "
                        "target before hard-negative pressure ramps in: 1 = "
                        "brief soft-start, 5 = long recall-only opening "
                        "(starts with almost no hard-negative influence and "
                        "earns pressure slowly). Bare --recall-first means "
                        "level 3. Presets --hn-alpha (the start), "
                        "--hn-alpha-end, --imbalance-ratio-start, and (in "
                        "fixed-ramp mode) --ramp-epochs; any flag you pass "
                        "explicitly overrides its preset. Combines with "
                        "--smart, which paces the pressure itself, so only "
                        "the endpoints apply there")

    s = p.add_argument_group(
        "smart mode",
        "adaptive alternative to the fixed ramp: cyclic (half-cosine) learning "
        "rate, with hard-negative pressure raised/held/rewound at cycle "
        "boundaries based on validation metrics. Pressure endpoints reuse the "
        "ramp flags (--hn-alpha -> --hn-alpha-end, --imbalance-ratio-start -> "
        "--imbalance-ratio). Mutually exclusive with --ramp-epochs.",
    )
    s.add_argument("--smart", type=int, nargs="?", const=3, default=None,
                   choices=sorted(SMART_PRESETS),
                   help="enable the smart controller at an effort level: 1 = "
                        "minimal/fast (short cycles, aggressive pressure "
                        "steps, gives up early), 5 = marathon (long settled "
                        "cycles, tiny pressure steps, deep LR anneals, many "
                        "retries - slowly reaches the goal over a long "
                        "horizon). Bare --smart means level 3. Levels preset "
                        "epochs, cycle length, lr-min, pressure step, "
                        "rewinds, rescue, patience, and top-k; any flag you "
                        "pass explicitly overrides its preset")
    s.add_argument("--lr-cycle-epochs", type=int, default=None,
                   help="epochs per LR cycle; decisions happen at the trough "
                        "(default: per --smart level)")
    s.add_argument("--lr-min", type=float, default=None,
                   help="trough learning rate (default: --lr divided per "
                        "--smart level: 10/20/25/50/100)")
    s.add_argument("--pressure-step", type=float, default=None,
                   help="initial pressure increment per successful cycle "
                        "(default: per --smart level)")
    s.add_argument("--max-rewinds", type=int, default=None,
                   help="rewinds at one pressure level before accepting it "
                        "as the run's ceiling (default: per --smart level)")
    s.add_argument("--keep-top-k", type=int, default=None,
                   help="snapshots/ archive size (default: per --smart level)")
    s.add_argument("--rescue", action="store_true",
                   help="class rescue: at each cycle trough, boost the loss "
                        "weight and sampling of genuine classes lagging the "
                        "recall target (smart mode only); pressure raises are "
                        "blocked while any class is under rescue (enabled "
                        "automatically at --smart levels 3+)")
    s.add_argument("--rescue-alpha-max", type=float, default=3.0,
                   help="cap on a rescued class's focal alpha (scales with "
                        "its recall deficit)")
    s.add_argument("--rescue-oversample-max", type=int, default=3,
                   help="cap on a rescued class's per-epoch repeat factor")
    s.add_argument("--rescue-ema", type=float, default=None,
                   help="EMA smoothing of per-class recall across troughs; "
                        "0 = react to the latest trough only (default: per "
                        "--smart level)")

    mi = p.add_argument_group("hard-negative mining")
    mi.add_argument("--no-mining", action="store_true",
                    help="uniform hard-negative subsampling instead of error-driven")
    mi.add_argument("--mining-random-frac", type=float, default=0.2,
                    help="share of the hard-negative budget drawn uniformly at random")

    ou = p.add_argument_group(
        "optuna search",
        "optional hyper-parameter search (pip install optuna): every trial "
        "is a full training run into <out-dir>/trial_NNN. Per trial the "
        "precedence is explicit CLI flags (pinned; colliding dimensions "
        "leave the space) > sampled space > config file > presets > "
        "defaults. Maximizes HN specificity at the recall target (unmet "
        "trials rank beneath every met one).",
    )
    ou.add_argument("--optuna", type=int, default=None, metavar="N_TRIALS",
                    help="run an Optuna study of this many trials instead "
                         "of a single training run")
    ou.add_argument("--optuna-space", default=None, metavar="SPACE.json",
                    help="search space (default: lr, weight-decay, "
                         "optimizer, focal-gamma, hn-alpha-end, "
                         "recall-first); see example_optuna_space.json")
    ou.add_argument("--optuna-storage", default=None, metavar="URL",
                    help="Optuna storage URL (default: sqlite db in the "
                         "out dir, so an interrupted study resumes)")
    ou.add_argument("--optuna-prune-warmup", type=int, default=3,
                    help="epochs a trial runs before the median pruner may "
                         "stop it (raise for --smart, whose rewinds make "
                         "early epochs non-monotonic)")
    ou.add_argument("--optuna-no-prune", action="store_true",
                    help="run every trial to completion")

    d = p.add_argument_group("data")
    d.add_argument("--augment-plugin", action="append", default=None,
                   metavar="FILE.py",
                   help="Python file defining extra augmentations (an "
                        "AUGMENTATIONS dict of {name: factory}, optional "
                        "POST_RESIZE set) merged into the built-in catalog "
                        "before --augment specs are resolved; repeatable. "
                        "See example_augment_plugin.py")
    d.add_argument("--imagenet-norm", action="store_true",
                   help="ImageNet mean/std normalization (default: just /255)")
    d.add_argument("--augment", nargs="+", default=None,
                   metavar="NAME[:k=v,...]",
                   help="training-split augmentations, applied in the order "
                        "given (validation/eval/inference are never "
                        "augmented). Each spec is a name with optional "
                        "parameters, e.g. rotation:p=0.7,degrees=30 or "
                        "erasing:p=0.5,scale=0.02-0.2 (ranges as lo-hi); in "
                        "a --config file, JSON objects like "
                        "{\"name\": \"erasing\", \"p\": 0.5} also work. "
                        "Catalog: " + ", ".join(sorted(AUGMENTATIONS))
                        + ". CAUTION: photometric ops (colorjitter, invert, "
                        "solarize, equalize, autocontrast, posterize, "
                        "grayscale) alter intensity and can destroy the label "
                        "when classes are intensity-coded; rotation, "
                        "perspective, gaussianblur, sharpness, and erasing "
                        "are the safer subset there")

    return p


def _load_json_object(flag: str, path) -> dict:
    """Read a JSON side-file (--config / --optuna-space) with clean errors
    instead of raw tracebacks for the two most common user mistakes."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except OSError as e:
        raise SystemExit(f"{flag}: cannot read {path}: {e}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"{flag}: {path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"{flag}: top level must be a JSON object")
    return data


def _json_key_dest(key: str) -> str | None:
    """One key dialect for BOTH JSON side-files: leading dashes are
    optional, inner dashes equal underscores, and a leading underscore
    marks a comment key (returns None - the entry is skipped)."""
    if key.startswith("_"):
        return None
    return key.lstrip("-").replace("-", "_")


def _explicit_dests(argv) -> set:
    """Which dests were explicitly given on the CLI: re-parse with every
    default suppressed, so only provided options appear in the namespace."""
    aux = build_parser()
    for action in aux._actions:
        if action.dest != "help":
            action.default = argparse.SUPPRESS
    return set(vars(aux.parse_args(argv)))


def _apply_config(p: argparse.ArgumentParser, args: argparse.Namespace,
                  explicit: set) -> set:
    """Overlay config-file values onto `args` for every option the user did
    NOT pass explicitly on the command line. Returns the dests it set."""
    cfg = _load_json_object("--config", args.config)

    applied: set = set()
    actions = {a.dest: a for a in p._actions}
    for key, value in cfg.items():
        dest = _json_key_dest(key)
        if dest is None:
            continue  # "_"-prefixed keys are comments, as in --optuna-space
        if dest == "config":
            continue  # a config file cannot chain-load another
        action = actions.get(dest)
        if action is None or dest == "help":
            p.error(f"--config: unknown option '{key}'")
        if dest in explicit:
            continue  # explicit CLI wins
        if value is None:
            continue  # null == not specified; keep the parser default
        if action.type is not None:
            if isinstance(value, str):
                value = action.type(value)
            elif isinstance(value, list):
                value = [action.type(v) if isinstance(v, str) else v
                         for v in value]
        if action.choices is not None:
            for v in value if isinstance(value, list) else [value]:
                if v not in action.choices:
                    p.error(f"--config: '{key}': {v!r} not in "
                            f"{sorted(action.choices)}")
        setattr(args, dest, value)
        applied.add(dest)
    return applied


def parse_args(argv=None, overrides=None) -> argparse.Namespace:
    """overrides: {dest: value} applied over the parsed argv and treated as
    explicit - an Optuna trial's sampled values. Unlike appended argv
    tokens, overrides can also express False for a store_true flag, so
    sampled values beat the config file and presets for every type."""
    p = build_parser()
    args = p.parse_args(argv)
    explicit = _explicit_dests(sys.argv[1:] if argv is None else argv)
    for dest, value in (overrides or {}).items():
        setattr(args, dest, value)
        explicit.add(dest)
    if args.config:
        explicit |= _apply_config(p, args, explicit)
    if args.h5 is None:
        p.error("the dataset .h5 path is required (positional argument or "
                "\"h5\" in --config)")
    if args.smart and args.ramp_epochs > 0:
        p.error("--smart and --ramp-epochs are mutually exclusive (smart mode "
                "replaces the fixed ramp)")
    if args.rescue and not args.smart:
        p.error("--rescue requires --smart (rescue decisions run at cycle "
                "troughs)")
    if args.optuna is not None and args.optuna < 1:
        p.error("--optuna expects a positive trial count")
    if args.optuna and args.resume:
        p.error("--optuna and --resume are mutually exclusive (trials start "
                "fresh; an interrupted study resumes via its storage)")

    # --recall-first preset: pressure endpoints (and, in fixed-ramp mode,
    # the ramp length) for options not given on the CLI or in the config.
    if args.recall_first:
        for attr, value in RECALL_FIRST_PRESETS[args.recall_first].items():
            if value is None or attr in explicit:
                continue
            if attr == "ramp_epochs" and args.smart:
                continue  # smart mode paces pressure at cycle troughs
            setattr(args, attr, value)

    # Fill unset (None) values: from the --smart level preset, or from the
    # base defaults in non-smart mode. Explicit flags always win.
    if args.smart:
        preset = SMART_PRESETS[args.smart]
        for attr in ("epochs", "lr_cycle_epochs", "pressure_step",
                     "max_rewinds", "keep_top_k", "rescue_ema", "patience"):
            if getattr(args, attr) is None:
                setattr(args, attr, preset[attr])
        if args.lr_min is None:
            args.lr_min = args.lr / preset["lr_min_div"]
        if preset["rescue"] and "rescue" not in explicit:
            args.rescue = True
    else:
        for attr, value in dict(epochs=50, patience=10, lr_cycle_epochs=10,
                                pressure_step=0.25, max_rewinds=3,
                                keep_top_k=3, rescue_ema=0.5).items():
            if getattr(args, attr) is None:
                setattr(args, attr, value)
        if args.lr_min is None:
            args.lr_min = args.lr / 25
    return args


def parse_class_alphas(pairs, classes, hn_index) -> dict[int, float]:
    """--class-alpha NAME=VALUE entries -> {class_index: alpha}."""
    out: dict[int, float] = {}
    for pair in pairs or []:
        name, _, value = pair.partition("=")
        if not value:
            raise SystemExit(f"--class-alpha expects NAME=VALUE, got '{pair}'")
        if name not in classes:
            raise SystemExit(f"--class-alpha: unknown class '{name}' "
                             f"(classes: {', '.join(classes)})")
        idx = classes.index(name)
        if idx == hn_index:
            raise SystemExit("--class-alpha cannot target hard_negative; "
                             "use --hn-alpha")
        out[idx] = float(value)
    return out


def build_optimizer(args, model) -> torch.optim.Optimizer:
    """The training optimizer; smart-mode rewinds rebuild it fresh too."""
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr,
                               momentum=args.momentum,
                               nesterov=args.momentum > 0,
                               weight_decay=args.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=args.lr,
                             weight_decay=args.weight_decay)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp,
                    miner=None, desc: str = "train", progress: bool = True) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    bar = tqdm(loader, desc=desc, unit="batch", leave=False, disable=not progress)
    for imgs, labs, idxs in bar:
        imgs = imgs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(imgs)
            per_sample = criterion(logits, labs)
        loss = per_sample.mean()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if miner is not None:
            miner.update(idxs, per_sample)

        total_loss += float(loss.detach()) * len(labs)
        total_n += len(labs)
        bar.set_postfix(loss=f"{total_loss / total_n:.4f}")
    return total_loss / max(total_n, 1)


def pressure_values(args, f: float, n_genuine: int, n_hn: int) -> tuple[float, float]:
    """(hn_alpha, imbalance_ratio) at pressure fraction f in [0, 1].

    Linear interpolation between the ramp endpoint flags. An infinite ratio
    endpoint is interpolated in budget space (inf == the ratio that admits
    every hard negative), so the hard-negative count still grows smoothly.
    """
    alpha_end = args.hn_alpha_end if args.hn_alpha_end is not None else args.hn_alpha
    ratio_start = (args.imbalance_ratio_start
                   if args.imbalance_ratio_start is not None else args.imbalance_ratio)
    ratio_end = args.imbalance_ratio

    full_ratio = max(1.0, n_hn / max(n_genuine, 1))  # ratio admitting all HN
    rs = full_ratio if math.isinf(ratio_start) else ratio_start
    re = full_ratio if math.isinf(ratio_end) else ratio_end

    hn_alpha = args.hn_alpha + f * (alpha_end - args.hn_alpha)
    ratio = max(1.0, rs + f * (re - rs))
    if f >= 1.0 and math.isinf(ratio_end):
        ratio = math.inf
    return hn_alpha, ratio


def ramp_values(args, ramp_progress: int, n_genuine: int, n_hn: int) -> tuple[float, float]:
    """(hn_alpha, imbalance_ratio) at the given ramp progress (fixed-ramp
    mode; ramp_epochs == 0 means the end values apply from epoch 0)."""
    if args.ramp_epochs <= 0:
        return pressure_values(args, 1.0, n_genuine, n_hn)
    return pressure_values(args, min(1.0, ramp_progress / args.ramp_epochs),
                           n_genuine, n_hn)


def validate(model, loader, device, hn_index, target_recall, recall_agg,
             min_threshold: float = 0.0, threshold_mode: str = "global",
             per_class_min_count: int = 20,
             desc: str = "validate", progress: bool = True) -> dict:
    probs, labels = collect_probs(model, loader, device, desc=desc,
                                  progress=progress)
    if threshold_mode == "per-class":
        op = sweep_class_thresholds(probs, labels, hn_index, target_recall,
                                    agg=recall_agg, min_threshold=min_threshold,
                                    min_count=per_class_min_count)
    else:
        op = sweep_threshold(probs, labels, hn_index, target_recall,
                             agg=recall_agg, min_threshold=min_threshold)
    try:
        _, _, auroc = genuine_vs_hn_roc(probs, labels, hn_index)
    except ValueError:
        auroc = float("nan")
    op["auroc"] = auroc
    return op


def selection_key(op: dict) -> tuple:
    """Checkpoint ranking: meeting the target dominates; then specificity at
    the target; then aggregated recall (the tie-breaker while the target is
    out of reach)."""
    spec = op["specificity"]
    if math.isnan(spec):
        spec = -math.inf
    return (int(op["target_met"]), spec if op["target_met"] else -math.inf,
            op["recall"])


def save_checkpoint(path: Path, *, model, optimizer, scaler, epoch, args, classes,
                    hn_index, op, best_key, miner=None, ramp_progress=0,
                    controller=None) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "config": vars(args),
        "classes": classes,
        "hard_negative_index": hn_index,
        "arch": args.arch,
        "imagenet_norm": args.imagenet_norm,
        "threshold": op["threshold"],
        "threshold_mode": args.threshold_mode,
        "class_thresholds": op.get("class_thresholds"),
        "min_threshold": args.min_threshold,
        "recall_agg": args.recall_agg,
        "val_metrics": op,
        "best_key": best_key,
        "miner_state": miner.state_dict() if miner is not None else None,
        "ramp_progress": ramp_progress,
        "controller_state": controller.state_dict() if controller is not None else None,
    }, path)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.optuna:
        run_optuna(args, sys.argv[1:] if argv is None else list(argv))
        return
    train(args)


def train(args, on_epoch_end=None) -> dict:
    """One training run. on_epoch_end(epoch, op) -> bool is called after
    each epoch's checkpoints are written; returning True stops the run
    early (Optuna pruning). Returns the best epoch/metrics and paths."""
    seed_everything(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = args.amp and device.type == "cuda"

    out_dir = Path(args.out_dir or f"runs/{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolved options (CLI + config file + presets applied) — itself a valid
    # --config file, so any run can be reproduced from its output directory.
    (out_dir / "config.json").write_text(
        json.dumps({k: v for k, v in vars(args).items() if k != "config"},
                   indent=2, default=str),
        encoding="utf-8")

    if args.augment_plugin:
        plugged = load_augmentation_plugins(args.augment_plugin)
        if plugged:
            print("augmentation plugins:", ", ".join(plugged))

    summary = validate_h5(args.h5)
    classes = summary["classes"]
    hn_index = summary["hard_negative_index"]
    print(f"dataset: {args.h5}")
    for split_name, c in summary["counts"].items():
        print(f"  {split_name}: {c['genuine']} genuine, {c['hard_negative']} hard negatives")

    train_ds = H5SnippetDataset(args.h5, SPLIT_TRAIN,
                                imagenet_norm=args.imagenet_norm,
                                augment=args.augment)
    val_ds = H5SnippetDataset(args.h5, SPLIT_VAL, imagenet_norm=args.imagenet_norm)
    if args.augment:
        print("augmentations (train split only):",
              ", ".join(str(a) for a in args.augment))

    miner = None if args.no_mining else HardNegativeMiner(train_ds.labels, hn_index)
    n_genuine = int((train_ds.labels != hn_index).sum())
    n_hn = int((train_ds.labels == hn_index).sum())

    hn_alpha0, ratio0 = ramp_values(args, 0, n_genuine, n_hn)
    sampler = ImbalanceCapSampler(
        train_ds.labels, hn_index, ratio=ratio0, miner=miner,
        random_frac=args.mining_random_frac, seed=args.seed,
    )
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kw)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained,
                        weights_path=args.weights_path).to(device)
    criterion = FocalLoss(len(classes), hn_index, gamma=args.focal_gamma,
                          hn_alpha=hn_alpha0).to(device)
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    base_alphas = parse_class_alphas(args.class_alpha, classes, hn_index)
    if base_alphas:
        criterion.set_class_alphas(
            {c: base_alphas.get(c, 1.0) for c in range(len(classes))
             if c != hn_index})

    controller = None
    if args.smart:
        print(f"smart level {args.smart}: epochs {args.epochs}, "
              f"cycle {args.lr_cycle_epochs}, lr {args.lr:g} -> {args.lr_min:.2e}, "
              f"pressure step {args.pressure_step}, max rewinds {args.max_rewinds}, "
              f"rescue {'on' if args.rescue else 'off'}"
              f"{f' (ema {args.rescue_ema})' if args.rescue else ''}, "
              f"patience {args.patience} cycles, top-k {args.keep_top_k}")
        controller = SmartController(
            lr_max=args.lr, lr_min=args.lr_min, cycle_epochs=args.lr_cycle_epochs,
            pressure_step=args.pressure_step, max_rewinds=args.max_rewinds,
            keep_top_k=args.keep_top_k, rescue=args.rescue,
            rescue_alpha_max=args.rescue_alpha_max,
            rescue_oversample_max=args.rescue_oversample_max,
            rescue_ema=args.rescue_ema, target_recall=args.target_recall,
        )

    start_epoch = 0
    best_key = None
    best_op = None
    best_epoch = None
    stopped_early = False
    ramp_progress = 0
    if args.resume:
        resume_path = find_checkpoint(args.resume, "last")
        if resume_path is None:
            raise SystemExit(f"--resume: no checkpoint found at {args.resume}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        ck_opt = (ckpt.get("config") or {}).get("optimizer", "adamw")
        if ck_opt != args.optimizer:
            raise SystemExit(
                f"--resume: checkpoint was trained with '{ck_opt}' but "
                f"--optimizer {args.optimizer} was requested; optimizer "
                "state cannot carry over - resume with the same optimizer")
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        # best_key describes this directory's best checkpoint; when resuming
        # into a fresh directory that file doesn't exist, so the new run must
        # track its own best from scratch or it may never write one.
        best_key = (ckpt.get("best_key")
                    if find_checkpoint(out_dir, "best") is not None else None)
        ramp_progress = ckpt.get("ramp_progress", 0)
        if miner is not None and ckpt.get("miner_state") is not None:
            miner.load_state_dict(ckpt["miner_state"])
        if controller is not None and ckpt.get("controller_state") is not None:
            controller.load_state_dict(ckpt["controller_state"], out_dir=out_dir)
        print(f"resumed from {resume_path} at epoch {start_epoch}")

    csv_path = out_dir / "metrics.csv"
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    # Long-format per-class log: one row per (epoch, genuine class) in both
    # threshold modes — the per-class recall history lives here.
    class_csv_path = out_dir / "class_thresholds.csv"
    new_class_csv = not class_csv_path.exists()
    class_csv_file = open(class_csv_path, "a", newline="")
    class_writer = csv.DictWriter(class_csv_file, fieldnames=CLASS_CSV_FIELDS)
    if new_class_csv:
        class_writer.writeheader()

    epochs_since_best = 0
    improved_this_cycle = False
    stop = False
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if controller is not None:
            hn_alpha, ratio = pressure_values(args, controller.p_try,
                                              n_genuine, n_hn)
            lr = controller.lr_at()
            for g in optimizer.param_groups:
                g["lr"] = lr
            p_used, cycle_used = controller.p_try, controller.cycle
            # Rescue boosts (from the last trough) apply on top of any
            # manual --class-alpha values; recovered classes fall back.
            class_alphas = {c: base_alphas.get(c, 1.0)
                            for c in range(len(classes)) if c != hn_index}
            for c, a in controller.rescue_alphas.items():
                class_alphas[c] = max(class_alphas[c], a)
            criterion.set_class_alphas(class_alphas)
            repeats_used = dict(controller.rescue_repeats)
            sampler.set_genuine_repeats(repeats_used)
        else:
            hn_alpha, ratio = ramp_values(args, ramp_progress, n_genuine, n_hn)
            lr = args.lr
            p_used, cycle_used = "", ""
            class_alphas = {c: base_alphas.get(c, 1.0)
                            for c in range(len(classes)) if c != hn_index}
            repeats_used = {}
        criterion.set_hn_alpha(hn_alpha)
        sampler.set_ratio(ratio)
        sampler.set_epoch(epoch)

        progress = not args.no_progress
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, amp, miner=miner,
                                     desc=f"epoch {epoch} train", progress=progress)
        op = validate(model, val_loader, device, hn_index, args.target_recall,
                      args.recall_agg, min_threshold=args.min_threshold,
                      threshold_mode=args.threshold_mode,
                      per_class_min_count=args.per_class_min_count,
                      desc=f"epoch {epoch} validate", progress=progress)
        dt = time.time() - t0

        if controller is None and op["target_met"] and args.ramp_epochs > 0:
            ramp_progress = min(ramp_progress + 1, args.ramp_epochs)

        key = selection_key(op)
        improved = best_key is None or key > tuple(best_key)

        # best.pt and cycle_best.pt hold the weights that were actually
        # evaluated this epoch — saved before any boundary rewind below.
        ckpt_kw = dict(model=model, optimizer=optimizer, scaler=scaler,
                       epoch=epoch, args=args, classes=classes,
                       hn_index=hn_index, op=op, miner=miner,
                       ramp_progress=ramp_progress, controller=controller)
        if controller is not None:
            improved_this_cycle = improved_this_cycle or improved
            if controller.observe(key):
                save_checkpoint(out_dir / "cycle_best.pt", best_key=best_key,
                                **ckpt_kw)
        if improved:
            best_key = key
            best_op, best_epoch = op, epoch
            epochs_since_best = 0
            best_path = out_dir / checkpoint_name("best", epoch, op)
            save_checkpoint(best_path, best_key=best_key, **ckpt_kw)
            prune_role(out_dir, "best", best_path)
        else:
            epochs_since_best += 1

        # --- cycle boundary: decide, possibly rewind -------------------
        event = ""
        boundary_msg = None
        op_for_last = op
        if controller is not None:
            controller.epoch_in_cycle += 1
            if controller.at_boundary():
                controller.cycles_since_best = (
                    0 if improved_this_cycle else controller.cycles_since_best + 1)
                improved_this_cycle = False
                boundary_cycle = controller.cycle
                base_event = controller.end_cycle(
                    out_dir, per_class_recall=op["per_class_recall"])
                rescued = sorted(set(controller.rescue_alphas)
                                 | set(controller.rescue_repeats))
                event = base_event
                if rescued:
                    event += " rescue:" + ",".join(classes[c] for c in rescued)
                if base_event in ("rewind", "ceiling") and (out_dir / "milestone.pt").exists():
                    mckpt = torch.load(out_dir / "milestone.pt",
                                       map_location=device, weights_only=False)
                    model.load_state_dict(mckpt["model_state"])
                    optimizer = build_optimizer(args, model)
                    scaler = torch.amp.GradScaler(device.type, enabled=amp)
                    op_for_last = mckpt["val_metrics"]
                boundary_msg = (f"  cycle {boundary_cycle}: {event}  ->  "
                                f"pressure {controller.p_try:.2f} "
                                f"(stable {controller.p_stable:.2f}, "
                                f"step {controller.step:.3f}, "
                                f"rewinds {controller.rewinds})")
                if args.patience and controller.cycles_since_best >= args.patience:
                    stop = True

        class_thr = op.get("class_thresholds")
        thr_values = list(class_thr.values()) if class_thr else [op["threshold"]]
        writer.writerow({
            "epoch": epoch, "train_loss": f"{train_loss:.6f}",
            "threshold": f"{op['threshold']:.6f}",
            "threshold_mode": args.threshold_mode,
            "thr_min": f"{min(thr_values):.6f}", "thr_max": f"{max(thr_values):.6f}",
            "target_met": int(op["target_met"]),
            "recall": f"{op['recall']:.6f}", "recall_agg": op["recall_agg"],
            "specificity": f"{op['specificity']:.6f}",
            "max_recall": f"{op['max_recall']:.6f}",
            "auroc": f"{op['auroc']:.6f}", "hn_alpha": f"{hn_alpha:.4f}",
            "imbalance_ratio": ratio, "ramp_progress": ramp_progress,
            "pressure": p_used if p_used == "" else f"{p_used:.3f}",
            "cycle": cycle_used, "event": event,
            "lr": f"{lr:.3e}", "epoch_time_s": f"{dt:.1f}",
        })
        csv_file.flush()

        fallback_set = set(op.get("fallback_classes", []))
        for c, r in sorted(op["per_class_recall"].items()):
            class_writer.writerow({
                "epoch": epoch, "class": classes[c],
                "threshold": f"{(class_thr or {}).get(c, op['threshold']):.6f}",
                "recall": f"{r:.6f}",
                "predicted_n": op["predicted_counts"].get(c, 0),
                "accepted_n": op.get("accepted_counts", {}).get(c, ""),
                "fallback": int(c in fallback_set) if class_thr else "",
                "alpha": f"{class_alphas.get(c, 1.0):.3f}",
                "repeat": repeats_used.get(c, 1),
            })
        class_csv_file.flush()

        marker = " *" if improved else ""
        if class_thr:
            thr_txt = f"thr {min(thr_values):.3f}..{max(thr_values):.3f}"
        else:
            thr_txt = f"thr {op['threshold']:.4f}"
        smart_txt = (f"  [c{cycle_used} p {p_used:.2f} lr {lr:.1e}]"
                     if controller is not None else "")
        print(
            f"epoch {epoch:3d}  loss {train_loss:.4f}  "
            f"{op['recall_agg']}-recall {op['recall']:.4f}"
            f"{'' if op['target_met'] else ' (below target)'}  "
            f"spec {op['specificity']:.4f}  {thr_txt}  "
            f"auroc {op['auroc']:.4f}  {dt:.1f}s{smart_txt}{marker}"
        )
        if boundary_msg:
            print(boundary_msg)
        if stop:
            print(f"early stop: no improvement for {args.patience} cycles")

        # The last checkpoint carries the post-decision state (post-rewind
        # weights and controller state), so --resume continues exactly where
        # the controller left off; its metrics/threshold match its weights.
        last_path = out_dir / checkpoint_name("last", epoch, op_for_last)
        save_checkpoint(last_path, model=model, optimizer=optimizer,
                        scaler=scaler, epoch=epoch, args=args, classes=classes,
                        hn_index=hn_index, op=op_for_last, miner=miner,
                        ramp_progress=ramp_progress, controller=controller,
                        best_key=best_key)
        prune_role(out_dir, "last", last_path)

        if on_epoch_end is not None and on_epoch_end(epoch, op):
            print(f"stopped after epoch {epoch} (pruned)")
            stopped_early = True
            break
        if stop:
            break
        if controller is None and args.patience and epochs_since_best >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    csv_file.close()
    class_csv_file.close()

    if not args.no_report and find_checkpoint(out_dir, "best") is not None:
        try:
            from dataset import SPLIT_TEST
            from report import build_report
            path = build_report(
                out_dir, args.h5,
                split=SPLIT_TEST if args.report_test else SPLIT_VAL,
                thumbs=args.report_thumbs, device=str(device),
                progress=not args.no_progress)
            print(f"report: {path}")
        except Exception as e:  # a report failure must never eat a finished run
            print(f"report generation failed (training outputs unaffected): {e}")

    print(f"done. checkpoints and metrics.csv in {out_dir}")
    return {"out_dir": out_dir, "best_epoch": best_epoch, "best_metrics": best_op,
            "best_path": find_checkpoint(out_dir, "best"),
            "stopped_early": stopped_early}


# ---- optional Optuna search -------------------------------------------------

def optuna_objective_value(op: dict) -> float:
    """The scalar a study maximizes: HN specificity once the recall target is
    met; below it, recall - 1 (always negative), so unmet trials rank beneath
    every met one, ordered by how close they came."""
    if op["target_met"]:
        return float(op["specificity"])
    return float(op["recall"]) - 1.0


def _optuna_suggest(trial, name: str, spec: dict):
    kind = spec.get("type", "float")
    if kind == "categorical":
        return trial.suggest_categorical(name, list(spec["choices"]))
    if kind == "int":
        return trial.suggest_int(name, int(spec["low"]), int(spec["high"]),
                                 step=int(spec.get("step", 1)),
                                 log=bool(spec.get("log", False)))
    if kind == "float":
        return trial.suggest_float(name, float(spec["low"]), float(spec["high"]),
                                   step=spec.get("step"),
                                   log=bool(spec.get("log", False)))
    raise SystemExit(f"--optuna-space: '{name}': unknown type '{kind}' "
                     "(float, int, or categorical)")


def _check_spec(name: str, spec, action) -> None:
    """Reject a malformed dimension spec BEFORE the study or its storage
    exist, so a bad space never crashes inside trial 0."""
    if not isinstance(spec, dict):
        raise SystemExit(f"--optuna-space: '{name}': spec must be an object "
                         "(type/low/high or choices)")
    kind = spec.get("type", "float")
    if kind == "categorical":
        choices = spec.get("choices")
        if not isinstance(choices, list) or not choices:
            raise SystemExit(f"--optuna-space: '{name}': categorical needs "
                             "a non-empty 'choices' list")
        if action.choices is not None:
            bad = [c for c in choices
                   if c is not None and c not in action.choices]
            if bad:
                raise SystemExit(f"--optuna-space: '{name}': {bad} not in "
                                 f"{sorted(action.choices)}")
    elif kind in ("int", "float"):
        low, high = spec.get("low"), spec.get("high")
        for v in (low, high):
            if not isinstance(v, (int, float)) or isinstance(v, bool):
                raise SystemExit(f"--optuna-space: '{name}': {kind} needs "
                                 "numeric 'low' and 'high'")
        if high < low:
            raise SystemExit(f"--optuna-space: '{name}': low {low} exceeds "
                             f"high {high}")
    else:
        raise SystemExit(f"--optuna-space: '{name}': unknown type '{kind}' "
                         "(float, int, or categorical)")


def _optuna_space(args, parser: argparse.ArgumentParser) -> dict:
    """The search space, fully validated up front: shared key dialect
    (dashes, comments), duplicate-spelling detection, option-name checks
    with the USER'S spelling in messages, reserved options, spec shapes."""
    actions = {a.dest: a for a in parser._actions}
    items = []  # (dest, raw_key, spec)
    if args.optuna_space:
        seen: dict[str, str] = {}
        for raw, spec in _load_json_object("--optuna-space",
                                           args.optuna_space).items():
            dest = _json_key_dest(raw)
            if dest is None:
                if isinstance(spec, dict) and ({"type", "low", "choices"}
                                               & set(spec)):
                    print(f"note: --optuna-space key '{raw}' looks like a "
                          "search dimension but leading-underscore keys "
                          "are comments - ignored")
                continue
            if dest in seen:
                raise SystemExit(f"--optuna-space: '{raw}' and '{seen[dest]}'"
                                 f" both mean '{dest}' - keep one spelling")
            seen[dest] = raw
            items.append((dest, raw, spec))
        if not items:
            raise SystemExit("--optuna-space: no search dimensions found")
    else:
        items = [(k, k, v) for k, v in OPTUNA_DEFAULT_SPACE.items()]
    for dest, raw, spec in items:
        action = actions.get(dest)
        if action is None or not action.option_strings:
            raise SystemExit(f"--optuna-space: '{raw}' is not a train.py "
                             "option")
        if dest in OPTUNA_RESERVED or dest in OPTUNA_KEYS:
            raise SystemExit(f"--optuna-space: '{raw}' cannot be searched "
                             "(the study manages it per trial)")
        _check_spec(raw, spec, action)
    return {dest: spec for dest, raw, spec in items}


def run_optuna(args, argv: list[str]) -> None:
    """Study of args.optuna trials. Sampled values are applied as parse-time
    overrides (never argv tokens), so explicit CLI flags pin their option
    out of the space while samples beat the config file and presets for
    every value type; each trial records its EFFECTIVE post-parse values,
    which is what trials.csv and best_trial.json report."""
    try:
        import optuna
    except ImportError:
        raise SystemExit("--optuna needs the optuna package: pip install optuna")
    parser = build_parser()
    space = _optuna_space(args, parser)

    def flag_of(dest):
        a = next(a for a in parser._actions if a.dest == dest)
        return next((o for o in a.option_strings if o.startswith("--")),
                    a.option_strings[0])

    explicit = _explicit_dests(argv)
    pinned = sorted(k for k in space if k in explicit)
    if pinned:
        print("note: pinned by explicit CLI flags and removed from the "
              "search space: " + ", ".join(flag_of(k) for k in pinned))
    space = {k: v for k, v in space.items() if k not in explicit}
    if not space:
        raise SystemExit("every search dimension is pinned on the command "
                         "line; nothing left to search")

    base = Path(args.out_dir or f"runs/optuna_{datetime.now():%Y%m%d_%H%M%S}")
    base.mkdir(parents=True, exist_ok=True)
    storage = args.optuna_storage or f"sqlite:///{(base / 'optuna.db').as_posix()}"
    pruner = (optuna.pruners.NopPruner() if args.optuna_no_prune else
              optuna.pruners.MedianPruner(n_startup_trials=5,
                                          n_warmup_steps=args.optuna_prune_warmup))
    study = optuna.create_study(
        study_name=base.name, storage=storage, load_if_exists=True,
        direction="maximize", pruner=pruner,
        sampler=optuna.samplers.TPESampler(seed=args.seed))
    print(f"optuna study '{study.study_name}' ({storage}): {args.optuna} trials"
          + (f", {len(study.trials)} already recorded" if study.trials else "")
          + f"; searching {', '.join(space)}")

    class TrialArgsError(Exception):
        """A sampled combination the parser rejects - the trial FAILs and
        the study continues instead of aborting with a stranded trial."""

    def objective(trial):
        params = {name: _optuna_suggest(trial, name, spec)
                  for name, spec in space.items()}
        trial_dir = base / f"trial_{trial.number:03d}"
        overrides = {k: v for k, v in params.items() if v is not None}
        overrides["out_dir"] = str(trial_dir)
        overrides["no_report"] = True
        try:
            t_args = parse_args(argv, overrides=overrides)
        except SystemExit as e:
            raise TrialArgsError(
                f"trial {trial.number} parameters rejected by the parser: "
                + ", ".join(f"{k}={v}" for k, v in params.items())) from e
        for k in OPTUNA_KEYS:  # the trial's config.json must not relaunch a study
            setattr(t_args, k, None if k != "optuna_no_prune" else False)
        # what the trial ACTUALLY runs with (config/presets resolved; a
        # sampled null falls back) - the honest record for every output
        effective = {k: getattr(t_args, k) for k in list(space) + pinned}
        trial.set_user_attr("effective", effective)
        trial.set_user_attr("out_dir", str(trial_dir))
        print(f"\n=== trial {trial.number}: "
              + ", ".join(f"{k}={effective[k]}" for k in space) + " ===")

        def on_epoch_end(epoch, op):
            trial.report(optuna_objective_value(op), epoch)
            return trial.should_prune()

        result = train(t_args, on_epoch_end=on_epoch_end)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if result["stopped_early"] or result["best_metrics"] is None:
            raise optuna.TrialPruned()
        op = result["best_metrics"]
        trial.set_user_attr("best_epoch", result["best_epoch"])
        trial.set_user_attr("recall", float(op["recall"]))
        trial.set_user_attr("specificity", float(op["specificity"]))
        trial.set_user_attr("target_met", bool(op["target_met"]))
        return optuna_objective_value(op)

    study.optimize(objective, n_trials=args.optuna, gc_after_trial=True,
                   catch=(TrialArgsError,))

    # columns: this run's dimensions and pins first, then anything older
    # recorded trials searched (a resumed study may have sampled an option
    # that is pinned now - its history must not vanish from the report)
    names = list(space) + sorted(
        {n for t_ in study.trials
         for n in set(t_.params) | set(t_.user_attrs.get("effective", {}))}
        - set(space))
    with open(base / "trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial", "state", "value", *names, "best_epoch", "recall",
                    "specificity", "target_met", "out_dir"])
        for t in study.trials:
            eff = t.user_attrs.get("effective", {})
            w.writerow([t.number, t.state.name,
                        "" if t.value is None else f"{t.value:.6f}",
                        *[eff.get(n, t.params.get(n, "")) for n in names],
                        *[t.user_attrs.get(k, "") for k in
                          ("best_epoch", "recall", "specificity", "target_met",
                           "out_dir")]])
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"\n{len(complete)} of {len(study.trials)} trials complete "
          f"({len(study.trials) - len(complete)} pruned/failed); "
          f"trials.csv in {base}")
    if not complete:
        print("no trial completed - nothing to pick a best from")
        return
    best = study.best_trial
    best_dir = Path(best.user_attrs["out_dir"])
    best_eff = best.user_attrs.get("effective", best.params)
    summary = {"trial": best.number, "value": best.value, "params": best.params,
               "effective": best_eff,
               "best_epoch": best.user_attrs.get("best_epoch"),
               "recall": best.user_attrs.get("recall"),
               "specificity": best.user_attrs.get("specificity"),
               "target_met": best.user_attrs.get("target_met"),
               "out_dir": str(best_dir), "config": str(best_dir / "config.json")}
    (base / "best_trial.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    print(f"best: trial {best.number}  value {best.value:.4f}  "
          f"(recall {summary['recall']:.4f}, specificity "
          f"{summary['specificity']:.4f}, target "
          f"{'met' if summary['target_met'] else 'NOT met'})")
    print("  " + ", ".join(f"{k}={v}" for k, v in best_eff.items()))
    print(f"  reproduce: python train.py --config {best_dir / 'config.json'}")
    print(f"  summary: {base / 'best_trial.json'}")


if __name__ == "__main__":
    main()
