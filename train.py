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
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from controller import SmartController
from dataset import SPLIT_TRAIN, SPLIT_VAL, H5SnippetDataset, validate_h5
from losses import FocalLoss
from metrics import (RECALL_AGGREGATES, collect_probs, genuine_vs_hn_roc,
                     sweep_class_thresholds, sweep_threshold)
from model import ARCHS, build_model
from sampler import HardNegativeMiner, ImbalanceCapSampler

CSV_FIELDS = [
    "epoch", "train_loss", "threshold", "threshold_mode", "thr_min", "thr_max",
    "target_met", "recall", "recall_agg", "specificity", "max_recall", "auroc",
    "hn_alpha", "imbalance_ratio", "ramp_progress", "pressure", "cycle",
    "event", "lr", "epoch_time_s",
]
CLASS_CSV_FIELDS = ["epoch", "class", "threshold", "recall", "predicted_n",
                    "fallback", "alpha", "repeat"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("h5", help="dataset .h5 file (h5_format.md)")
    p.add_argument("--out-dir", default=None,
                   help="output directory (default: runs/<timestamp>)")

    m = p.add_argument_group("model")
    m.add_argument("--arch", choices=sorted(ARCHS), default="resnet18")
    m.add_argument("--no-pretrained", action="store_true",
                   help="random init instead of ImageNet weights")
    m.add_argument("--weights-path", default=None,
                   help="local ImageNet .pth (from download_weights.py) for offline use")

    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    t.add_argument("--workers", type=int, default=0, help="DataLoader workers")
    t.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    t.add_argument("--patience", type=int, default=10,
                   help="early-stop after N epochs without improvement (0 = off)")
    t.add_argument("--no-progress", action="store_true",
                   help="disable per-batch progress bars (for logged runs)")
    t.add_argument("--resume", default=None, help="checkpoint to resume from")

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

    s = p.add_argument_group(
        "smart mode",
        "adaptive alternative to the fixed ramp: cyclic (half-cosine) learning "
        "rate, with hard-negative pressure raised/held/rewound at cycle "
        "boundaries based on validation metrics. Pressure endpoints reuse the "
        "ramp flags (--hn-alpha -> --hn-alpha-end, --imbalance-ratio-start -> "
        "--imbalance-ratio). Mutually exclusive with --ramp-epochs.",
    )
    s.add_argument("--smart", action="store_true",
                   help="enable the smart controller")
    s.add_argument("--lr-cycle-epochs", type=int, default=10,
                   help="epochs per LR cycle; decisions happen at the trough")
    s.add_argument("--lr-min", type=float, default=None,
                   help="trough learning rate (default: --lr / 25)")
    s.add_argument("--pressure-step", type=float, default=0.25,
                   help="initial pressure increment per successful cycle")
    s.add_argument("--max-rewinds", type=int, default=3,
                   help="rewinds at one pressure level before accepting it "
                        "as the run's ceiling")
    s.add_argument("--keep-top-k", type=int, default=3,
                   help="snapshots/ archive size (best cycle checkpoints)")
    s.add_argument("--rescue", action="store_true",
                   help="class rescue: at each cycle trough, boost the loss "
                        "weight and sampling of genuine classes lagging the "
                        "recall target (smart mode only); pressure raises are "
                        "blocked while any class is under rescue")
    s.add_argument("--rescue-alpha-max", type=float, default=3.0,
                   help="cap on a rescued class's focal alpha (scales with "
                        "its recall deficit)")
    s.add_argument("--rescue-oversample-max", type=int, default=3,
                   help="cap on a rescued class's per-epoch repeat factor")
    s.add_argument("--rescue-ema", type=float, default=0.5,
                   help="EMA smoothing of per-class recall across troughs "
                        "(0 = react to the latest trough only)")

    mi = p.add_argument_group("hard-negative mining")
    mi.add_argument("--no-mining", action="store_true",
                    help="uniform hard-negative subsampling instead of error-driven")
    mi.add_argument("--mining-random-frac", type=float, default=0.2,
                    help="share of the hard-negative budget drawn uniformly at random")

    d = p.add_argument_group("data")
    d.add_argument("--imagenet-norm", action="store_true",
                   help="ImageNet mean/std normalization (default: just /255)")

    args = p.parse_args(argv)
    if args.smart and args.ramp_epochs > 0:
        p.error("--smart and --ramp-epochs are mutually exclusive (smart mode "
                "replaces the fixed ramp)")
    if args.rescue and not args.smart:
        p.error("--rescue requires --smart (rescue decisions run at cycle "
                "troughs)")
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
    seed_everything(args.seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = args.amp and device.type == "cuda"

    out_dir = Path(args.out_dir or f"runs/{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = validate_h5(args.h5)
    classes = summary["classes"]
    hn_index = summary["hard_negative_index"]
    print(f"dataset: {args.h5}")
    for split_name, c in summary["counts"].items():
        print(f"  {split_name}: {c['genuine']} genuine, {c['hard_negative']} hard negatives")

    train_ds = H5SnippetDataset(args.h5, SPLIT_TRAIN, imagenet_norm=args.imagenet_norm)
    val_ds = H5SnippetDataset(args.h5, SPLIT_VAL, imagenet_norm=args.imagenet_norm)

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    base_alphas = parse_class_alphas(args.class_alpha, classes, hn_index)
    if base_alphas:
        criterion.set_class_alphas(
            {c: base_alphas.get(c, 1.0) for c in range(len(classes))
             if c != hn_index})

    controller = None
    if args.smart:
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
    ramp_progress = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        # best_key describes out_dir/best.pt; when resuming into a fresh
        # directory that file doesn't exist, so the new run must track its
        # own best from scratch or it may never write one.
        best_key = ckpt.get("best_key") if (out_dir / "best.pt").exists() else None
        ramp_progress = ckpt.get("ramp_progress", 0)
        if miner is not None and ckpt.get("miner_state") is not None:
            miner.load_state_dict(ckpt["miner_state"])
        if controller is not None and ckpt.get("controller_state") is not None:
            controller.load_state_dict(ckpt["controller_state"], out_dir=out_dir)
        print(f"resumed from {args.resume} at epoch {start_epoch}")

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
            epochs_since_best = 0
            save_checkpoint(out_dir / "best.pt", best_key=best_key, **ckpt_kw)
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
                    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                                  weight_decay=args.weight_decay)
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

        # last.pt carries the post-decision state (post-rewind weights and
        # controller state), so --resume continues exactly where the
        # controller left off; its metrics/threshold match its weights.
        save_checkpoint(out_dir / "last.pt", model=model, optimizer=optimizer,
                        scaler=scaler, epoch=epoch, args=args, classes=classes,
                        hn_index=hn_index, op=op_for_last, miner=miner,
                        ramp_progress=ramp_progress, controller=controller,
                        best_key=best_key)

        if stop:
            break
        if controller is None and args.patience and epochs_since_best >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    csv_file.close()
    class_csv_file.close()
    print(f"done. checkpoints and metrics.csv in {out_dir}")


if __name__ == "__main__":
    main()
