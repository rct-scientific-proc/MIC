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

from dataset import SPLIT_TRAIN, SPLIT_VAL, H5SnippetDataset, validate_h5
from losses import FocalLoss
from metrics import (RECALL_AGGREGATES, collect_probs, genuine_vs_hn_roc,
                     sweep_threshold)
from model import ARCHS, build_model
from sampler import HardNegativeMiner, ImbalanceCapSampler

CSV_FIELDS = [
    "epoch", "train_loss", "threshold", "target_met", "recall", "recall_agg",
    "specificity", "max_recall", "auroc", "hn_alpha", "imbalance_ratio",
    "ramp_progress", "lr", "epoch_time_s",
]


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

    mi = p.add_argument_group("hard-negative mining")
    mi.add_argument("--no-mining", action="store_true",
                    help="uniform hard-negative subsampling instead of error-driven")
    mi.add_argument("--mining-random-frac", type=float, default=0.2,
                    help="share of the hard-negative budget drawn uniformly at random")

    d = p.add_argument_group("data")
    d.add_argument("--imagenet-norm", action="store_true",
                   help="ImageNet mean/std normalization (default: just /255)")

    return p.parse_args(argv)


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


def ramp_values(args, ramp_progress: int, n_genuine: int, n_hn: int) -> tuple[float, float]:
    """(hn_alpha, imbalance_ratio) at the given ramp progress.

    Linear interpolation over --ramp-epochs steps. An infinite ratio endpoint
    is interpolated in budget space (inf == the ratio that admits every hard
    negative), so the hard-negative count still grows smoothly.
    """
    alpha_end = args.hn_alpha_end if args.hn_alpha_end is not None else args.hn_alpha
    ratio_start = (args.imbalance_ratio_start
                   if args.imbalance_ratio_start is not None else args.imbalance_ratio)
    ratio_end = args.imbalance_ratio

    if args.ramp_epochs <= 0:
        return alpha_end, ratio_end

    f = min(1.0, ramp_progress / args.ramp_epochs)
    full_ratio = max(1.0, n_hn / max(n_genuine, 1))  # ratio admitting all HN
    rs = full_ratio if math.isinf(ratio_start) else ratio_start
    re = full_ratio if math.isinf(ratio_end) else ratio_end

    hn_alpha = args.hn_alpha + f * (alpha_end - args.hn_alpha)
    ratio = max(1.0, rs + f * (re - rs))
    if f >= 1.0 and math.isinf(ratio_end):
        ratio = math.inf
    return hn_alpha, ratio


def validate(model, loader, device, hn_index, target_recall, recall_agg,
             desc: str = "validate", progress: bool = True) -> dict:
    probs, labels = collect_probs(model, loader, device, desc=desc,
                                  progress=progress)
    op = sweep_threshold(probs, labels, hn_index, target_recall, agg=recall_agg)
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
                    hn_index, op, best_key, miner=None, ramp_progress=0) -> None:
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
        "recall_agg": args.recall_agg,
        "val_metrics": op,
        "best_key": best_key,
        "miner_state": miner.state_dict() if miner is not None else None,
        "ramp_progress": ramp_progress,
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

    start_epoch = 0
    best_key = None
    ramp_progress = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_key = ckpt.get("best_key")
        ramp_progress = ckpt.get("ramp_progress", 0)
        if miner is not None and ckpt.get("miner_state") is not None:
            miner.load_state_dict(ckpt["miner_state"])
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    csv_path = out_dir / "metrics.csv"
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if new_csv:
        writer.writeheader()

    epochs_since_best = 0
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        hn_alpha, ratio = ramp_values(args, ramp_progress, n_genuine, n_hn)
        criterion.set_hn_alpha(hn_alpha)
        sampler.set_ratio(ratio)
        sampler.set_epoch(epoch)

        progress = not args.no_progress
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, amp, miner=miner,
                                     desc=f"epoch {epoch} train", progress=progress)
        op = validate(model, val_loader, device, hn_index, args.target_recall,
                      args.recall_agg, desc=f"epoch {epoch} validate",
                      progress=progress)
        dt = time.time() - t0

        if op["target_met"] and args.ramp_epochs > 0:
            ramp_progress = min(ramp_progress + 1, args.ramp_epochs)

        writer.writerow({
            "epoch": epoch, "train_loss": f"{train_loss:.6f}",
            "threshold": f"{op['threshold']:.6f}", "target_met": int(op["target_met"]),
            "recall": f"{op['recall']:.6f}", "recall_agg": op["recall_agg"],
            "specificity": f"{op['specificity']:.6f}",
            "max_recall": f"{op['max_recall']:.6f}",
            "auroc": f"{op['auroc']:.6f}", "hn_alpha": f"{hn_alpha:.4f}",
            "imbalance_ratio": ratio, "ramp_progress": ramp_progress,
            "lr": args.lr, "epoch_time_s": f"{dt:.1f}",
        })
        csv_file.flush()

        key = selection_key(op)
        improved = best_key is None or key > tuple(best_key)
        marker = " *" if improved else ""
        print(
            f"epoch {epoch:3d}  loss {train_loss:.4f}  "
            f"{op['recall_agg']}-recall {op['recall']:.4f}"
            f"{'' if op['target_met'] else ' (below target)'}  "
            f"spec {op['specificity']:.4f}  thr {op['threshold']:.4f}  "
            f"auroc {op['auroc']:.4f}  {dt:.1f}s{marker}"
        )

        ckpt_kw = dict(model=model, optimizer=optimizer, scaler=scaler, epoch=epoch,
                       args=args, classes=classes, hn_index=hn_index, op=op,
                       miner=miner, ramp_progress=ramp_progress)
        if improved:
            best_key = key
            epochs_since_best = 0
            save_checkpoint(out_dir / "best.pt", best_key=best_key, **ckpt_kw)
        else:
            epochs_since_best += 1

        save_checkpoint(out_dir / "last.pt", best_key=best_key, **ckpt_kw)

        if args.patience and epochs_since_best >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    csv_file.close()
    print(f"done. checkpoints and metrics.csv in {out_dir}")


if __name__ == "__main__":
    main()
