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

from dataset import SPLIT_TRAIN, SPLIT_VAL, H5SnippetDataset, validate_h5
from losses import FocalLoss
from metrics import collect_probs, genuine_vs_hn_roc, sweep_threshold
from model import ARCHS, build_model
from sampler import ImbalanceCapSampler

CSV_FIELDS = [
    "epoch", "train_loss", "threshold", "target_met", "macro_recall",
    "specificity", "max_macro_recall", "auroc", "hn_alpha", "imbalance_ratio",
    "lr", "epoch_time_s",
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
    t.add_argument("--resume", default=None, help="checkpoint to resume from")

    o = p.add_argument_group("objective")
    o.add_argument("--target-recall", type=float, default=0.95,
                   help="target macro recall over genuine classes")
    o.add_argument("--imbalance-ratio", type=float, default=math.inf,
                   help="max hard negatives per epoch = ratio * genuine count (1..inf)")
    o.add_argument("--focal-gamma", type=float, default=2.0)
    o.add_argument("--hn-alpha", type=float, default=0.25,
                   help="focal alpha for the hard_negative class (genuine classes = 1)")

    d = p.add_argument_group("data")
    d.add_argument("--imagenet-norm", action="store_true",
                   help="ImageNet mean/std normalization (default: just /255)")

    return p.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp) -> float:
    model.train()
    total_loss, total_n = 0.0, 0
    for imgs, labs, idxs in loader:
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

        total_loss += float(loss.detach()) * len(labs)
        total_n += len(labs)
    return total_loss / max(total_n, 1)


def validate(model, loader, device, hn_index, target_recall) -> dict:
    probs, labels = collect_probs(model, loader, device)
    op = sweep_threshold(probs, labels, hn_index, target_recall)
    try:
        _, _, auroc = genuine_vs_hn_roc(probs, labels, hn_index)
    except ValueError:
        auroc = float("nan")
    op["auroc"] = auroc
    return op


def selection_key(op: dict) -> tuple:
    """Checkpoint ranking: meeting the target dominates; then specificity at
    the target; then macro recall (the tie-breaker while the target is out of
    reach)."""
    spec = op["specificity"]
    if math.isnan(spec):
        spec = -math.inf
    return (int(op["target_met"]), spec if op["target_met"] else -math.inf,
            op["macro_recall"])


def save_checkpoint(path: Path, *, model, optimizer, scaler, epoch, args, classes,
                    hn_index, op, best_key) -> None:
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
        "val_metrics": op,
        "best_key": best_key,
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

    sampler = ImbalanceCapSampler(
        train_ds.labels, hn_index, ratio=args.imbalance_ratio, seed=args.seed,
    )
    loader_kw = dict(num_workers=args.workers, pin_memory=device.type == "cuda")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, **loader_kw)

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained,
                        weights_path=args.weights_path).to(device)
    criterion = FocalLoss(len(classes), hn_index, gamma=args.focal_gamma,
                          hn_alpha=args.hn_alpha).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start_epoch = 0
    best_key = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_key = ckpt.get("best_key")
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
        sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, amp)
        op = validate(model, val_loader, device, hn_index, args.target_recall)
        dt = time.time() - t0

        writer.writerow({
            "epoch": epoch, "train_loss": f"{train_loss:.6f}",
            "threshold": f"{op['threshold']:.6f}", "target_met": int(op["target_met"]),
            "macro_recall": f"{op['macro_recall']:.6f}",
            "specificity": f"{op['specificity']:.6f}",
            "max_macro_recall": f"{op['max_macro_recall']:.6f}",
            "auroc": f"{op['auroc']:.6f}", "hn_alpha": f"{args.hn_alpha:.4f}",
            "imbalance_ratio": sampler.ratio, "lr": args.lr,
            "epoch_time_s": f"{dt:.1f}",
        })
        csv_file.flush()

        key = selection_key(op)
        improved = best_key is None or key > tuple(best_key)
        marker = " *" if improved else ""
        print(
            f"epoch {epoch:3d}  loss {train_loss:.4f}  "
            f"recall {op['macro_recall']:.4f}{'' if op['target_met'] else ' (below target)'}  "
            f"spec {op['specificity']:.4f}  thr {op['threshold']:.4f}  "
            f"auroc {op['auroc']:.4f}  {dt:.1f}s{marker}"
        )

        if improved:
            best_key = key
            epochs_since_best = 0
            save_checkpoint(out_dir / "best.pt", model=model, optimizer=optimizer,
                            scaler=scaler, epoch=epoch, args=args, classes=classes,
                            hn_index=hn_index, op=op, best_key=best_key)
        else:
            epochs_since_best += 1

        save_checkpoint(out_dir / "last.pt", model=model, optimizer=optimizer,
                        scaler=scaler, epoch=epoch, args=args, classes=classes,
                        hn_index=hn_index, op=op, best_key=best_key)

        if args.patience and epochs_since_best >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    csv_file.close()
    print(f"done. checkpoints and metrics.csv in {out_dir}")


if __name__ == "__main__":
    main()
