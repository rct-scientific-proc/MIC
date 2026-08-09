"""Evaluate a trained checkpoint on a dataset split using its STORED threshold.

The operating threshold was chosen on the validation split during training and
travels inside the checkpoint — this script never re-tunes it, so evaluating
on the test split stays honest.

Outputs in --out-dir (default: <checkpoint dir>/eval_<split>):
    report.txt                 metrics summary
    confusion.csv              raw confusion matrix (rows = true class)
    confusion.png              row-normalized heatmap
    roc_genuine_vs_hn.png      binary ROC with the operating point marked
    roc_per_class.png          one-vs-rest ROCs
    calibration.png            reliability diagram of the genuineness score
    history.png                metric-vs-epoch curves (when metrics.csv is
                               found next to the checkpoint, or --history-csv)

Example:
    python evaluate.py runs/exp1/best.pt data.h5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import SPLIT_NAMES, SPLIT_TEST, H5SnippetDataset, validate_h5
from metrics import (apply_threshold, calibration_bins, collect_probs,
                     final_prediction, genuine_vs_hn_roc, per_class_ovr_roc)
from model import build_model
from plots import (plot_calibration, plot_confusion, plot_genuine_vs_hn_roc,
                   plot_history, plot_per_class_rocs)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoint", help="checkpoint from train.py (best.pt / last.pt)")
    p.add_argument("h5", help="dataset .h5 file (h5_format.md)")
    p.add_argument("--split", type=int, default=SPLIT_TEST, choices=sorted(SPLIT_NAMES),
                   help="0 train, 1 validate, 2 test (default: test)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--out-dir", default=None,
                   help="default: <checkpoint dir>/eval_<split>")
    p.add_argument("--history-csv", default=None,
                   help="metrics.csv to plot (default: next to the checkpoint)")
    p.add_argument("--no-progress", action="store_true",
                   help="disable the inference progress bar (for logged runs)")
    return p.parse_args(argv)


def confusion_matrix(labels: np.ndarray, pred: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(labels * k + pred, minlength=k * k).reshape(k, k)


def main(argv=None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    hn_index = ckpt["hard_negative_index"]
    threshold = ckpt["threshold"]

    summary = validate_h5(args.h5)
    if summary["classes"] != classes:
        raise ValueError(
            f"class list mismatch: checkpoint {classes} vs dataset {summary['classes']}"
        )

    split_name = SPLIT_NAMES[args.split]
    out_dir = Path(args.out_dir or Path(args.checkpoint).parent / f"eval_{split_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    recall_agg = ckpt.get("recall_agg", "macro")

    model = build_model(ckpt["arch"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    ds = H5SnippetDataset(args.h5, args.split, imagenet_norm=ckpt["imagenet_norm"])
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    probs, labels = collect_probs(model, loader, device,
                                  desc=f"evaluate {split_name}",
                                  progress=not args.no_progress)
    res = apply_threshold(probs, labels, hn_index, threshold, agg=recall_agg)
    pred = final_prediction(probs, hn_index, threshold)
    cm = confusion_matrix(labels, pred, len(classes))

    lines = [
        f"checkpoint : {args.checkpoint} (epoch {ckpt['epoch']}, arch {ckpt['arch']})",
        f"dataset    : {args.h5}  split={split_name}  n={len(ds)}",
        f"threshold  : {threshold:.6f} (chosen on validation during training)",
        f"{recall_agg} recall (genuine classes) : {res['recall']:.4f}",
        f"HN specificity at threshold    : {res['specificity']:.4f}",
        f"genuine acceptance rate (TPR)  : {res['tpr']:.4f}",
        "per-class recall:",
    ]
    for c, r in res["per_class_recall"].items():
        lines.append(f"  {classes[c]:<20s} {r:.4f}   (n={int((labels == c).sum())})")

    try:
        fpr, tpr, auc = genuine_vs_hn_roc(probs, labels, hn_index)
        lines.append(f"genuine-vs-HN AUROC            : {auc:.4f}")
        plot_genuine_vs_hn_roc(
            fpr, tpr, auc, out_dir / "roc_genuine_vs_hn.png",
            operating_point=res,
        )
    except ValueError as e:
        lines.append(f"genuine-vs-HN ROC skipped: {e}")

    cal = calibration_bins(probs, labels, hn_index)
    lines.append(f"ECE (genuineness score, {cal['n_bins']} bins) : {cal['ece']:.4f}")
    plot_calibration(cal, out_dir / "calibration.png", threshold=threshold)

    rocs = {}
    for c in range(len(classes)):
        if c == hn_index:
            continue
        try:
            rocs[classes[c]] = per_class_ovr_roc(probs, labels, c)
        except ValueError:
            pass  # class absent from this split
    if rocs:
        dropped = plot_per_class_rocs(rocs, out_dir / "roc_per_class.png")
        if dropped:
            lines.append(
                f"roc_per_class.png shows the {len(rocs) - len(dropped)} worst-AUC "
                f"classes; omitted: {', '.join(dropped)}"
            )

    plot_confusion(cm, classes, out_dir / "confusion.png")
    header = "true\\pred," + ",".join(classes)
    np.savetxt(out_dir / "confusion.csv", cm, fmt="%d", delimiter=",",
               header=header, comments="")

    history_csv = Path(args.history_csv) if args.history_csv else \
        Path(args.checkpoint).parent / "metrics.csv"
    if history_csv.exists():
        plot_history(history_csv, out_dir / "history.png")
    else:
        lines.append(f"history plot skipped: {history_csv} not found")

    report = "\n".join(lines)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"\noutputs written to {out_dir}")


if __name__ == "__main__":
    main()
