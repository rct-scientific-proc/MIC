"""Train every ResNet size on one h5 with identical options, then compare
their best checkpoints on the test split.

Each architecture trains through the real train.py CLI into
<out-dir>/<arch>/ with the same SHARED_TRAIN options (override with
--train-args), and each best checkpoint is scored by evaluate.py (its usual
report.txt / PDF in <arch>/eval_test/). The comparison itself re-runs every
best model over the test split in-process and writes to <out-dir>:

    comparison.csv     one row per architecture: operating-point metrics at
                       the stored threshold(s), threshold-free separation,
                       the threshold re-swept on test (how well the
                       validation threshold transferred), size and speed
    comparison.txt     the same as a readable table, plus pairwise exact
                       McNemar tests on per-chip decisions and the classes
                       whose recall differs most between architectures
    compare_roc.png    genuine-vs-hard-negative ROC of every architecture
    compare_history.png  validation recall / specificity / train loss per epoch

    python tests/compare_archs.py runs/cifar100_archs/cifar100_hn50.h5 \\
        --out-dir runs/cifar100_archs
    python tests/compare_archs.py DATA.h5 --out-dir DIR --skip-train   # re-compare
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

ARCHS = ("resnet18", "resnet34", "resnet50")
# identical for every architecture: 96 px (CIFAR's 32 px upsampled 3x for
# the ImageNet-pretrained stems), 30 epochs on a fixed cosine schedule with
# early stopping off so every model gets the same budget
SHARED_TRAIN = [
    "--input-size", "96", "--epochs", "30", "--batch-size", "128",
    "--lr", "3e-4", "--weight-decay", "1e-4", "--target-recall", "0.8",
    "--recall-agg", "harmonic", "--threshold-mode", "global",
    "--patience", "0", "--amp", "--workers", "4", "--seed", "0",
    "--no-progress",
]
SPEC_AT = (0.85, 0.90, 0.95)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("h5", help="dataset .h5 (h5_format.md)")
    p.add_argument("--out-dir", required=True,
                   help="parent directory; each architecture trains into <out-dir>/<arch>")
    p.add_argument("--archs", nargs="+", default=list(ARCHS), choices=ARCHS)
    p.add_argument("--train-args", default=None,
                   help="replace the shared train.py options (one quoted string)")
    p.add_argument("--skip-train", action="store_true",
                   help="compare existing runs only")
    p.add_argument("--retrain", action="store_true",
                   help="train even when <out-dir>/<arch> already holds a best checkpoint")
    p.add_argument("--split", type=int, default=2, help="split to compare on (default: test)")
    p.add_argument("--role", action="append", default=[], metavar="ARCH=ROLE",
                   help="which checkpoint of an architecture to compare: best "
                        "(default, selected on validation) or last (e.g. a run "
                        "stopped early); repeatable")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def split_args(text: str) -> list[str]:
    if os.name != "nt":
        return shlex.split(text)
    return [t[1:-1] if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'" else t
            for t in shlex.split(text, posix=False)]


def run(cmd: list, log_path: Path) -> float:
    """Run a CLI, teeing its output to log_path; returns wall seconds."""
    print(":: " + " ".join(str(c) for c in cmd[1:]), flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen([str(c) for c in cmd], cwd=REPO, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace")
        for line in proc.stdout:
            log.write(line)
            if "Warning" not in line:
                sys.stdout.write("   " + line)
                sys.stdout.flush()
        if proc.wait() != 0:
            raise SystemExit(f"command failed (exit {proc.returncode}); log: {log_path}")
    return time.time() - t0


def mcnemar_exact(a_ok, b_ok) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired per-chip correctness: (A right & B
    wrong, A wrong & B right, p)."""
    import numpy as np
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return b, c, 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            - n * math.log(2) for k in range(min(b, c) + 1)]
    top = max(logs)
    tail = math.exp(top) * sum(math.exp(v - top) for v in logs)
    return b, c, float(min(1.0, 2 * tail)) if np.isfinite(tail) else 1.0


def score_arch(run_dir: Path, h5: str, split: int, device, role: str = "best") -> dict:
    """Re-run the best checkpoint over `split` and compute the comparison."""
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    from checkpoints import find_checkpoint, load_checkpoint
    from dataset import H5SnippetDataset
    from metrics import (apply_threshold, collect_probs, final_prediction,
                         genuine_vs_hn_roc, genuineness_scores, non_hn_argmax,
                         sweep_threshold)
    from model import build_model, to_device

    ck_path = find_checkpoint(run_dir, role)
    if ck_path is None:
        raise SystemExit(f"no {role} checkpoint in {run_dir}")
    ck = load_checkpoint(ck_path, device)
    classes, hn = ck["classes"], ck["hard_negative_index"]
    model = to_device(build_model(ck["arch"], len(classes), pretrained=False,
                                  in_channels=ck.get("in_channels", 3)), device)
    model.load_state_dict(ck["model_state"])
    ds = H5SnippetDataset(h5, split, imagenet_norm=ck["imagenet_norm"],
                          input_size=ck.get("input_size"))
    loader = DataLoader(ds, batch_size=256, num_workers=2, pin_memory=True,
                        persistent_workers=False)
    amp = bool((ck.get("config") or {}).get("amp"))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    probs, labels = collect_probs(model, loader, device, progress=False, amp=amp)
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - t0

    agg = ck.get("recall_agg", "harmonic")
    per_class_mode = ck.get("threshold_mode") == "per-class" and ck.get("class_thresholds")
    operating = ck["class_thresholds"] if per_class_mode else ck["threshold"]
    target = float((ck.get("config") or {}).get("target_recall", 0.8))
    res = apply_threshold(probs, labels, hn, operating, agg=agg)
    macro = apply_threshold(probs, labels, hn, operating, agg="macro")["recall"]
    final = final_prediction(probs, hn, operating)
    genuine = labels != hn
    _, _, auroc = genuine_vs_hn_roc(probs, labels, hn)
    top1 = float((non_hn_argmax(probs, hn)[genuine] == labels[genuine]).mean())
    s = genuineness_scores(probs, hn)
    pos = np.sort(s[genuine])[::-1]
    neg = s[~genuine]
    spec_at = {}
    for r in SPEC_AT:
        t = float(pos[min(len(pos) - 1, max(0, math.ceil(r * len(pos)) - 1))])
        spec_at[r] = float((neg < t).mean())
    oracle = sweep_threshold(probs, labels, hn, target, agg=agg,
                             min_threshold=float(ck.get("min_threshold") or 0.0))
    rows = list(csv.DictReader(open(run_dir / "metrics.csv", encoding="utf-8")))
    return {
        "arch": ck["arch"], "checkpoint": ck_path.name, "role": role,
        "best_epoch": int(ck["epoch"]),
        "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        "train_gpu_min": sum(float(r["epoch_time_s"]) for r in rows) / 60,
        "infer_img_s": len(labels) / infer_s,
        "threshold": float(ck["threshold"]), "target": target,
        "val_recall": float(ck["val_metrics"]["recall"]),
        "val_specificity": float(ck["val_metrics"]["specificity"]),
        "recall": res["recall"], "macro_recall": macro,
        "target_met": res["recall"] >= target,
        "specificity": res["specificity"], "tpr": res["tpr"],
        "auroc": auroc, "genuine_top1": top1,
        **{f"spec_at_{int(r * 100)}": v for r, v in spec_at.items()},
        "test_swept_threshold": oracle["threshold"],
        "test_swept_specificity": oracle["specificity"],
        "test_swept_target_met": bool(oracle["target_met"]),
        # kept for the pairwise tests / plots, not written to the CSV
        "_per_class": res["per_class_recall"], "_classes": classes,
        "_correct": np.where(genuine, final == labels, final == hn),
        "_genuine": genuine, "_roc": genuine_vs_hn_roc(probs, labels, hn),
        "_history": rows, "_n": len(labels),
    }


CSV_COLS = ["arch", "role", "params_m", "best_epoch", "train_gpu_min", "train_wall_min",
            "infer_img_s", "threshold", "val_recall", "val_specificity", "recall",
            "macro_recall", "target_met", "specificity", "tpr", "auroc",
            "genuine_top1", "spec_at_85", "spec_at_90", "spec_at_95",
            "test_swept_threshold", "test_swept_specificity",
            "test_swept_target_met", "checkpoint"]


def write_outputs(results: list[dict], out: Path, split_name: str, shared: list[str]):
    import numpy as np
    from plots import SERIES, _legend, _new_axes, _save, _style_axes
    import matplotlib.pyplot as plt

    with open(out / "comparison.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_COLS)
        for r in results:
            w.writerow([f"{r[c]:.4f}" if isinstance(r[c], float) else r[c]
                        for c in CSV_COLS])

    rows_txt = [
        ("checkpoint compared", "role", "{}"),
        ("parameters (M)", "params_m", "{:.1f}"), ("checkpoint epoch", "best_epoch", "{}"),
        ("train GPU time (min)", "train_gpu_min", "{:.1f}"),
        ("train wall time (min)", "train_wall_min", "{:.1f}"),
        ("inference incl. loading (img/s)", "infer_img_s", "{:,.0f}"),
        ("stored threshold", "threshold", "{:.4f}"),
        ("val recall at that epoch", "val_recall", "{:.4f}"),
        ("val specificity at that epoch", "val_specificity", "{:.4f}"),
        (f"{split_name}: harmonic recall", "recall", "{:.4f}"),
        (f"{split_name}: macro recall", "macro_recall", "{:.4f}"),
        (f"{split_name}: target met", "target_met", "{}"),
        (f"{split_name}: HN specificity", "specificity", "{:.4f}"),
        (f"{split_name}: genuine acceptance (TPR)", "tpr", "{:.4f}"),
        (f"{split_name}: AUROC genuine vs HN", "auroc", "{:.4f}"),
        (f"{split_name}: genuine top-1 ({len(results[0]['_classes']) - 1}-way)",
         "genuine_top1", "{:.4f}"),
        (f"{split_name}: spec @ 85% acceptance", "spec_at_85", "{:.4f}"),
        (f"{split_name}: spec @ 90% acceptance", "spec_at_90", "{:.4f}"),
        (f"{split_name}: spec @ 95% acceptance", "spec_at_95", "{:.4f}"),
        (f"{split_name}: threshold re-swept on it", "test_swept_threshold", "{:.4f}"),
        (f"{split_name}: specificity at that", "test_swept_specificity", "{:.4f}"),
    ]
    archs = [r["arch"] for r in results]
    lines = [f"ResNet comparison on the {split_name} split "
             f"({results[0]['_n']} chips, {int(results[0]['_genuine'].sum())} genuine)",
             "shared train.py options: " + " ".join(shared), ""]
    lines.append(f"{'metric':42s}" + "".join(f"{a:>12s}" for a in archs))
    for label, key, f_ in rows_txt:
        lines.append(f"{label:42s}" + "".join(
            f"{f_.format(r[key]) if r.get(key) is not None else '-':>12s}" for r in results))

    lines += ["", "pairwise exact McNemar on per-chip decisions at each model's stored",
              "threshold (genuine: accepted as the right class; hard negative: rejected)",
              f"{'pair':22s}{'chips':>8s}{'A only':>9s}{'B only':>9s}{'p':>11s}"]
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            A, B = results[i], results[j]
            for name, mask in (("genuine", A["_genuine"]), ("hard neg", ~A["_genuine"])):
                b, c, p = mcnemar_exact(A["_correct"][mask], B["_correct"][mask])
                lines.append(f"{A['arch'][6:] + ' vs ' + B['arch'][6:] + ' ' + name:22s}"
                             f"{int(mask.sum()):8d}{b:9d}{c:9d}{p:11.2e}")

    classes = results[0]["_classes"]
    pc = {c: [r["_per_class"].get(c, float('nan')) for r in results]
          for c in results[0]["_per_class"]}
    spread = sorted(pc, key=lambda c: -(max(pc[c]) - min(pc[c])))
    lines += ["", "classes whose recall differs most between architectures",
              f"{'class':20s}" + "".join(f"{a:>12s}" for a in archs)]
    for c in spread[:10]:
        lines.append(f"{classes[c]:20s}" + "".join(f"{v:12.3f}" for v in pc[c]))
    lines += ["", "hardest classes per architecture (lowest recall)"]
    for r in results:
        worst = sorted(r["_per_class"], key=r["_per_class"].get)[:5]
        lines.append(f"  {r['arch']}: " + ", ".join(
            f"{classes[c]} {r['_per_class'][c]:.2f}" for c in worst))
    text = "\n".join(lines)
    (out / "comparison.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text)

    fig, ax = _new_axes((5.6, 5.2))
    ax.plot([0, 1], [0, 1], color="#898781", linewidth=1, linestyle=(0, (4, 4)))
    for k, r in enumerate(results):
        fpr, tpr, auc = r["_roc"]
        ax.plot(fpr, tpr, color=SERIES[k], linewidth=2, label=f"{r['arch']} (AUC {auc:.4f})")
        ax.plot([1 - r["specificity"]], [r["tpr"]], marker="o", markersize=7,
                color=SERIES[k], markeredgecolor="#fcfcfb", markeredgewidth=1.5)
    ax.set_xlabel("hard negatives accepted (1 - specificity)")
    ax.set_ylabel("genuine accepted (TPR)")
    ax.set_title(f"Genuine vs hard-negative ROC ({split_name}); dots = stored threshold")
    _legend(ax)
    _save(fig, out / "compare_roc.png")

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), facecolor="#fcfcfb",
                             sharex=True, constrained_layout=True)
    for ax, key, label in zip(axes, ("recall", "specificity", "train_loss"),
                              ("validation harmonic recall", "validation HN specificity",
                               "train loss")):
        _style_axes(ax)
        for k, r in enumerate(results):
            h = r["_history"]
            ax.plot([int(x["epoch"]) for x in h], [float(x[key]) for x in h],
                    color=SERIES[k], linewidth=2, label=r["arch"])
        ax.set_ylabel(label)
    axes[0].set_title("Training history (same options, same data)")
    axes[-1].set_xlabel("epoch")
    _legend(axes[0])
    _save(fig, out / "compare_history.png")
    print(f"\nwritten: {out / 'comparison.csv'}, comparison.txt, compare_roc.png, "
          "compare_history.png")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    shared = split_args(args.train_args) if args.train_args else SHARED_TRAIN
    py = sys.executable
    times_path = out / "train_wall_times.json"
    times = json.loads(times_path.read_text()) if times_path.exists() else {}

    from checkpoints import find_checkpoint
    from dataset import SPLIT_NAMES
    roles = {}
    for spec in args.role:
        arch, _, role = spec.partition("=")
        if arch not in ARCHS or role not in ("best", "last"):
            raise SystemExit(f"--role expects ARCH=best|last, got '{spec}'")
        roles[arch] = role
    for arch in args.archs:
        run_dir = out / arch
        if args.skip_train or (find_checkpoint(run_dir, "best") and not args.retrain):
            print(f"{arch}: using the existing run in {run_dir}")
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            times[arch] = run([py, "-u", REPO / "train.py", args.h5, "--arch", arch,
                               "--out-dir", run_dir, *shared], run_dir / "train.log")
            times_path.write_text(json.dumps(times, indent=2))
        if not (run_dir / f"eval_{SPLIT_NAMES[args.split]}" / "report.txt").exists() \
                or not args.skip_train:
            run([py, "-u", REPO / "evaluate.py",
                 find_checkpoint(run_dir, roles.get(arch, "best")), args.h5, "--split",
                 str(args.split), "--device", args.device, "--no-progress"],
                run_dir / "evaluate.log")

    import torch
    device = torch.device(args.device)
    results = []
    for arch in args.archs:
        r = score_arch(out / arch, args.h5, args.split, device,
                       roles.get(arch, "best"))
        r["train_wall_min"] = times[arch] / 60 if arch in times else None
        results.append(r)
        torch.cuda.empty_cache()
    write_outputs(results, out, SPLIT_NAMES[args.split], shared)


if __name__ == "__main__":
    main()
