"""Config-driven CIFAR-10 architecture benchmark.

Reads one YAML config per experiment (see ``configs/``), trains each model under
its declared budget, and writes a metrics table plus figures. For the best model
it also writes the two hero artifacts: a 10x10 confusion matrix and a
one-vs-rest macro-averaged ROC curve.

Data source:
    --data cifar                     torchvision CIFAR-10 (downloads on first use)
    --data data/cifar10_sample.npz   the committed offline sample (no network)

Examples:
    python scripts/benchmark.py --configs configs --data cifar --subset 5000
    python scripts/benchmark.py --configs configs --data data/cifar10_sample.npz --epochs 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from imgclf.config import ExperimentConfig, load_configs
from imgclf.data import cifar10_loaders, class_names, sample_loaders
from imgclf.metrics import collect_predictions, confusion_matrix, roc_curves
from imgclf.models import build_from_arch
from imgclf.train import Trainer, set_seed


def get_loaders(data: str, subset: int, batch_size: int):
    """Return ``(train_loader, test_loader)`` for the chosen data source."""
    if data == "cifar":
        return cifar10_loaders(subset=subset if subset > 0 else None, batch_size=batch_size)
    return sample_loaders(data, batch_size=batch_size)


def run_experiment(cfg: ExperimentConfig, train_loader, test_loader, ckpt_dir: Path):
    """Train one configured model and return its result record and the model."""
    set_seed(cfg.train.seed)
    model = build_from_arch(
        cfg.model.arch,
        num_classes=cfg.model.num_classes,
        batch_norm=cfg.model.batch_norm,
        dropout=cfg.model.dropout,
    )
    trainer = Trainer(model, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    trainer.fit(
        train_loader,
        test_loader,
        epochs=cfg.train.epochs,
        verbose=False,
        checkpoint_path=ckpt_dir / f"{cfg.name}.pt",
    )
    record = {
        "name": cfg.name,
        "arch": cfg.model.arch,
        "batch_norm": cfg.model.batch_norm,
        "dropout": cfg.model.dropout,
        "weight_decay": cfg.train.weight_decay,
        "epochs": cfg.train.epochs,
        "best_acc": round(trainer.best_acc, 4),
        "best_epoch": trainer.best_epoch,
        "history": {k: [round(v, 4) for v in vals] for k, vals in trainer.history.items()},
    }
    return record, trainer.model


def plot_curves(records: list[dict], key: str, ylabel: str, title: str, path: Path) -> None:
    """Plot a per-epoch curve (``test_acc`` or ``train_loss``) for every model."""
    plt.figure(figsize=(7, 4.5))
    for rec in records:
        curve = rec["history"][key]
        plt.plot(range(1, len(curve) + 1), curve, marker="o", ms=3, label=rec["name"])
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion(cm: np.ndarray, name: str, path: Path) -> None:
    """Render an annotated 10x10 CIFAR-10 confusion matrix."""
    labels = class_names()
    plt.figure(figsize=(7, 6))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.xticks(range(10), labels, rotation=45, ha="right")
    plt.yticks(range(10), labels)
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"Confusion matrix, best model ({name})")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def plot_confusion_grid(panels: list[dict], path: Path) -> None:
    """Render a grid of row-normalized confusion matrices, one per architecture.

    This is the signature hero figure. Each panel is a single architecture's
    confusion matrix, row-normalized so the diagonal reads as per-class recall
    on a shared 0-to-1 color scale, which makes the architectures directly
    comparable at a glance.

    Args:
        panels: One dict per architecture with keys ``name``, ``acc`` (test
            accuracy), and ``cm`` (the integer confusion matrix).
        path: Output PNG path.
    """
    labels = class_names()
    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, panel in zip(axes, panels, strict=False):
        cm = panel["cm"].astype(np.float64)
        row_sums = cm.sum(axis=1, keepdims=True)
        norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
        im = ax.imshow(norm, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(f"{panel['name']} (acc {panel['acc']:.2f})", fontsize=11)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Per-architecture confusion matrices (row-normalized recall)", fontsize=13)
    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label="recall")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_roc(roc, name: str, path: Path) -> None:
    """Render per-class (faint) and macro-averaged (bold) one-vs-rest ROC curves."""
    labels = class_names()
    plt.figure(figsize=(7, 6))
    for c, label in enumerate(labels):
        plt.plot(
            roc.mean_fpr, roc.per_class_tpr[c], lw=1, alpha=0.5,
            label=f"{label} ({roc.per_class_auc[c]:.2f})",
        )
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="chance")
    plt.plot(
        roc.mean_fpr, roc.mean_tpr, color="black", lw=2.5,
        label=f"macro-average ({roc.macro_auc:.3f})",
    )
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    plt.title(f"One-vs-rest ROC, best model ({name})")
    plt.legend(loc="lower right", fontsize=8, title="class (AUC)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", default="configs", help="config dir or file")
    parser.add_argument("--data", default="cifar", help="'cifar' or path to sample .npz")
    parser.add_argument("--subset", type=int, default=5000, help="cifar subset size (0=full)")
    parser.add_argument("--epochs", type=int, default=0, help="override epochs (0=use config)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--out", default="results")
    parser.add_argument("--metrics-name", default="metrics.json")
    parser.add_argument("--no-hero", action="store_true", help="skip confusion/ROC figures")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    configs = load_configs(args.configs)
    if args.epochs > 0:
        for cfg in configs:
            cfg.train.epochs = args.epochs

    train_loader, test_loader = get_loaders(args.data, args.subset, args.batch_size)
    n_train = len(train_loader.dataset)
    n_test = len(test_loader.dataset)
    print(f"Data: {args.data} | train={n_train} test={n_test}")

    want_hero = not args.no_hero
    records: list[dict] = []
    panels: list[dict] = []
    best_model, best_name, best_acc = None, None, -1.0
    best_probs, best_labels = None, None
    for cfg in configs:
        rec, model = run_experiment(cfg, train_loader, test_loader, ckpt_dir)
        records.append(rec)
        print(f"  {cfg.name:<14} best_acc={rec['best_acc']:.4f} @ epoch {rec['best_epoch']}")
        if want_hero:
            probs, labels = collect_predictions(model, test_loader)
            preds = probs.argmax(axis=1)
            panels.append(
                {"name": cfg.name, "acc": rec["best_acc"],
                 "cm": confusion_matrix(labels, preds, num_classes=10)}
            )
            if rec["best_acc"] > best_acc:
                best_probs, best_labels = probs, labels
        if rec["best_acc"] > best_acc:
            best_model, best_name, best_acc = model, cfg.name, rec["best_acc"]

    summary = {
        "data": args.data,
        "n_train": n_train,
        "n_test": n_test,
        "best": {"name": best_name, "best_acc": round(best_acc, 4)},
        "results": records,
    }
    (out_dir / args.metrics_name).write_text(json.dumps(summary, indent=2))

    plot_curves(records, "test_acc", "test accuracy", "Architecture comparison",
                out_dir / "accuracy_curves.png")
    plot_curves(records, "train_loss", "train loss", "Training loss",
                out_dir / "loss_curves.png")

    if want_hero and best_model is not None and best_probs is not None:
        plot_confusion_grid(panels, out_dir / "confusion_grid.png")
        best_preds = best_probs.argmax(axis=1)
        cm = confusion_matrix(best_labels, best_preds, num_classes=10)
        plot_confusion(cm, best_name, out_dir / "confusion_matrix.png")
        roc = roc_curves(best_probs, best_labels, num_classes=10)
        plot_roc(roc, best_name, out_dir / "roc.png")
        summary["best"]["macro_auc"] = round(roc.macro_auc, 4)
        (out_dir / args.metrics_name).write_text(json.dumps(summary, indent=2))
        torch.save(best_model.state_dict(), out_dir / "best_model.pt")
        print(f"Best: {best_name} acc={best_acc:.4f} macro-AUC={roc.macro_auc:.4f}")

    print(f"Wrote metrics and figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
