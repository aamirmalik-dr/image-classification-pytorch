"""Compare CIFAR-10 architectures and run a regularization ablation.

Trains each architecture on a CPU-friendly subset of CIFAR-10 under one budget,
reports a test-accuracy table, and writes accuracy-curve and confusion-matrix
figures.

Usage:
    python scripts/benchmark.py --subset 5000 --epochs 8
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

from imgclf.data import cifar10_loaders, class_names
from imgclf.models import build_model
from imgclf.train import Trainer, accuracy, set_seed

ARCHITECTURES = ["mlp", "cnn", "cnn_bn_drop", "vgg", "resnet"]


def confusion_matrix(model, loader, num_classes=10, device="cpu"):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
            for t, p in zip(y.numpy(), pred, strict=False):
                cm[t, p] += 1
    return cm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(0)
    train_loader, test_loader = cifar10_loaders(subset=args.subset)
    print(f"Training on {len(train_loader.dataset)} images, "
          f"testing on {len(test_loader.dataset)} images")

    results: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    best_model, best_name, best_acc = None, None, -1.0
    for name in ARCHITECTURES:
        set_seed(0)
        wd = 5e-4 if name in {"vgg", "resnet", "cnn_bn_drop"} else 0.0
        trainer = Trainer(build_model(name), lr=1e-3, weight_decay=wd)
        trainer.fit(train_loader, test_loader, epochs=args.epochs, verbose=False)
        acc = accuracy(trainer.model, test_loader)
        results[name] = acc
        curves[name] = trainer.history["test_acc"]
        if acc > best_acc:
            best_model, best_name, best_acc = trainer.model, name, acc
        print(f"  {name:<12} test accuracy {acc:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(7, 4.5))
    for name, curve in curves.items():
        plt.plot(range(1, len(curve) + 1), curve, label=name)
    plt.xlabel("epoch")
    plt.ylabel("test accuracy")
    plt.title("CIFAR-10 architecture comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curves.png", dpi=120)
    plt.close()

    cm = confusion_matrix(best_model, test_loader)
    plt.figure(figsize=(6, 5.5))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10), class_names(), rotation=90)
    plt.yticks(range(10), class_names())
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.title(f"Confusion matrix ({best_name})")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion.png", dpi=120)
    plt.close()

    print(f"\nBest architecture: {best_name} ({best_acc:.4f}). Wrote figures to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
