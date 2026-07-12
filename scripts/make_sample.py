"""Carve a small, balanced CIFAR-10 sample into a committed ``.npz``.

The sample is a class-balanced subset of the genuine torchvision CIFAR-10 train
and test splits, stored as CHW uint8 arrays so it stays small enough to commit.
It powers the offline quickstart. Run this after CIFAR-10 has been downloaded
(see ``scripts/download_data.py``).

Usage:
    python scripts/make_sample.py --root data --train-per-class 40 --test-per-class 15
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _balanced_indices(labels: np.ndarray, per_class: int, num_classes: int, rng) -> np.ndarray:
    idx: list[int] = []
    for c in range(num_classes):
        pool = np.where(labels == c)[0]
        take = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
        idx.extend(take.tolist())
    idx_arr = np.array(idx)
    rng.shuffle(idx_arr)
    return idx_arr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    parser.add_argument("--out", default="data/cifar10_sample.npz")
    parser.add_argument("--train-per-class", type=int, default=40)
    parser.add_argument("--test-per-class", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from torchvision import datasets

    rng = np.random.default_rng(args.seed)
    train = datasets.CIFAR10(root=args.root, train=True, download=True)
    test = datasets.CIFAR10(root=args.root, train=False, download=True)

    # train.data is HWC uint8 (n, 32, 32, 3); transpose to CHW for the loader.
    def carve(ds, per_class):
        images = np.asarray(ds.data)
        labels = np.asarray(ds.targets)
        sel = _balanced_indices(labels, per_class, 10, rng)
        x = images[sel].transpose(0, 3, 1, 2).astype(np.uint8)
        y = labels[sel].astype(np.int64)
        return x, y

    x_train, y_train = carve(train, args.train_per_class)
    x_test, y_test = carve(test, args.test_per_class)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test
    )
    size_mb = out.stat().st_size / 1e6
    print(
        f"Wrote {out} ({size_mb:.2f} MB): "
        f"{len(x_train)} train, {len(x_test)} test images"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
