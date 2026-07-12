"""CIFAR-10 data loading and a synthetic dataset for offline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 channel statistics for normalization.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def class_names() -> list[str]:
    """Return the CIFAR-10 class names in label order."""
    return list(CIFAR10_CLASSES)


class SyntheticImages(Dataset):
    """Random 3x32x32 images with random labels, for tests and CI.

    The labels are weakly correlated with the mean pixel value so a model can
    reach above-chance accuracy, which lets a training test assert progress
    without any download.
    """

    def __init__(self, n: int = 256, num_classes: int = 10, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.images = rng.standard_normal((n, 3, 32, 32)).astype(np.float32)
        signal = self.images.mean(axis=(1, 2, 3))
        bins = np.quantile(signal, np.linspace(0, 1, num_classes + 1))
        labels = np.clip(np.digitize(signal, bins[1:-1]), 0, num_classes - 1)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.images[idx]), int(self.labels[idx])


def cifar10_loaders(
    root: str = "data",
    batch_size: int = 128,
    subset: int | None = 5000,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Return CIFAR-10 train and test loaders, optionally subsampled.

    Downloads CIFAR-10 through torchvision on first use. Subsampling keeps the
    demo fast on a CPU; pass ``subset=None`` for the full dataset.

    Args:
        root: Directory for the torchvision download (gitignored).
        batch_size: Batch size.
        subset: If set, use this many training and ``subset // 2`` test images.
        seed: Seed for the subsample.

    Returns:
        A ``(train_loader, test_loader)`` tuple.
    """
    from torchvision import datasets, transforms

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=tf)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=tf)

    if subset is not None:
        rng = np.random.default_rng(seed)
        train_idx = rng.choice(len(train), size=min(subset, len(train)), replace=False)
        test_idx = rng.choice(len(test), size=min(subset // 2, len(test)), replace=False)
        train = Subset(train, train_idx.tolist())
        test = Subset(test, test_idx.tolist())

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def _normalize(images: np.ndarray) -> np.ndarray:
    """Apply CIFAR-10 channel normalization to a ``(n, 3, 32, 32)`` float array."""
    mean = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.array(CIFAR10_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (images - mean) / std


class NpzImages(Dataset):
    """A dataset backed by an in-memory CHW uint8 image array.

    Images are scaled to ``[0, 1]`` and channel-normalized on load, matching the
    torchvision transform used for the full dataset.
    """

    def __init__(self, images_uint8: np.ndarray, labels: np.ndarray) -> None:
        imgs = images_uint8.astype(np.float32) / 255.0
        self.images = _normalize(imgs)
        self.labels = labels.astype(np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        return torch.from_numpy(self.images[idx]), int(self.labels[idx])


def sample_loaders(
    path: str | Path = "data/cifar10_sample.npz",
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Load train and test loaders from a committed CIFAR-10 sample ``.npz``.

    The sample file holds a few hundred CIFAR-10 images per split as CHW uint8
    arrays under the keys ``x_train``, ``y_train``, ``x_test``, ``y_test``. It
    lets the quickstart train and evaluate with no download.

    Args:
        path: Path to the ``.npz`` sample.
        batch_size: Batch size for both loaders.

    Returns:
        A ``(train_loader, test_loader)`` tuple.

    Raises:
        FileNotFoundError: If the sample file is missing.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"sample not found: {p}. Generate it with "
            "python scripts/make_sample.py after downloading CIFAR-10."
        )
    with np.load(p) as data:
        train = NpzImages(data["x_train"], data["y_train"])
        test = NpzImages(data["x_test"], data["y_test"])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
