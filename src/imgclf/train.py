"""Training loop and metrics for the image classifiers."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader


def set_seed(seed: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def accuracy(model: torch.nn.Module, loader: DataLoader, device: str = "cpu") -> float:
    """Top-1 accuracy of ``model`` over ``loader``."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += int((pred == y).sum())
        total += y.shape[0]
    return correct / max(total, 1)


@dataclass
class Trainer:
    """Trains an image classifier with Adam and cross-entropy."""

    model: torch.nn.Module
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    history: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "test_acc": []}
    )

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader | None = None,
        epochs: int = 8,
        verbose: bool = True,
    ) -> Trainer:
        self.model.to(self.device)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss()
        for epoch in range(epochs):
            self.model.train()
            running, n = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                loss = loss_fn(self.model(x), y)
                loss.backward()
                opt.step()
                running += loss.item() * y.shape[0]
                n += y.shape[0]
            train_loss = running / max(n, 1)
            self.history["train_loss"].append(train_loss)
            test_acc = float("nan")
            if test_loader is not None:
                test_acc = accuracy(self.model, test_loader, self.device)
            self.history["test_acc"].append(test_acc)
            if verbose:
                print(f"epoch {epoch + 1:3d}  train_loss={train_loss:.4f}  test_acc={test_acc:.4f}")
        return self
