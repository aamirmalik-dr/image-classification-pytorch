"""Training loop and metrics for the image classifiers."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from pathlib import Path

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


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: str = "cpu"
) -> tuple[float, float]:
    """Return ``(mean_cross_entropy, top1_accuracy)`` of ``model`` over ``loader``."""
    model.eval()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += float(loss_fn(logits, y))
        correct += int((logits.argmax(dim=1) == y).sum())
        total += y.shape[0]
    n = max(total, 1)
    return total_loss / n, correct / n


@dataclass
class Trainer:
    """Trains an image classifier with Adam and cross-entropy.

    The trainer logs per-epoch train loss, test loss, and test accuracy, and
    keeps the best-by-validation-accuracy weights so the reported model is the
    checkpoint that generalized best, not merely the last epoch.
    """

    model: torch.nn.Module
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"
    history: dict[str, list[float]] = field(
        default_factory=lambda: {"train_loss": [], "test_loss": [], "test_acc": []}
    )
    best_acc: float = -1.0
    best_epoch: int = -1
    _best_state: dict | None = None

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader | None = None,
        epochs: int = 8,
        verbose: bool = True,
        checkpoint_path: str | Path | None = None,
    ) -> Trainer:
        """Train for ``epochs`` epochs, tracking the best validation checkpoint.

        Args:
            train_loader: Training data loader.
            test_loader: Optional validation loader; drives checkpoint selection.
            epochs: Number of epochs.
            verbose: Print a one-line summary per epoch.
            checkpoint_path: If set, write the best weights to this path.

        Returns:
            ``self``, with :attr:`history`, :attr:`best_acc`, and
            :attr:`best_epoch` populated.
        """
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

            test_loss, test_acc = float("nan"), float("nan")
            if test_loader is not None:
                test_loss, test_acc = evaluate(self.model, test_loader, self.device)
                if test_acc > self.best_acc:
                    self.best_acc = test_acc
                    self.best_epoch = epoch + 1
                    self._best_state = copy.deepcopy(self.model.state_dict())
            self.history["test_loss"].append(test_loss)
            self.history["test_acc"].append(test_acc)
            if verbose:
                print(
                    f"epoch {epoch + 1:3d}  train_loss={train_loss:.4f}  "
                    f"test_loss={test_loss:.4f}  test_acc={test_acc:.4f}"
                )

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
            if checkpoint_path is not None:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                torch.save(self._best_state, checkpoint_path)
        return self
