"""Evaluation metrics: confusion matrix and one-vs-rest ROC curves.

The ROC computation is implemented directly on NumPy arrays so the package does
not depend on scikit-learn. It produces per-class curves and a macro-averaged
curve interpolated onto a shared false-positive-rate grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: str = "cpu"
) -> tuple[np.ndarray, np.ndarray]:
    """Run ``model`` over ``loader`` and collect softmax probabilities.

    Args:
        model: A classifier returning raw logits.
        loader: A data loader yielding ``(images, labels)`` batches.
        device: Torch device string.

    Returns:
        A tuple ``(probs, labels)`` where ``probs`` has shape
        ``(n_samples, n_classes)`` and ``labels`` has shape ``(n_samples,)``.
    """
    model.eval()
    model.to(device)
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for x, y in loader:
        logits = model(x.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(np.asarray(y))
    return np.concatenate(all_probs), np.concatenate(all_labels)


def confusion_matrix(
    labels: np.ndarray, preds: np.ndarray, num_classes: int = 10
) -> np.ndarray:
    """Compute a confusion matrix with true classes on rows.

    Args:
        labels: Integer true labels, shape ``(n,)``.
        preds: Integer predicted labels, shape ``(n,)``.
        num_classes: Number of classes.

    Returns:
        An integer array of shape ``(num_classes, num_classes)`` where entry
        ``[t, p]`` counts examples of true class ``t`` predicted as ``p``.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(labels.astype(int), preds.astype(int), strict=True):
        cm[t, p] += 1
    return cm


def _binary_roc(scores: np.ndarray, positives: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ROC curve (fpr, tpr) for one binary problem.

    Uses the standard threshold-sweep construction over the unique scores.
    """
    order = np.argsort(-scores, kind="mergesort")
    y = positives[order].astype(np.int64)
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    total_pos = tps[-1] if tps.size else 0
    total_neg = fps[-1] if fps.size else 0
    tpr = tps / total_pos if total_pos else np.zeros_like(tps, dtype=float)
    fpr = fps / total_neg if total_neg else np.zeros_like(fps, dtype=float)
    # Prepend the origin so every curve starts at (0, 0).
    tpr = np.concatenate([[0.0], tpr])
    fpr = np.concatenate([[0.0], fpr])
    return fpr, tpr


# ``np.trapz`` was renamed to ``np.trapezoid`` in NumPy 2.0 and removed in 2.4.
_trapezoid = getattr(np, "trapezoid", None) or np.trapz


def _auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """Area under a curve by the trapezoidal rule."""
    return float(_trapezoid(tpr, fpr))


@dataclass
class RocResult:
    """One-vs-rest ROC curves and their areas.

    Attributes:
        per_class_auc: AUC for each class, length ``n_classes``.
        macro_auc: Unweighted mean of the per-class AUCs.
        mean_fpr: Shared false-positive-rate grid for the curves.
        mean_tpr: Macro-averaged true-positive rate on ``mean_fpr``.
        per_class_tpr: True-positive rate per class on ``mean_fpr``,
            shape ``(n_classes, len(mean_fpr))``.
    """

    per_class_auc: np.ndarray
    macro_auc: float
    mean_fpr: np.ndarray
    mean_tpr: np.ndarray
    per_class_tpr: np.ndarray


def roc_curves(probs: np.ndarray, labels: np.ndarray, num_classes: int = 10) -> RocResult:
    """Compute one-vs-rest ROC curves and macro-averaged AUC.

    Args:
        probs: Predicted class probabilities, shape ``(n, num_classes)``.
        labels: Integer true labels, shape ``(n,)``.
        num_classes: Number of classes.

    Returns:
        A :class:`RocResult` with per-class AUCs, the macro AUC, and a
        macro-averaged curve interpolated onto a shared grid of 200 points.
    """
    mean_fpr = np.linspace(0.0, 1.0, 200)
    interp_tprs = np.zeros((num_classes, mean_fpr.size))
    aucs = np.zeros(num_classes)
    for c in range(num_classes):
        positives = (labels == c).astype(np.int64)
        fpr, tpr = _binary_roc(probs[:, c], positives)
        aucs[c] = _auc(fpr, tpr)
        interp_tprs[c] = np.interp(mean_fpr, fpr, tpr)
        interp_tprs[c][0] = 0.0
    mean_tpr = interp_tprs.mean(axis=0)
    mean_tpr[-1] = 1.0
    return RocResult(
        per_class_auc=aucs,
        macro_auc=float(aucs.mean()),
        mean_fpr=mean_fpr,
        mean_tpr=mean_tpr,
        per_class_tpr=interp_tprs,
    )
