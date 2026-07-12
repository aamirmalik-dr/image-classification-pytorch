"""A CIFAR-10 architecture study in PyTorch.

Four image classifiers behind one interface, an MLP, a plain CNN, a VGG-style
network, and a small ResNet-style network, plus a regularization ablation
(batch norm, dropout, weight decay). Everything is sized to train on a CPU on a
subset of CIFAR-10 in a few minutes.
"""

from imgclf.data import (
    SyntheticImages,
    cifar10_loaders,
    class_names,
)
from imgclf.models import MLP, ResNetStyle, SimpleCNN, VGGStyle, build_model
from imgclf.train import Trainer, accuracy, set_seed

__all__ = [
    "SyntheticImages",
    "cifar10_loaders",
    "class_names",
    "MLP",
    "SimpleCNN",
    "VGGStyle",
    "ResNetStyle",
    "build_model",
    "Trainer",
    "accuracy",
    "set_seed",
]

__version__ = "0.1.0"
