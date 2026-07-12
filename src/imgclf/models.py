"""Four CIFAR-10 architectures with a shared factory.

All models take 3x32x32 input tensors and return class logits of shape
``(batch, num_classes)``. They are intentionally compact so a CPU can train them
on a subset of CIFAR-10 quickly, while still spanning the range from a plain MLP
to a residual network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """A flatten-then-dense baseline that ignores spatial structure."""

    def __init__(self, num_classes: int = 10, hidden: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleCNN(nn.Module):
    """A small convolutional network with optional regularization.

    The ablation flags toggle batch normalization and dropout so the effect of
    each can be measured against the plain convolutional baseline.
    """

    def __init__(
        self,
        num_classes: int = 10,
        batch_norm: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            layers: list[nn.Module] = [nn.Conv2d(cin, cout, 3, padding=1)]
            if batch_norm:
                layers.append(nn.BatchNorm2d(cout))
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
        head: list[nn.Module] = [nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU()]
        if dropout > 0:
            head.append(nn.Dropout(dropout))
        head.append(nn.Linear(256, num_classes))
        self.classifier = nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class VGGStyle(nn.Module):
    """A compact VGG-style network: stacked 3x3 conv blocks with batch norm."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        def conv(cin: int, cout: int) -> list[nn.Module]:
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU()]

        self.features = nn.Sequential(
            *conv(3, 64), *conv(64, 64), nn.MaxPool2d(2),
            *conv(64, 128), *conv(128, 128), nn.MaxPool2d(2),
            *conv(128, 256), *conv(256, 256), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(256 * 4 * 4, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class _ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + self.shortcut(x))


class ResNetStyle(nn.Module):
    """A small ResNet-style network with residual blocks and skip connections."""

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = _ResidualBlock(32, 32)
        self.layer2 = _ResidualBlock(32, 64, stride=2)
        self.layer3 = _ResidualBlock(64, 128, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """Construct a model from one of the named presets.

    Args:
        name: One of ``mlp``, ``cnn``, ``cnn_bn_drop``, ``vgg``, ``resnet``.
        num_classes: Number of output classes.

    Raises:
        ValueError: If the name is not recognized.
    """
    builders = {
        "mlp": lambda: MLP(num_classes),
        "cnn": lambda: SimpleCNN(num_classes),
        "cnn_bn_drop": lambda: SimpleCNN(num_classes, batch_norm=True, dropout=0.3),
        "vgg": lambda: VGGStyle(num_classes),
        "resnet": lambda: ResNetStyle(num_classes),
    }
    if name not in builders:
        raise ValueError(f"unknown model {name!r}; choose from {list(builders)}")
    return builders[name]()


def build_from_arch(
    arch: str,
    num_classes: int = 10,
    batch_norm: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    """Construct a model from an architecture family plus regularization flags.

    This is the config-driven entry point used by the benchmark. Unlike
    :func:`build_model`, the ``batch_norm`` and ``dropout`` knobs are explicit,
    which is what the regularization ablation varies on the small CNN.

    Args:
        arch: One of ``mlp``, ``cnn``, ``vgg``, ``resnet``.
        num_classes: Number of output classes.
        batch_norm: Enable batch normalization (only affects ``cnn``).
        dropout: Dropout probability in the classifier head (only ``cnn``).

    Raises:
        ValueError: If ``arch`` is not recognized.
    """
    if arch == "mlp":
        return MLP(num_classes)
    if arch == "cnn":
        return SimpleCNN(num_classes, batch_norm=batch_norm, dropout=dropout)
    if arch == "vgg":
        return VGGStyle(num_classes)
    if arch == "resnet":
        return ResNetStyle(num_classes)
    raise ValueError(f"unknown arch {arch!r}; choose from ['mlp', 'cnn', 'vgg', 'resnet']")
