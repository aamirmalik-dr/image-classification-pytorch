"""YAML experiment configuration for the CIFAR-10 study.

Each experiment is one architecture plus its training budget, expressed as a
small YAML file under ``configs/``. This module parses those files into typed
dataclasses so the benchmark driver stays declarative: adding a run means adding
a YAML file, not editing Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Architecture description.

    Args:
        arch: One of ``mlp``, ``cnn``, ``vgg``, ``resnet``.
        num_classes: Number of output classes.
        batch_norm: Enable batch normalization (only affects ``cnn``).
        dropout: Dropout probability in the classifier head (only ``cnn``).
    """

    arch: str
    num_classes: int = 10
    batch_norm: bool = False
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Training budget and optimizer settings."""

    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    seed: int = 0


@dataclass
class ExperimentConfig:
    """A single named experiment: one model under one training budget."""

    name: str
    model: ModelConfig
    train: TrainConfig = field(default_factory=TrainConfig)


def config_from_dict(data: dict) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from a parsed YAML mapping.

    Args:
        data: Mapping with a ``name`` key, a ``model`` mapping, and an optional
            ``train`` mapping.

    Returns:
        The typed experiment configuration.

    Raises:
        ValueError: If required keys are missing.
    """
    if "name" not in data:
        raise ValueError("config is missing required key 'name'")
    if "model" not in data:
        raise ValueError("config is missing required key 'model'")
    model = ModelConfig(**data["model"])
    train = TrainConfig(**data.get("train", {}))
    return ExperimentConfig(name=data["name"], model=model, train=train)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load a single experiment config from a YAML file."""
    text = Path(path).read_text(encoding="utf-8")
    return config_from_dict(yaml.safe_load(text))


def load_configs(source: str | Path) -> list[ExperimentConfig]:
    """Load experiment configs from a directory or a single file.

    Args:
        source: A directory of ``*.yaml`` files, or one YAML file. Directories
            are read non-recursively and sorted by filename for stable order.

    Returns:
        A list of experiment configs.

    Raises:
        FileNotFoundError: If ``source`` does not exist.
        ValueError: If a directory contains no YAML files.
    """
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"config source not found: {source}")
    if p.is_file():
        return [load_config(p)]
    files = sorted(p.glob("*.yaml")) + sorted(p.glob("*.yml"))
    if not files:
        raise ValueError(f"no YAML configs found in {source}")
    return [load_config(f) for f in files]
