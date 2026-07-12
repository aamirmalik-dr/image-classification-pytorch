from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from imgclf.config import config_from_dict, load_configs
from imgclf.data import SyntheticImages, class_names, sample_loaders
from imgclf.metrics import confusion_matrix, roc_curves
from imgclf.models import build_from_arch, build_model
from imgclf.train import Trainer, accuracy, set_seed

MODELS = ["mlp", "cnn", "cnn_bn_drop", "vgg", "resnet"]
CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


def test_class_names():
    assert len(class_names()) == 10
    assert "airplane" in class_names()


def test_all_models_output_shape():
    x = torch.randn(4, 3, 32, 32)
    for name in MODELS:
        model = build_model(name)
        out = model(x)
        assert out.shape == (4, 10)


def test_build_model_unknown_raises():
    import pytest

    with pytest.raises(ValueError):
        build_model("does_not_exist")


def test_synthetic_dataset_shapes():
    ds = SyntheticImages(n=32)
    img, label = ds[0]
    assert img.shape == (3, 32, 32)
    assert 0 <= label < 10


def test_accuracy_range():
    ds = SyntheticImages(n=32)
    loader = DataLoader(ds, batch_size=8)
    acc = accuracy(build_model("cnn"), loader)
    assert 0.0 <= acc <= 1.0


def test_trainer_reduces_loss_on_synthetic():
    set_seed(0)
    train = DataLoader(SyntheticImages(n=128, seed=0), batch_size=32, shuffle=True)
    trainer = Trainer(build_model("cnn"), lr=1e-3)
    trainer.fit(train, epochs=3, verbose=False)
    assert trainer.history["train_loss"][-1] < trainer.history["train_loss"][0]


def test_build_from_arch_regularization_flags():
    plain = build_from_arch("cnn", batch_norm=False, dropout=0.0)
    reg = build_from_arch("cnn", batch_norm=True, dropout=0.3)
    has_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in reg.modules())
    no_bn = any(isinstance(m, torch.nn.BatchNorm2d) for m in plain.modules())
    assert has_bn and not no_bn


def test_config_from_dict_defaults_and_override():
    cfg = config_from_dict({"name": "x", "model": {"arch": "cnn", "batch_norm": True}})
    assert cfg.name == "x"
    assert cfg.model.arch == "cnn" and cfg.model.batch_norm is True
    assert cfg.train.epochs == 10 and cfg.train.seed == 0


def test_load_configs_directory_is_sorted_and_nonempty():
    cfgs = load_configs(CONFIGS_DIR)
    names = [c.name for c in cfgs]
    assert names == ["mlp", "cnn", "cnn_bn_drop", "vgg", "resnet"]


def test_confusion_matrix_counts():
    labels = np.array([0, 0, 1, 1])
    preds = np.array([0, 1, 1, 1])
    cm = confusion_matrix(labels, preds, num_classes=2)
    assert cm.tolist() == [[1, 1], [0, 2]]
    assert cm.sum() == 4


def test_roc_perfect_separation_gives_auc_one():
    # Two classes, probabilities that perfectly rank the true class first.
    probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    roc = roc_curves(probs, labels, num_classes=2)
    assert roc.macro_auc > 0.99
    assert roc.per_class_auc.shape == (2,)


def test_sample_loaders_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    x_train = rng.integers(0, 256, size=(20, 3, 32, 32), dtype=np.uint8)
    y_train = np.arange(20) % 10
    x_test = rng.integers(0, 256, size=(10, 3, 32, 32), dtype=np.uint8)
    y_test = np.arange(10) % 10
    npz = tmp_path / "sample.npz"
    np.savez_compressed(npz, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    train_loader, test_loader = sample_loaders(npz, batch_size=8)
    xb, yb = next(iter(train_loader))
    assert xb.shape[1:] == (3, 32, 32)
    assert xb.dtype == torch.float32
    assert len(train_loader.dataset) == 20 and len(test_loader.dataset) == 10
