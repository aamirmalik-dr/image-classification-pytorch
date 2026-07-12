import torch
from torch.utils.data import DataLoader

from imgclf.data import SyntheticImages, class_names
from imgclf.models import build_model
from imgclf.train import Trainer, accuracy, set_seed

MODELS = ["mlp", "cnn", "cnn_bn_drop", "vgg", "resnet"]


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
