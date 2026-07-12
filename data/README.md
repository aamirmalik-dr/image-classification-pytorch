# Data

## Committed sample: `cifar10_sample.npz`

This file is a small, class-balanced carve of the genuine torchvision CIFAR-10
train and test splits, not synthetic data. It holds 40 training and 15 test
images per class (400 train, 150 test) stored as CHW uint8 arrays under the keys
`x_train`, `y_train`, `x_test`, `y_test`. It exists so the offline quickstart
trains and evaluates every architecture with no network access. It is far too
small to reach meaningful accuracy, so results on it are a controlled pipeline
check, not a benchmark.

Regenerate it after downloading CIFAR-10:

```bash
python scripts/make_sample.py --root data --train-per-class 40 --test-per-class 15
```

CIFAR-10 is released for research use by the University of Toronto. Only a few
hundred images are carved here, and they are used purely to exercise the code.

## Full dataset

The full CIFAR-10 (about 340 MB when extracted) is gitignored and downloaded
through torchvision on first use:

```bash
python scripts/download_data.py --root data
```

The benchmark trains on a 5000-image subset of the full dataset by default so it
runs on a CPU in a few minutes; pass `--subset 0` for the full training set. The
unit tests use a small synthetic image dataset and need no download.
