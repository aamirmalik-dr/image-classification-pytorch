# image-classification-pytorch

A CIFAR-10 architecture study in PyTorch. Four network families, a plain MLP, a small CNN, a compact VGG-style stack, and a ResNet-style network with skip connections, are trained under one budget and compared, together with a batch-norm plus dropout regularization ablation on the small CNN.

## What it does

- Implements five configurations behind a single `build_model` factory: `mlp`, `cnn`, `cnn_bn_drop` (batch norm plus dropout), `vgg`, and `resnet`.
- Loads CIFAR-10 through torchvision with standard channel normalization, with optional subsampling so the whole comparison runs on a CPU in minutes.
- Trains every architecture with the same optimizer, learning rate, and epoch budget, then reports a test-accuracy table and writes accuracy-curve and confusion-matrix figures.
- Ships a synthetic image dataset so the unit tests and CI run with no download.

## What it does not do

- No data augmentation, learning-rate scheduling, or long training runs. The models are intentionally compact and the default budget is small, so absolute accuracy is well below full-dataset state of the art.
- No pretrained backbones or transfer learning. Every network is trained from random initialization.
- No hyperparameter search. Each architecture uses one fixed configuration.

## Install

```
python -m venv .venv
.venv\Scripts\activate      # Windows, or: source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python 3.11 or newer. On Linux CI, install the CPU build of PyTorch first: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`.

## Run

```
python scripts/download_data.py --root data      # optional, prefetch CIFAR-10
python scripts/benchmark.py --subset 5000 --epochs 10    # architecture comparison
python scripts/benchmark.py --subset 0 --epochs 30       # closer to full-data (slow on CPU)
pytest -q                                         # tests, fully offline
```

The demo notebook is `notebooks/demo.ipynb`, executed with saved outputs.

## Results

Produced by `python scripts/benchmark.py --subset 5000 --epochs 10`: 5000 training images and 2500 test images sampled from CIFAR-10, Adam at 1e-3, weight decay 5e-4 on the regularized and larger models, single fixed seed.

| Architecture | Test accuracy |
| --- | --- |
| mlp | 0.4084 |
| cnn | 0.5520 |
| cnn_bn_drop | 0.5876 |
| vgg | 0.5184 |
| resnet | 0.4824 |

Two things stand out, both reported as observed. First, the MLP trails every convolutional model by a wide margin: flattening the image discards the spatial structure that convolutions exploit. Second, in this deliberately small-data, short-budget regime the compact CNN with batch norm and dropout is the strongest model, and the deeper VGG-style and ResNet-style networks do not pull ahead. That is expected here: with only 5000 images and 10 epochs, the higher-capacity networks are data and iteration starved, so their extra depth does not yet pay off and can hurt. Increasing the subset and epoch budget (for example `--subset 0 --epochs 30`) is where the deeper networks are expected to overtake the small CNN. Chance accuracy on ten classes is 0.10, so all five models learn well above chance.

## Package layout

```
src/imgclf/         library code (models, data, trainer)
scripts/            download_data.py, benchmark.py
notebooks/          demo.ipynb with executed outputs
tests/              pytest suite, runs on synthetic data offline
data/               gitignored, CIFAR-10 downloaded on demand
```

## Author

Aamir Malik

- GitHub: https://github.com/aamirmalik-dr
- LinkedIn: https://linkedin.com/in/dr-aamirmalik

## License

MIT, see LICENSE.
