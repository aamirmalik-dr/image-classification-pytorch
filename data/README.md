# Data

This directory is gitignored. No datasets are committed.

CIFAR-10 is downloaded through torchvision on first use:

```bash
python scripts/download_data.py --root data
```

The benchmark trains on a subset of CIFAR-10 by default so it runs on a CPU in a
few minutes; pass `--subset 0`-style options (see `--help`) or edit the call to
use the full dataset. The unit tests use a small synthetic image dataset and
need no download.
