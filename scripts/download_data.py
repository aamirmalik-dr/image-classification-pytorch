"""Download CIFAR-10 through torchvision into the data directory.

Usage:
    python scripts/download_data.py --root data
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data")
    args = parser.parse_args()
    from torchvision import datasets

    datasets.CIFAR10(root=args.root, train=True, download=True)
    datasets.CIFAR10(root=args.root, train=False, download=True)
    print(f"CIFAR-10 downloaded to {args.root}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
