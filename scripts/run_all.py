"""Run the full experiment suite and regenerate RESULTS.md.

This drives ``benchmark.py`` twice: once for the architecture comparison (which
also writes the confusion-matrix and ROC hero figures) and once for the
regularization ablation on the small CNN. It then renders both metrics files
into ``RESULTS.md``.

Usage:
    python scripts/run_all.py --data cifar --subset 5000 --epochs 10
    python scripts/run_all.py --data data/cifar10_sample.npz --epochs 2   # offline smoke run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_benchmark(configs: str, metrics_name: str, data: str, subset: int,
                  epochs: int, out: str, hero: bool) -> None:
    cmd = [
        sys.executable, str(REPO / "scripts" / "benchmark.py"),
        "--configs", configs, "--data", data, "--subset", str(subset),
        "--out", out, "--metrics-name", metrics_name,
    ]
    if epochs > 0:
        cmd += ["--epochs", str(epochs)]
    if not hero:
        cmd += ["--no-hero"]
    subprocess.run(cmd, check=True, cwd=REPO)


def render_results(out_dir: Path) -> None:
    main = json.loads((out_dir / "metrics.json").read_text())
    ablation = json.loads((out_dir / "ablation.json").read_text())

    lines: list[str] = []
    lines.append("# Results\n")
    lines.append(
        "All numbers below were produced by `python scripts/run_all.py`. "
        f"Data source `{main['data']}`, {main['n_train']} training and "
        f"{main['n_test']} test images, Adam at 1e-3, fixed seed 0. Reported "
        "accuracy is the best-by-validation checkpoint, not the last epoch.\n"
    )

    best = main["best"]
    auc = best.get("macro_auc")
    auc_str = f", macro-averaged one-vs-rest ROC AUC {auc:.3f}" if auc is not None else ""
    lines.append("## Architecture comparison\n")
    lines.append("| Architecture | Best test accuracy | Best epoch |")
    lines.append("| --- | --- | --- |")
    for r in main["results"]:
        lines.append(f"| {r['name']} | {r['best_acc']:.4f} | {r['best_epoch']} |")
    lines.append("")
    lines.append(
        f"Best model: **{best['name']}** at {best['best_acc']:.4f} test accuracy{auc_str}.\n"
    )

    lines.append("## Regularization ablation (small CNN)\n")
    lines.append(
        "Each variant toggles one regularizer on the same three-block CNN, all "
        "else equal.\n"
    )
    lines.append("| Variant | Batch norm | Dropout | Weight decay | Best test accuracy |")
    lines.append("| --- | --- | --- | --- | --- |")
    for r in ablation["results"]:
        lines.append(
            f"| {r['name']} | {r['batch_norm']} | {r['dropout']} | "
            f"{r['weight_decay']} | {r['best_acc']:.4f} |"
        )
    lines.append("")
    lines.append("## Figures\n")
    lines.append("- `results/confusion_grid.png`, per-architecture confusion matrices, the hero figure.")
    lines.append("- `results/roc.png`, one-vs-rest ROC curves for the best model.")
    lines.append("- `results/confusion_matrix.png`, full-count confusion matrix of the best model.")
    lines.append("- `results/accuracy_curves.png`, per-epoch test accuracy per architecture.")
    lines.append("- `results/loss_curves.png`, per-epoch training loss per architecture.")
    lines.append("")

    (REPO / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Wrote RESULTS.md")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="cifar")
    parser.add_argument("--subset", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    run_benchmark("configs", "metrics.json", args.data, args.subset,
                  args.epochs, args.out, hero=True)
    run_benchmark("configs/ablation", "ablation.json", args.data, args.subset,
                  args.epochs, args.out, hero=False)
    render_results(Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
