# Results

All numbers below were produced by `python scripts/run_all.py`. Data source `cifar`, 2500 training and 1250 test images, Adam at 1e-3, fixed seed 0. Reported accuracy is the best-by-validation checkpoint, not the last epoch.

## Architecture comparison

| Architecture | Best test accuracy | Best epoch |
| --- | --- | --- |
| mlp | 0.3904 | 4 |
| cnn | 0.4744 | 6 |
| cnn_bn_drop | 0.5024 | 6 |
| vgg | 0.5072 | 5 |
| resnet | 0.4528 | 6 |

Best model: **vgg** at 0.5072 test accuracy, macro-averaged one-vs-rest ROC AUC 0.892.

## Regularization ablation (small CNN)

Each variant toggles one regularizer on the same three-block CNN, all else equal.

| Variant | Batch norm | Dropout | Weight decay | Best test accuracy |
| --- | --- | --- | --- | --- |
| plain | False | 0.0 | 0.0 | 0.4744 |
| batchnorm | True | 0.0 | 0.0 | 0.5168 |
| dropout | False | 0.3 | 0.0 | 0.4592 |
| weight_decay | False | 0.0 | 0.0005 | 0.4792 |
| all | True | 0.3 | 0.0005 | 0.5024 |

## Figures

- `results/confusion_grid.png`, per-architecture confusion matrices, the hero figure.
- `results/roc.png`, one-vs-rest ROC curves for the best model.
- `results/confusion_matrix.png`, full-count confusion matrix of the best model.
- `results/accuracy_curves.png`, per-epoch test accuracy per architecture.
- `results/loss_curves.png`, per-epoch training loss per architecture.
