# AI lightning template

A compact training template for PyTorch Lightning projects. It is designed to be runnable immediately with synthetic regression data, then easy to adapt to real datasets and models.

## Features

- Hydra configuration with composable `train`, `model`, `callbacks`, and `loggers` files.
- Reproducible Lightning training loop with seed control and deterministic mode.
- Example `LightningDataModule` and `LightningModule` that can run without external data.
- Checkpointing, early stopping, LR monitoring, CSV logging, and TensorBoard logging.
- Separate commands for training, testing, and prediction.

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch selector before installing the rest of the requirements.

## Suggested Layout

```text
configs/      Hydra configs
models/       LightningModule, DataModule, model components
scripts/      CLI entry points for train/test/predict
utils/        shared helpers
results/      generated experiment outputs, ignored by git
notebooks/    exploratory work
```
