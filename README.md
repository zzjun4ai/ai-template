# AI lightning template

A compact training template for PyTorch Lightning projects. It provides training infrastructure without a concrete model architecture or task loss. Implement your own LightningModule before training.

## Features

- Hydra configuration with composable `train`, `model`, `callbacks`, and `loggers` files.
- Reproducible Lightning training loop with seed control and deterministic mode.
- Generic `BaseModule` with AdamW and a validation-metric LR scheduler; no task implementation.
- Synthetic regression DataModule as a replaceable data-loading example.
- Checkpointing, early stopping, LR monitoring, CSV logging, and TensorBoard logging.
- Separate commands for training, testing, and prediction.

## Setup

```bash
pip install -r requirements.txt
```

For GPU training, install the PyTorch build that matches your CUDA version from the official PyTorch selector before installing the rest of the requirements.

## Adapt and run

1. Subclass `BaseModule` in `models/model.py` (or another module), define trainable layers, and implement `forward`, `training_step`, `validation_step`, `test_step`, and `predict_step`. Prediction should return a tensor per batch for the current export script.
2. Point `_target_` in `configs/model/default.yaml` to your subclass and add its constructor arguments. Call `super().__init__()` with the optimizer/scheduler settings in your subclass.
3. Replace `train.data` in `configs/train/default.yaml` with your DataModule and its arguments. The included synthetic regression loader is only a data example.
4. Log `valid/loss` in validation, or update the scheduler and callback monitor settings together. Override `configure_optimizers` if your task needs a different optimizer or schedule.

Run commands from the template root after implementing your task:

```bash
python scripts/train.py
python scripts/train.py ckpt_path=/path/to/last.ckpt
python scripts/test.py ckpt_path=/path/to/model.ckpt
python scripts/prediction.py ckpt_path=/path/to/model.ckpt
```

`ckpt_path` defaults to `null`. Training passes it to Lightning to restore training state; testing and prediction require a checkpoint. Use the matching model and data configuration when loading it.

`python scripts/main.py` and `python scripts/pipeline.py` are aliases for training. Module execution, such as `python -m scripts.main`, is also supported.

To inspect the default configuration without implementing a model:

```bash
python scripts/main.py --cfg job --resolve
```

The default `BaseModule` deliberately raises `NotImplementedError` when task methods or trainable parameters are missing. The Docker training command also requires your task implementation first.

## Suggested Layout

```text
configs/      Hydra configs
models/       LightningModule, DataModule, model components
scripts/      CLI entry points for train/test/predict
utils/        shared helpers
results/      generated experiment outputs, ignored by git
notebooks/    exploratory work
```
