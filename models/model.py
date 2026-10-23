import lightning.pytorch as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau


class BaseModule(pl.LightningModule):
    """Implement the model and task steps in a project-specific subclass."""

    def __init__(
        self,
        lr=1e-3,
        weight_decay=1e-3,
        scheduler_factor=0.5,
        scheduler_patience=5,
        scheduler_min_lr=3e-6,
        scheduler_monitor="valid/loss",
    ):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Implement forward() in your LightningModule subclass.")

    def training_step(self, batch, batch_idx):
        raise NotImplementedError("Implement training_step() and your task loss.")

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError(
            "Implement validation_step() and log the configured monitor (default: valid/loss)."
        )

    def test_step(self, batch, batch_idx):
        raise NotImplementedError("Implement test_step() for your task.")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        raise NotImplementedError("Implement predict_step() to return prediction tensors.")

    def configure_optimizers(self):
        decay_params = []
        no_decay_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            name_lower = name.lower()
            if param.ndim < 2 or "logit_scale" in name_lower or "norm" in name_lower:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        if not decay_params and not no_decay_params:
            raise NotImplementedError("Add trainable parameters in your LightningModule subclass.")
        optimizer = AdamW(
            [{"params": decay_params, "weight_decay": self.hparams.weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}],
            lr=self.hparams.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=self.hparams.scheduler_min_lr,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.hparams.scheduler_monitor},
        }
