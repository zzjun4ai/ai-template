import torchmetrics

from models.losses import *
from models.operations import *
from models.layers import *
import lightning.pytorch as pl
from omegaconf import OmegaConf
from torch.optim.adamw import AdamW
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup


class BaseModule(pl.LightningModule):
    def __init__(self, total_steps=10000, lr=1e-3, lr_warmup_steps=100, weight_decay=1e-3, loss_name="infonce", **kwargs):
        super().__init__()
        self.total_steps = total_steps
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.weight_decay = weight_decay
        self.loss_name = loss_name
        self._loss_fn()

    def _loss_fn(self):
        pass

    def configure_optimizers(self):  # type: ignore
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
        optimizer = AdamW(
            [{"params": decay_params, "weight_decay": self.weight_decay}, {"params": no_decay_params, "weight_decay": 0.0}],
            lr=self.lr,
            betas=(0.9, 0.98),
            eps=1e-8,
        )
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer, self.lr_warmup_steps, self.total_steps, min_lr_rate=0.01)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}
