from pathlib import Path
import numpy as np
from pytorch_lightning.core.module import LightningModule
from timm.utils import ModelEmaV2
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
import torch

from .model import get_model_from_cfg
from .loss import get_loss
from .util import mixup, get_augment_policy

from timm.utils import ModelEmaV3

from .metric import calc_metric


class MyModel(LightningModule):
    def __init__(self, cfg, mode="train"):
        super().__init__()
        self.preds = None
        self.gts = None

        self.cfg = cfg
        self.mode = mode
        
        self.model = get_model_from_cfg(cfg)

        # epoch 集計用
        self.val_outputs = []
        self.val_targets = []

        if mode != "test" and cfg.model.ema:
            self.model_ema = ModelEmaV3(
                self.model,
                decay=cfg.model.ema_decay,
                update_after_step=cfg.model.ema_update_after_step,
            )

        self.loss = get_loss(cfg)


    def forward(self, left_img, right_img):
        return self.model(left_img, right_img)

    def training_step(self, batch, batch_idx):
        left_img, right_img, targets = batch  # (B, 5)
        targets = targets.float()

        # outputs = (total, gdm, green, clover, dead)
        outputs = self(left_img, right_img)
        loss_dict = self.loss(outputs, targets)

        self.log_dict(
            loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss_dict["loss"]

    def on_train_batch_end(self, out, batch, batch_idx):
        if self.cfg.model.ema:
            self.model_ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        left_img, right_img, targets = batch
        targets = targets.float()

        outputs = self(left_img, right_img)
        loss_dict = self.loss(outputs, targets)

        self.log("val_loss", loss_dict["loss"], prog_bar=True, sync_dist=True)
        
        self.val_outputs.append(tuple(o.detach() for o in outputs))
        self.val_targets.append(targets.detach())

        return loss_dict

    def on_validation_epoch_end(self):
        outputs = torch.cat(
            [torch.stack(t, dim=1) for t in self.val_outputs],
            dim=0
        ).cpu().numpy()
        outputs = outputs.squeeze(-1)  # (N, 5)

        targets = torch.cat(self.val_targets).cpu().numpy()

        weighted_r2, r2_scores = calc_metric(self.cfg, outputs, targets)

        # メトリクスをログ
        self.log("val_weighted_r2", weighted_r2, prog_bar=True)

        # 複数ターゲットなら個別ログも可
        for i, r2 in enumerate(r2_scores):
            self.log(f"val_r2_target_{i}", r2)

        # 次epochに向けてクリア
        self.val_outputs.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(model_or_params=self.model, **self.cfg.opt)

        scheduler, _ = create_scheduler_v2(
            optimizer=optimizer,
            num_epochs=self.cfg.trainer.max_epochs,
            **self.cfg.scheduler
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "val_weighted_r2",
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step(epoch=self.current_epoch)
    #     # scheduler.step_update(num_updates=self.global_step)
