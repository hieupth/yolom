import math
import torch
import random
import torch.nn as nn

from utils.loss_tal_dual import ComputeLoss 
from lightning.pytorch import LightningModule

class LitYOLO(LightningModule):
    def __init__(
        self,
        opt,
        cfg,
        model,
        model_device,
        gs, imgsz, multi_scale,
        optimizer, scheduler,
        loss_fn=None,
        dist:bool=False,
    ):
        super(LitYOLO, self).__init__()
        self.opt = opt
        self.cfg = cfg
        self.dist = dist
        self.mloss = torch.zeros(3, device=model_device)  # mean losses
        self.model = model
        self.model_device = model_device
        self.loss_fn = loss_fn if loss_fn else ComputeLoss(self.model)
        self.gs = gs
        self.imgsz = imgsz
        self.multi_scale = multi_scale
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    def training_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        loss, loss_item = self.compute_loss(imgs, targets)

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=self.dist)

        if self.mloss.device != loss_item.device:
            self.mloss = self.mloss.to(loss_item)
        self.mloss = (self.mloss * batch_idx + loss_item) / (batch_idx + 1) 

        for idx, x in enumerate(['box', 'obj', 'cls']):
            self.log(
                f'train/{x}',
                self.mloss[idx],
                on_epoch=True, 
                on_step=False,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            )

        return loss
    
    def on_train_epoch_end(self):
        self.mloss = torch.zeros(3, device=self.model_device)
    
    def validation_step(self, batch, batch_idx, conf_thres=0.001, iou_thres=0.6):
        imgs, targets, paths, _ = batch
        loss, loss_item = self.compute_loss(imgs, targets)
        return loss

    def on_validation_epoch_end(self, outputs, batch, batch_idx, dataloader_idx = 0):
        return super().on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)
    
    def compute_loss(self, images, targets):
        imgs = images.to(self.model_device, non_blocking=True).float() / 255
        if self.multi_scale:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
        pred = self.model(imgs)
        loss, loss_items = self.loss_fn(pred, targets.to(self.model_device))
        print(loss, loss_items)
        return loss, loss_items

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        return [self.optimizer], [self.scheduler]
    