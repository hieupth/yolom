import pytorch_lightning as pl
import get_model as gs
import random
import math

import os
import torch.nn as nn
import torch
from arguments import training_arguments

opt = training_arguments(True)

RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class YOLOv9LightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super(YOLOv9LightningModule, self).__init__()
        self.loss_fn, self.gs, self.imgsz, self.model_device, self.amp, self.yolo_model = gs.get_model(opt)
        self.lr = lr
        self.train_outputs = []
        self.val_outputs = []

    def forward(self, x):
        return self.yolo_model(x)

    def training_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)  # Tính thêm accuracy
        self.train_outputs.append({"loss": loss, "acc": acc})
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._common_step(batch, batch_idx)
        self.val_outputs.append({"loss": loss, "acc": acc})
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack([x["loss"] for x in self.train_outputs]).mean()
        avg_acc = torch.stack([x["acc"] for x in self.train_outputs]).mean()

        print(f"\n\nEpoch {self.current_epoch}: Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_acc:.4f}")
        
        self.train_outputs.clear()

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack([x["loss"] for x in self.val_outputs]).mean()
        avg_val_acc = torch.stack([x["acc"] for x in self.val_outputs]).mean()

        print(f"\n\nEpoch {self.current_epoch}: Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_acc:.4f}")
        
        self.val_outputs.clear()

    def _common_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        imgs = imgs.to(self.model_device, non_blocking=True).float() / 255
        if opt.multi_scale:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs
            sf = sz / max(imgs.shape[2:])
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        with torch.cuda.amp.autocast(self.amp):
            model_outputs = self.yolo_model(imgs)
            _loss, _loss_item = [], []
            for name, _loss_fn in self.loss_fn.items():
                if isinstance(targets, dict) and isinstance(model_outputs, dict):
                    _ls, _ls_items = _loss_fn(model_outputs[name], targets[name].to(self.model_device))
                else:
                    _ls, _ls_items = _loss_fn(model_outputs, targets.to(self.model_device))
                _loss.append(_ls)
                _loss_item.append(_ls_items)

            loss = _loss[0] if len(_loss) == 1 else sum(_loss) / len(_loss)
            if RANK != -1:
                loss *= WORLD_SIZE
            if opt.quad:
                loss *= 4.

            # Tính toán độ chính xác (accuracy)
            acc = self._calculate_accuracy(model_outputs, targets)
            
        return loss, acc

    def _calculate_accuracy(self, outputs, targets):
        # Implement logic to calculate accuracy here
        acc = torch.tensor(0.9)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer



