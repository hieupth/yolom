import pytorch_lightning as pl
import get_model as gs
import random
import math
import yaml
import os
import torch.nn as nn
import torch
from multitasks.utils.datasets import create_dataloader
from arguments import training_arguments

opt = training_arguments(True)

RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class YOLOv9LightningModule(pl.LightningModule):
    def __init__(self, model_config, hyp_path, lr=1e-3):
        super(YOLOv9LightningModule, self).__init__()
        # Sử dụng setattr để gán thuộc tính
        self.loss_fn, self.gs, self.imgsz, self.model_device, self.amp, self.yolo_model = gs.get_model(opt)
        self.lr = lr

    def forward(self, x):
        return self.yolo_model(x)  # Thay self.model bằng self.yolo_model để tránh xung đột

    def training_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        imgs = imgs.to(self.model_device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        
        # Multi-scale
        if opt.multi_scale:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        with torch.cuda.amp.autocast(self.amp):
            model_outputs = self.yolo_model(imgs)  # forward
            # if don't use multiloss, auto detect last layer and compute_loss
            
            _loss, _loss_item = [], []
            for name, _loss_fn in self.loss_fn.items():
                if isinstance(targets, dict) and isinstance(model_outputs, dict):
                    _ls, _ls_items = _loss_fn(model_outputs[name], targets[name].to(self.model_device))
                else:
                    _ls, _ls_items = _loss_fn(model_outputs, targets.to(self.model_device))
                _loss.append(_ls)
                _loss_item.append(_ls_items)
            
            loss =  _loss[0] if len(_loss) == 1 else sum(_loss) / len(_loss)
            
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        imgs = imgs.to(self.model_device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
        
        # Multi-scale
        if opt.multi_scale:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        with torch.cuda.amp.autocast(self.amp):
            model_outputs = self.yolo_model(imgs)  # forward
            
            _loss, _loss_item = [], []
            for name, _loss_fn in self.loss_fn.items():
                if isinstance(targets, dict) and isinstance(model_outputs, dict):
                    _ls, _ls_items = _loss_fn(model_outputs[name], targets[name].to(self.model_device))
                else:
                    _ls, _ls_items = _loss_fn(model_outputs, targets.to(self.model_device))
                _loss.append(_ls)
                _loss_item.append(_ls_items)
            
            loss =  _loss[0] if len(_loss) == 1 else sum(_loss) / len(_loss)
            
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

# Phần dưới giữ nguyên không thay đổi.
if __name__ == '__main__':
    data_yaml_path = r"C:\Users\admin\Desktop\datasets\data.yaml"
    hyp_yaml_path = r'D:\FPT\AI\Major6\OJT_yolo\yoloxyz\backbones\yolov9\data\hyps\hyp.scratch-high.yaml'

    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    train_path = data['train']
    val_path = data['val']

    with open(hyp_yaml_path, 'r') as f:
        hyp = yaml.safe_load(f)

    batch_size = 2
    img_size = 320
    num_workers = 4

    train_loader, _ = create_dataloader(
        path=train_path, imgsz=img_size, batch_size=batch_size, stride=32, 
        hyp=hyp, augment=True, cache=False, rect=False, 
        rank=-1, workers=num_workers, opt=opt
    )

    val_loader, _ = create_dataloader(
        path=val_path, imgsz=img_size, batch_size=batch_size, stride=32, 
        hyp=hyp, augment=False, cache=False, rect=True, 
        rank=-1, workers=num_workers, opt=opt
    )

    trainer = pl.Trainer(devices=1, accelerator="gpu", max_epochs=2)

    model_config_path = r'D:\FPT\AI\Major6\OJT_yolo\yoloxyz\backbones\yolov9\models\detect\yolov9-c.yaml'
    hyp_path = r'D:\FPT\AI\Major6\OJT_yolo\yoloxyz\backbones\yolov9\data\hyps\hyp.scratch-high.yaml'
    model = YOLOv9LightningModule(model_config=model_config_path, hyp_path=hyp_path)

    trainer.fit(model, train_loader, val_loader)
