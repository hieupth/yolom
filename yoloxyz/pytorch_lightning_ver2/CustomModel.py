import os
import sys
sys.path.append(os.path.join(os.getcwd(), "yoloxyz"))
import torch
import math
import random
import yaml
from pytorch_lightning_ver2.arguments import training_arguments
import pytorch_lightning as pl
from backbones.yolov9.models.experimental import attempt_load
from pytorch_lightning_ver2.CustomData import CustomDataModule
import torch.nn as nn
from backbones.yolov9.utils.loss_tal_dual import ComputeLoss
from torch.optim import lr_scheduler
from backbones.yolov9.utils.torch_utils import smart_optimizer
from backbones.yolov9.utils.general import one_cycle, one_flat_cycle
from backbones.yolov9.utils.torch_utils import ModelEMA

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None#check_git_info()

class YOLOv9LightningModel(pl.LightningModule):
    def __init__(self, model, opt, gs, model_device, computeLoss = ComputeLoss):
        super(YOLOv9LightningModel, self).__init__()
        self.model = model
        self.opt = opt
        self.gs = gs
        self.model_device = model_device
        self.computeLoss = computeLoss(model)
        self.ema_model = None
        self.ema = True

        self.automatic_optimization = False
    
    def forward(self, x):
        self.model.eval()
        detections = self.model(x)
        return detections
    
    def on_train_epoch_start(self):
        if self.ema is True:
            self.ema_model = ModelEMA(self.model)
        self.mloss = torch.zeros(3, device=self.model_device)
        self.train_outputs = []


    def training_step(self, batch, batch_idx):
        images, targets, paths, _ = batch
        loss, loss_items = self.compute_loss(images, targets)

        if isinstance(targets, dict):
            tgt_name = list(targets.keys())
            total_instances = sum([targets[x].shape[0] for x in tgt_name])
        else:
            total_instances = targets.shape[0]
        
        self.mloss = loss_items

        self.train_outputs.append({"box_loss": self.mloss[0].item(), "obj_loss": self.mloss[1].item(), 
                                   "cls_loss": self.mloss[2].item(), "total_instances": total_instances})
        # Backward
        optimizer = self.optimizers()
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        if self.ema is True:
            self.ema_model.update(self.model)
        self.lr_schedulers().step()
        return {"loss": loss, "box_loss": self.mloss[0].item(), "obj_loss": self.mloss[1].item(), "cls_loss": self.mloss[2].item()}
    
    def on_train_epoch_end(self):
        avg_box_loss = torch.tensor([x["box_loss"] for x in self.train_outputs]).mean()
        avg_obj_loss = torch.tensor([x["obj_loss"] for x in self.train_outputs]).mean()
        avg_cls_loss = torch.tensor([x["cls_loss"] for x in self.train_outputs]).mean()
        avg_total_instances = torch.tensor([float(x["total_instances"]) for x in self.train_outputs]).mean()
        mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'  # (GB)

        print(f"\n\n{'Epoch':>7} {'GPU_mem':>9} {'avg_box_loss':>10} {'avg_obj_loss':>10} {'avg_cls_loss':>10} {'avg_instances':>12} {'Size':>8}")
        # Print values in the next line, aligned under the headers
        print(f"\n{self.current_epoch + 1}/{self.opt.epochs:<5} {mem:>9} {avg_box_loss:>10.5f} {avg_obj_loss:>10.5f} "
            f"{avg_cls_loss:>10.5f} {avg_total_instances:>12} {self.opt.imgsz:>8}")
        
        self.train_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        images, targets, paths, _ = batch
        loss, loss_items = self.compute_loss(images, targets)
        if self.ema_model is not None:
            model = self.ema_model.ema
        else:
            model = self.model
        return loss
    
    def compute_loss(self, images, targets):
        imgs = images.to(self.model_device, non_blocking=True).float() / 255
        imgs = imgs.clone().requires_grad_(True)
        if self.opt.multi_scale:
            sz = random.randrange(self.opt.imgsz * 0.5, self.opt.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
        pred = self.model(imgs)
        # print(pred[0])
        loss, loss_items = self.computeLoss(pred, targets.to(self.model_device))  # loss scaled by batch_size
        if RANK != -1:
            loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
        if self.opt.quad:
            loss *= 4.
        return loss, loss_items

    def configure_optimizers(self):
        with open(self.opt.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)
        optimizer = smart_optimizer(self.model, self.opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
        
        if self.opt.cos_lr:
            lf = one_cycle(1, hyp['lrf'], self.opt.epochs)  # cosine 1->hyp['lrf']
        elif self.opt.flat_cos_lr:
            lf = one_flat_cycle(1, hyp['lrf'], self.opt.epochs)  # flat cosine 1->hyp['lrf']        
        elif self.opt.fixed_lr:
            lf = lambda x: 1.0
        else:
            lf = lambda x: (1 - x / self.opt.epochs) * (1.0 - hyp['lrf']) + hyp['lrf']

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        return [optimizer], [scheduler]
    
from pytorch_lightning import Trainer

if __name__ == '__main__':
    opt = training_arguments(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load("C:/Users/admin/Desktop/datasets/yolov9-c.pt", device)
    gs = max(int(model.stride.max()), 32)
    data_module = CustomDataModule(model= model, opt = opt, gs = gs)
    model = YOLOv9LightningModel(model, opt = opt, gs = gs, model_device = device)
    trainer = Trainer(max_epochs=2)
    trainer.fit(model, data_module)


