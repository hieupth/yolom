import yaml
import torch
import sys
import os

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import pytorch_lightning as pl
# from backbones.yolov9.utils.dataloaders import create_dataloader
from dataload.data_loader import create_dataloader
from dataload.config_data import create_cfg_data
from backbones.yolov9.utils.general import colorstr
from backbones.yolov9.models.experimental import attempt_load

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None#check_git_info()

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, model, opt, gs):
        super().__init__()
        self.data_yaml_path = opt.data
        self.hyp_yaml_path = opt.hyp
        self.opt = opt
        self.gs = gs

    def prepare_data(self):
        self.data_config_train, self.dataset_config_train = create_cfg_data(self.opt, phase= "train")
        self.data_config_valid, self.dataset_config_valid = create_cfg_data(self.opt, phase= "valid")
        with open(self.data_yaml_path, 'r') as f:
            self.data = yaml.safe_load(f)

        with open(self.hyp_yaml_path, 'r') as f:
            self.hyp = yaml.safe_load(f)

    def setup(self, stage=None):
        # self.train_loader, self.dataset = create_dataloader(self.data['train'],
        #                                       self.opt.imgsz,
        #                                       self.opt.batch_size,
        #                                       self.gs,
        #                                       self.opt.single_cls,
        #                                       hyp=self.hyp,
        #                                       augment=True,
        #                                       cache=None if self.opt.cache == 'val' else self.opt.cache,
        #                                       rect=self.opt.rect,
        #                                       rank=LOCAL_RANK,
        #                                       workers=self.opt.workers,
        #                                       image_weights=self.opt.image_weights,
        #                                       close_mosaic=self.opt.close_mosaic != 0,
        #                                       quad=self.opt.quad,
        #                                       prefix=colorstr('train: '),
        #                                       shuffle=True,
        #                                       min_items=self.opt.min_items)
        
        # self.val_loader = create_dataloader(self.data['val'],
        #                                self.opt.imgsz,
        #                                self.opt.batch_size,
        #                                self.gs,
        #                                self.opt.single_cls,
        #                                hyp=self.hyp,
        #                                cache=None if self.opt.noval else self.opt.cache,
        #                                rect=True,
        #                                rank=-1,
        #                                workers=self.opt.workers * 2,
        #                                pad=0.5,
        #                                prefix=colorstr('val: '))[0]
        self.train_loader = create_dataloader(self.data_config_train, self.dataset_config_train, self.dataset_config_train.phase)
        self.val_loader = create_dataloader(self.data_config_valid, self.dataset_config_valid, self.dataset_config_valid.phase)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

from yoloxyz.pytorch_lightning_ver2.CustomArguments import training_arguments
from backbones.yolov9.models.experimental import attempt_load
from backbones.yolov9.utils.loss_tal_dual import ComputeLoss
import math
import random
import torch.nn as nn
import numpy as np

if __name__ == '__main__':
    opt = training_arguments(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(opt.weights, device)
    compute_loss = ComputeLoss(model)
    gs = max(int(model.stride.max()), 32)
    train_data = CustomDataModule(model = model, opt = opt, gs = gs)

    train_data.prepare_data()
    train_data.setup()

    train_loader = train_data.val_dataloader()

    for batch in train_loader:
        _, imgs, targets, *_ = batch
        print(len(imgs), len(targets))
        print(type(imgs), type(targets))
        
        print()
        print()

        imgs = imgs.to(device, non_blocking=True).float() / 255
        if opt.multi_scale:
            sz = random.randrange(opt.imgsz * 0.5, opt.imgsz * 1.5 + gs) // gs * gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        with torch.no_grad():  # Ngăn chặn tính toán gradient
            pred = model(imgs)
            # print(pred[0])
            loss, loss_items = compute_loss(pred, targets.to(device))  # loss scaled by batch_size
            if RANK != -1:
                loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
            if opt.quad:
                loss *= 4.
            print(loss_items)
        break
        
