import yaml
import torch
import sys
sys.path.append("D:/FPT/AI/Major6/OJT_yolo/yoloxyz")
import pytorch_lightning as pl
from backbones.yolov9.utils.dataloaders import create_dataloader
from backbones.yolov9.utils.general import colorstr

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, opt, data_yaml_path, hyp_yaml_path):
        super().__init__()
        self.data_yaml_path = data_yaml_path
        self.hyp_yaml_path = hyp_yaml_path
        self.opt = opt
        # self.gs = max(int(model.stride.max()), 32)
        self.gs = 32

    def prepare_data(self):
        with open(self.data_yaml_path, 'r') as f:
            self.data = yaml.safe_load(f)

        with open(self.hyp_yaml_path, 'r') as f:
            self.hyp = yaml.safe_load(f)

    def setup(self, stage=None):
        self.train_loader = create_dataloader(self.data['train'],
                                       self.opt.imgsz,
                                       self.opt.batch_size,
                                       self.gs,
                                       self.opt.single_cls,
                                       hyp=self.hyp,
                                       cache=None if self.opt.noval else self.opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=self.opt.workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]
        self.val_loader = create_dataloader(self.data['val'],
                                       self.opt.imgsz,
                                       self.opt.batch_size,
                                       self.gs,
                                       self.opt.single_cls,
                                       hyp=self.hyp,
                                       cache=None if self.opt.noval else self.opt.cache,
                                       rect=True,
                                       rank=-1,
                                       workers=self.opt.workers * 2,
                                       pad=0.5,
                                       prefix=colorstr('val: '))[0]

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
