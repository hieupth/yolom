import yaml
import torch
import pytorch_lightning as pl
from multitasks.utils.datasets import create_dataloader

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, opt, data_yaml_path, hyp_yaml_path):
        super().__init__()
        self.data_yaml_path = data_yaml_path
        self.hyp_yaml_path = hyp_yaml_path
        self.opt = opt

    def prepare_data(self):
        with open(self.data_yaml_path, 'r') as f:
            self.data = yaml.safe_load(f)

        with open(self.hyp_yaml_path, 'r') as f:
            self.hyp = yaml.safe_load(f)

    def setup(self, stage=None):
        self.train_loader, _ = create_dataloader(
            path=self.data['train'], imgsz=self.opt.imgsz, batch_size=self.opt.batch_size, 
            stride=32, hyp=self.hyp, augment=True, cache=False, rect=False, 
            rank=-1, workers=self.opt.workers, opt=self.opt
        )
        self.val_loader, _ = create_dataloader(
            path=self.data['val'], imgsz=self.opt.imgsz, batch_size=self.opt.batch_size, 
            stride=32, hyp=self.hyp, augment=True, cache=False, rect=False, 
            rank=-1, workers=self.opt.workers, opt=self.opt
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader
