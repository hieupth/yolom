# import pytorch_lightning as pl
# from torch.utils.data import DataLoader, Dataset
# import torchvision.transforms as transforms
# from PIL import Image
# import os

# class YOLOv9Dataset(Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data  # List of image paths and corresponding labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         img_path, label = self.data[idx]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# class YOLOv9DataModule(pl.LightningDataModule):
#     def __init__(self, train_data, val_data, batch_size=2):
#         super().__init__()
#         self.train_data = train_data
#         self.val_data = val_data
#         self.batch_size = batch_size
#         self.transform = transforms.Compose([
#             transforms.Resize((320, 320)),  # Resize images to 320x320
#             transforms.ToTensor(),
#         ])

#     def setup(self, stage=None):
#         self.train_dataset = YOLOv9Dataset(self.train_data, transform=self.transform)
#         self.val_dataset = YOLOv9Dataset(self.val_data, transform=self.transform)

#     def train_dataloader(self):
#         return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

#     def val_dataloader(self):
#         return DataLoader(self.val_dataset, batch_size=self.batch_size)

# def get_list_data(images_dir, labels_dir):
#     data = []
#     for image_file in os.listdir(images_dir):
#         if image_file.endswith('.jpg'):
#             img_path = os.path.join(images_dir, image_file)
#             label_path = os.path.join(labels_dir, image_file.replace('.jpg', '.txt'))
#             if os.path.exists(label_path):  
#                 data.append((img_path, label_path))
#     return data


# if __name__ == '__main__':
#     train_dir = 'C:\\Users\\admin\\Desktop\\datasets\\train'
#     images_dir = os.path.join(train_dir, 'images')
#     labels_dir = os.path.join(train_dir, 'labels')

#     train_data = get_list_data(images_dir, labels_dir)
#     idx = 0  # Ví dụ chỉ số cần truy cập
#     img_path, label_path = train_data[idx]
#     # print("Image Path:", image)
#     # print("Label Path:", label_path)

import yaml
import torch
import sys
import os
sys.path.append("D:/FPT/AI/Major6/OJT_yolo/yoloxyz")
import pytorch_lightning as pl
from backbones.yolov9.utils.dataloaders import create_dataloader
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
        with open(self.data_yaml_path, 'r') as f:
            self.data = yaml.safe_load(f)

        with open(self.hyp_yaml_path, 'r') as f:
            self.hyp = yaml.safe_load(f)

    def setup(self, stage=None):
        self.train_loader, self.dataset = create_dataloader(self.data['train'],
                                              self.opt.imgsz,
                                              self.opt.batch_size,
                                              self.gs,
                                              self.opt.single_cls,
                                              hyp=self.hyp,
                                              augment=True,
                                              cache=None if self.opt.cache == 'val' else self.opt.cache,
                                              rect=self.opt.rect,
                                              rank=LOCAL_RANK,
                                              workers=self.opt.workers,
                                              image_weights=self.opt.image_weights,
                                              close_mosaic=self.opt.close_mosaic != 0,
                                              quad=self.opt.quad,
                                              prefix=colorstr('train: '),
                                              shuffle=True,
                                              min_items=self.opt.min_items)
        
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

from arguments import training_arguments
from models.experimental import attempt_load
import matplotlib.pyplot as plt
from utils.loss_tal_dual import ComputeLoss
import cv2
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

    train_loader = train_data.train_dataloader()

    for batch in train_loader:
        imgs, targets, __, _ = batch
        print(len(imgs), len(targets))
        print(type(imgs), type(targets))
        
        print(__)
        print(_)
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
        
