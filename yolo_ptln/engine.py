import os
import math
import torch
import random
import numpy as np
import torch.nn as nn
from torch.optim import lr_scheduler
from pathlib import Path

from utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from utils.general import LOGGER, Profile, non_max_suppression, scale_boxes, xywh2xyxy, xyxy2xywh, check_amp, one_cycle, one_flat_cycle
from utils.loss_tal_dual import ComputeLoss
from utils.torch_utils import smart_optimizer, ModelEMA
from lightning.pytorch import LightningModule

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

class LitYOLO(LightningModule):
    def __init__( self, opt, num_classes, model, model_device,  hyp, loss_fn=None ):
        super(LitYOLO, self).__init__()
        self.opt = opt
        self.dist = True if len(self.opt.device) > 1 else False
        self.num_classes = num_classes
        self.mloss = torch.zeros(3, device=model_device)  # mean losses
        self.model = model
        self.model_device = model_device
        self.loss_fn = loss_fn if loss_fn else ComputeLoss(model)
        self.gs = max(int(model.stride.max()), 32)
        self.hyp = hyp

        #validate
        self.iouv = torch.linspace(0.5, 0.95, 10, device=model_device)
        self.niou = self.iouv.numel()

        #optimizer
        amp = check_amp(model)
        self.scaler = torch.cuda.amp.GradScaler(enabled=amp)
        self.ema = ModelEMA(model) if RANK in {-1, 0} else None

        # auto optimizer
        self.automatic_optimization = False
        self.last_opt_step = -1
       
    def training_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch

        nb = self.trainer.num_training_batches
        nw = max(round(self.hyp['warmup_epochs'] * nb), 100) 
        ni = batch_idx + nb * self.current_epoch

        # Warmup
        if ni <= nw:
            xi = [0, nw]  # x interp
            # compute_loss.gr = np.interp(ni, xi, [0.0, 1.0])  # iou loss ratio (obj_loss = 1.0 or iou)
            self.accumulate = max(1, np.interp(ni, xi, [1, self.nbs / self.opt.batch_size]).round())
            for j, x in enumerate(self.optimizer.param_groups):
                # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                x['lr'] = np.interp(ni, xi, [self.hyp['warmup_bias_lr'] if j == 0 else 0.0, x['initial_lr'] * self.lf(self.current_epoch)])
                if 'momentum' in x:
                    x['momentum'] = np.interp(ni, xi, [self.hyp['warmup_momentum'], self.hyp['momentum']])

        loss, loss_item = self.compute_loss(imgs, targets, batch_idx)
        self.mloss = (self.mloss * batch_idx + loss_item) / (batch_idx + 1) 

        self.log('train/loss', loss, on_epoch=True, on_step=True, prog_bar=True, logger=True, sync_dist=self.dist)
        for idx, x in enumerate(['box', 'obj', 'cls']):
            self.log(
                f'train/{x}',
                self.mloss[idx],
                on_epoch=True, 
                on_step=True,
                prog_bar=True, 
                logger=True,
                sync_dist=self.dist
            )

        # Backward
        if RANK != -1:
            loss *= WORLD_SIZE  # gradient averaged between devices in DDP mode
        if self.opt.quad:
            loss *= 4.

        #optimizer
        self.scaler.scale(loss).backward()
        if ni - self.last_opt_step >= self.accumulate:
            self.scaler.unscale_(self.optimizer)
        
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)  # clip gradients
            self.clip_gradients(self.optimizer, gradient_clip_val=10.0, gradient_clip_algorithm="norm")
            self.scaler.step(self.optimizer)  # optimizer.step
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = ni

    def on_train_epoch_start(self):
        self.mloss = torch.zeros(3, device=self.model_device)
        self.optimizer.zero_grad()
    
    def on_train_epoch_end(self):
        self.lr = [x['lr'] for x in self.optimizer.param_groups]
        self.scheduler.step()
        self.ema.update_attr(self.model, include=['yaml', 'nc', 'hyp', 'names', 'stride', 'class_weights'])

    def on_validation_epoch_start(self):
        LOGGER.info(f'\nValidating...')
        self.cuda = self.model_device.type != 'cpu'

        self.dt = Profile(), Profile(), Profile()
        self.val_loss = torch.zeros(3, device=self.model_device)
        self.stats, self.jdict, self.ap_class = [], [], []
        self.confusion_matrix = ConfusionMatrix(nc=self.num_classes)
        self.seen = 0
        self.val_idx = 0

    def validation_step(self, batch, batch_idx, conf_thres=0.001, iou_thres= 0.6, max_det=300,
                        save_hybrid = False, augment = False, single_cls= False, plots = True,
                        save_txt = False, save_json = False, save_conf = False, save_dir=Path('')):
        self.val_idx += 1
        im, targets, paths, shapes = batch
        
        im = im.to(self.model_device, non_blocking=True).float() / 255
        nb, _, height, width = im.shape

        # Inference
        with self.dt[1]:
            preds, train_out = self.model(im) if self.loss_fn else (self.model(im, augment=augment), None)
        
        # Loss
        if self.loss_fn:
            preds = preds[1]
        else:
            preds = preds[0][1]

        # NMS
        targets[:, 2:] *= torch.tensor((width, height, width, height), device=self.model_device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:] for i in range(nb)] if save_hybrid else []  # for autolabelling
        with self.dt[2]:
            preds = non_max_suppression(preds,
                                        conf_thres,
                                        iou_thres,
                                        labels=lb,
                                        multi_label=True,
                                        agnostic=single_cls,
                                        max_det=max_det)

        # Metrics
        for si, pred in enumerate(preds):
            labels = targets[targets[:, 0] == si, 1:]
            nl, npr = labels.shape[0], pred.shape[0]  # number of labels, predictions
            path, shape = Path(paths[si]), shapes[si][0]
            correct = torch.zeros(npr, self.niou, dtype=torch.bool, device=self.model_device)  # init
            self.seen += 1

            if npr == 0:
                if nl:
                    self.stats.append((correct, *torch.zeros((2, 0), device=self.model_device), labels[:, 0]))
                    if plots:
                        self.confusion_matrix.process_batch(detections=None, labels=labels[:, 0])
                continue
        
            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_boxes(im[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_boxes(im[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels
                correct = process_batch(predn, labelsn, self.iouv)
                if plots:
                    self.confusion_matrix.process_batch(predn, labelsn)
        
            self.stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / f'{path.stem}.txt')
            if save_json:
                pass
                # save_one_json(predn, self.jdict, path, class_map)
        
        
    def on_validation_epoch_end(self, plots = True, save_dir=Path('')):
        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy

        names = self.model.names if hasattr(self.model, 'names') else self.model.module.names
        tp, fp, p, r, f1, mp, mr, map50, ap50, map = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        if len(stats) and stats[0].any():
            tp, fp, p, r, f1, ap, self.ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(int), minlength=self.num_classes) 

        # Print results
        s = ('%22s' + '%11s' * 6) % ('Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP50-95')
        LOGGER.info(s)
        pf = '%22s' + '%11i' * 2 + '%11.3g' * 4  # print format
        LOGGER.info(pf % ('all', self.seen, nt.sum(), mp, mr, map50, map))

        if nt.sum() == 0:
            LOGGER.warning(f'WARNING ⚠️ no labels found in val set, can not compute metrics without labels')

        # Print results per class
        training = self.model is not None
        verbose = bool(self.current_epoch == self.trainer.max_epochs - 1)
        if (verbose or (self.num_classes < 50 and not training)) and self.num_classes > 1 and len(stats):
            for i, c in enumerate(self.ap_class):
                LOGGER.info(pf % (names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]))


        maps = np.zeros(self.num_classes) + map
        for i, c in enumerate(self.ap_class):
            maps[c] = ap[i]
        
    
    def compute_loss(self, images, targets, batch_idx):
        imgs = images.to(self.model_device, non_blocking=True).float() / 255

        # Multi-scale
        if self.opt.multi_scale:
            sz = random.randrange(self.opt.imgsz * 0.5, self.opt.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

        pred = self.model(imgs)

        loss, loss_items = self.loss_fn(pred, targets.to(self.model_device))
        return loss, loss_items
    

    def configure_optimizers(self):
        self.nbs = 64  # nominal batch size
        self.accumulate = max(round(self.nbs / self.opt.batch_size), 1)  # accumulate loss before optimizing
        self.hyp['weight_decay'] *= self.opt.batch_size * self.accumulate / self.nbs
        optimizer = smart_optimizer(self.model, self.opt.optimizer, self.hyp['lr0'], self.hyp['momentum'], self.hyp['weight_decay'])
        
        if self.opt.cos_lr:
            self.lf = one_cycle(1, self.hyp['lrf'], self.opt.epochs)  # cosine 1->hyp['lrf']
        elif self.opt.flat_cos_lr:
            self.lf = one_flat_cycle(1, self.hyp['lrf'], self.opt.epochs)  # flat cosine 1->hyp['lrf']        
        elif self.opt.fixed_lr:
            self.lf = lambda x: 1.0
        else:
            self.lf = lambda x: (1 - x / self.opt.epochs) * (1.0 - self.hyp['lrf']) + self.hyp['lrf']  # linear
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lf)
        scheduler.last_epoch = -1
        self.optimizer = optimizer  # Save for manual control
        self.scheduler = scheduler 
        return [optimizer], [scheduler]


def process_batch(detections, labels, iouv):
    """
    Return correct prediction matrix
    Arguments:
        detections (array[N, 6]), x1, y1, x2, y2, conf, class
        labels (array[M, 5]), class, x1, y1, x2, y2
    Returns:
        correct (array[N, 10]), for 10 IoU levels
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0])).astype(bool)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & correct_class)  # IoU > threshold and classes match
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                # matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({
            'image_id': image_id,
            'category_id': class_map[int(p[5])],
            'bbox': [round(x, 3) for x in b],
            'score': round(p[4], 5)})