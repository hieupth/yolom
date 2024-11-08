import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path

from utils.metrics import ConfusionMatrix, box_iou, ap_per_class
from utils.general import LOGGER, Profile, non_max_suppression, scale_boxes, xywh2xyxy, xyxy2xywh
from utils.loss_tal_dual import ComputeLoss
# from utils.loss_tal import ComputeLoss 
from utils.plots import plot_images, output_to_target
from lightning.pytorch import LightningModule

class LitYOLO(LightningModule):
    def __init__(
        self,
        opt,
        cfg,
        num_classes,
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
        self.num_classes = num_classes
        self.mloss = torch.zeros(3, device=model_device)  # mean losses
        self.model = model
        self.model_device = model_device
        self.loss_fn = loss_fn if loss_fn else ComputeLoss(model)
        self.gs = gs
        self.imgsz = imgsz
        self.multi_scale = multi_scale
        self.optimizer = optimizer
        self.scheduler = scheduler

        #validate
        self.iouv = torch.linspace(0.5, 0.95, 10, device=model_device)
        self.niou = self.iouv.numel()
    
    def training_step(self, batch, batch_idx):
        imgs, targets, paths, _ = batch
        loss, loss_item = self.compute_loss(imgs, targets)

        self.log('train/loss', loss, on_epoch=True, on_step=False, prog_bar=True, logger=True, sync_dist=self.dist)

        # if self.mloss.device != loss_item.device:
        # self.mloss = self.mloss.to(loss_item)
        self.mloss = (self.mloss * batch_idx + loss_item) / (batch_idx + 1) 
        with open(r"C:\Users\admin\Desktop\fake3.txt", "a") as f:
            f.write(f"batch_i {batch_idx}: len(preds) {self.mloss}\n")

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
    
    def on_train_epoch_start(self):
        self.mloss = torch.zeros(3, device=self.model_device)
    
    def on_validation_epoch_start(self):
        LOGGER.info(f'\nValidating...\n')
        self.cuda = self.model_device.type != 'cpu'

        self.dt = Profile(), Profile(), Profile()
        self.val_loss = torch.zeros(3, device=self.model_device)
        self.stats, self.jdict, self.ap_class = [], [], []
        self.confusion_matrix = ConfusionMatrix(nc=self.num_classes)
        self.seen = 0
        self.val_idx = 0
        with open(r"C:\Users\admin\Desktop\fake1.txt", "a") as f:
            f.write(f"\n\n\n------------------------------------------------------\n")

    def validation_step(self, batch, batch_idx, conf_thres=0.001, iou_thres= 0.6, max_det=300,
                        save_hybrid = False, augment = False, single_cls= False, plots = True,
                        save_txt = False, save_json = False, save_conf = False, save_dir=Path('')):
        self.val_idx += 1
        im, targets, paths, shapes = batch
        
        im = im.to(self.model_device, non_blocking=True).float() / 255
        nb, _, height, width = im.shape

        # with open(r"C:\Users\admin\Desktop\fake2.txt", "a") as f:
        #     f.write(f"batch_i {batch_idx}: len(im) {len(im)}\n")
        #     for idx, item in enumerate(im):
        #         f.write(f"idx {idx}: {item} \n\n")
        
        # Inference
        with self.dt[1]:
            # preds, train_out = self.model(im) if self.loss_fn else (self.model(im, augment=augment), None)
            preds, train_out = self.model(im, augment=augment)
        
        # with open(r"C:\Users\admin\Desktop\fake1.txt", "a") as f:
        #     f.write(f"batch_i {batch_idx}: len(preds) {len(preds)}\n")
        #     f.write(f"type of ans: {type(self.model(im))} -- {type(self.model(im, augment=augment))}")
        #     for idx, item in enumerate(preds):
        #         f.write(f"idx {idx}: {item} \n\n")

        # Loss
        if self.loss_fn:
            preds = preds[1]
            #train_out = train_out[1]
            # self.val_loss += self.loss_fn(train_out, targets)[1]  # box, obj, cls
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
                # print("Predictions (predn):", predn)
                # print("Ground Truth Labels (labelsn):", labelsn)
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
        
        # return loss

    def on_validation_epoch_end(self, plots = True, save_dir=Path(''), verbose=False):
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # with open(r"C:\Users\admin\Desktop\fake1.txt", "a") as f:
        #     f.write(f"{now} len(stats) {len(self.stats)}\n")
        #     for idx, item in enumerate(self.stats[-1]):
        #         f.write(f"idx {idx}: {item} \n\n")

        stats = [torch.cat(x, 0).cpu().numpy() for x in zip(*self.stats)]  # to numpy

        # with open(r"C:\Users\admin\Desktop\fake2.txt", "a") as f:
        #     f.write(f"{now} len(stats) {len(stats)}\n")
        #     for idx, item in enumerate(stats):
        #         f.write(f"idex {idx}: {len(item)} \n")

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
        if (verbose or (self.num_classes < 50 and not training)) and self.num_classes > 1 and len(stats):
            for i, c in enumerate(self.ap_class):
                LOGGER.info(pf % (names[c], self.seen, nt[c], p[i], r[i], ap50[i], ap[i]))


        maps = np.zeros(self.num_classes) + map
        for i, c in enumerate(self.ap_class):
            maps[c] = ap[i]
        loss = (self.val_loss.cpu() / self.val_idx).tolist()
        print("log", self.val_loss, mp, mr, map50, map)
        print(pf % ('all', self.seen, nt.sum(), mp, mr, map50, map))

        # for idx, name in enumerate(['box_loss', 'obj_loss', 'cls_loss']):
        #     self.log(
        #         f"val/{name}", loss[idx],
        #         on_epoch=True, 
        #         on_step=False,
        #         prog_bar=True, 
        #         logger=True,
        #         sync_dist=self.dist
        #     )

        # for _, (name, value) in enumerate(zip(['mp', 'mr', 'map50', 'map'], [mp, mr, map50, map])):
        #     self.log(
        #         f'val/{name}',
        #         value,
        #         on_epoch=True, 
        #         on_step=False,
        #         prog_bar=True, 
        #         logger=True,
        #         sync_dist=self.dist
        #     ) 
        return sum(loss) / len(loss)
    
    def compute_loss(self, images, targets):
        imgs = images.to(self.model_device, non_blocking=True).float() / 255
        if self.multi_scale:
            sz = random.randrange(self.imgsz * 0.5, self.imgsz * 1.5 + self.gs) // self.gs * self.gs  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [math.ceil(x * sf / self.gs) * self.gs for x in imgs.shape[2:]]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)
        pred = self.model(imgs)
        # with open(r"C:\Users\admin\Desktop\fake3.txt", "a") as f:
        #     f.write(f"batch_i: len(preds) {len(pred)}\n")
        #     for idx, item in enumerate(pred):
        #         for i, ite in enumerate(item):
        #             f.write(f"idx {idx} - {i}: {ite} \n\n")

        loss, loss_items = self.loss_fn(pred, targets.to(self.model_device))
        return loss, loss_items

    def configure_optimizers(self):
        # self.hparams available because we called self.save_hyperparameters()
        optimizer, scheduler = self.optimizer, self.scheduler
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