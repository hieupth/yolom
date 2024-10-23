import os
import sys
import math
import random
import time
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy 
from torch.optim import lr_scheduler

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

# YoloV7
# from multitasks.models.yolov7.yolo import ModelV7
# from multitasks.models.yolov7.experimental import attempt_load as loadV7
from backbones.yolov9.utils.general import check_dataset

# YoloV9
from backbones.yolov9.models.yolo import Model as ModelV9
from backbones.yolov9.utils.general import (yaml_save, LOGGER, methods, init_seeds, check_suffix,
                                            intersect_dicts, check_amp, check_img_size, one_cycle, one_flat_cycle,
                                            colorstr, labels_to_class_weights, labels_to_image_weights, TQDM_BAR_FORMAT,
                                            strip_optimizer, print_args, check_file, check_yaml, increment_path, 
                                            get_latest_run, print_mutation
                                        )
from backbones.yolov9.utils.loggers.comet.comet_utils import check_comet_resume
from backbones.yolov9.utils.callbacks import Callbacks
from backbones.yolov9.utils.metrics import fitness
from backbones.yolov9.utils.autobatch import check_train_batch_size
from backbones.yolov9.utils.loggers import Loggers
from backbones.yolov9.utils.torch_utils import (torch_distributed_zero_first, smart_optimizer, ModelEMA, smart_resume, 
                                        smart_DDP, EarlyStopping, select_device, de_parallel
                                            )
from backbones.yolov9.models.experimental import attempt_download, attempt_load as loadV9 
from backbones.yolov9.utils.downloads import is_url
from backbones.yolov9.utils.plots import plot_evolve
from backbones.yolov9.utils.loss_tal_dual import ComputeLoss as ComputeLossV9

from arguments import training_arguments
from multitasks.utils.datasets import create_dataloader
from multitasks.utils.loss import ComputeLoss as ComputeLossV7
# import yoloxyz.test2 as validate
from backbones.yolov9 import val as validate


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None


def train(hyp, opt, device, callbacks):
    save_dir, epochs, batch_size, weights, single_cls, evolve, data, cfg, resume, noval, nosave, workers, freeze, multilosses, \
        detect_layer, warmup, basemodel, kpt_label = Path(opt.save_dir), opt.epochs, opt.batch_size, opt.weights, opt.single_cls, opt.evolve, \
        opt.data, opt.cfg, opt.resume, opt.noval, opt.nosave, opt.workers, opt.freeze, opt.multilosses, opt.detect_layer, opt.warmup, opt.basemodel, opt.kpt_label
    callbacks.run('on_pretrain_routine_start')

    # Directories
    w = save_dir / 'weights'  # weights dir
    (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / 'last.pt', w / 'best.pt'
    last_striped, best_striped = w / 'last_striped.pt', w / 'best_striped.pt'
    
    with open(opt.hyp) as f:
        hyp = yaml.safe_load(f)  # load hyps
    
    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset
        if resume:  # If resuming runs from remote artifact
            weights, epochs, hyp, batch_size = opt.weights, opt.epochs, opt.hyp, opt.batch_size
            
    
    # Config
    plots = not evolve and not opt.noplots  # create plots
    cuda = device.type != 'cpu'
    init_seeds(opt.seed + 1 + RANK, deterministic=True)
    with open(opt.data) as f:
        data_dict = yaml.safe_load(f)  # data dict
        
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)  # check if None
        
    train_path, val_path = data_dict['train'], data_dict['val']
    nc = 1 if single_cls else int(data_dict['nc'])  # number of classes
    names = {0: 'item'} if single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
    #is_coco = isinstance(val_path, str) and val_path.endswith('coco/val2017.txt')  # COCO dataset
    is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')  # COCO dataset

    # Model
    check_suffix(weights, '.pt')  # check weights
    pretrained = weights.endswith('.pt')
    headlayers = ['Detect', 'IDetect', 'IKeypoint', 'IDetectHead', 'IDetectBody'] # Define a list of Head layer support YoloV7
    
    if pretrained:
        if basemodel.lower() == 'v7':
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = loadV7(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model = ModelV7(opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), headlayers=headlayers).to(device)  # create
            
        elif basemodel.lower() == 'v9':
            with torch_distributed_zero_first(LOCAL_RANK):
                weights = attempt_download(weights)  # download if not found locally
            ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
            model = ModelV9(cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
        else:
            raise NotImplemented
        
        exclude = ['anchor'] if (cfg or hyp.get('anchors')) and not resume else []  # exclude keys
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        model.load_state_dict(csd, strict=False)  # load
        LOGGER.info(f'----- Transferred {len(csd)}/{len(model.state_dict())} items from {weights} -----')  # report
    else:
        if basemodel.lower() == 'v7':
            model = ModelV7(opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), headlayers=headlayers).to(device)  # create
        elif basemodel.lower() == 'v9':
            model = ModelV9(cfg, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # created
        else:
            raise NotImplemented

    amp = check_amp(model) # check AMP
    
    # Freeze
    if freeze is not None:
        _freeze = []
        for _sub_layer in freeze.split(','):
                start, end = _sub_layer.split('-')
                _freeze.extend([f'model.{x}.' for x in range(int(start), int(end))])  # layers to freeze
        for k, v in model.named_parameters():
            v.requires_grad = True  # train all layers
            if any(x in k for x in _freeze):
                print(f'freezing {k}')
                v.requires_grad = False
                
    # Image size
    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple

    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})
        
    
    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    optimizer = smart_optimizer(model, opt.optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])

    # Scheduler
    if opt.cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    elif opt.flat_cos_lr:
        lf = one_flat_cycle(1, hyp['lrf'], epochs)  # flat cosine 1->hyp['lrf']        
    elif opt.fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']  # linear
        
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    
    # EMA
    ema = ModelEMA(model) if RANK in {-1, 0} else None

    # Resume
    start_epoch, best_fitness = 0, 0.0
    if pretrained:
        if resume:
            best_fitness, start_epoch, epochs = smart_resume(ckpt, optimizer, ema, weights, epochs, resume)
        del ckpt, csd
        
    # DP mode
    if cuda and RANK == -1 and torch.cuda.device_count() > 1:
        LOGGER.warning('WARNING ⚠️ DP not recommended, use torch.distributed.run for best DDP Multi-GPU results.')
        model = torch.nn.DataParallel(model)
        
    # SyncBatchNorm
    if opt.sync_bn and cuda and RANK != -1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        LOGGER.info('Using SyncBatchNorm()')
        
    # Trainloader
    train_loader, dataset = create_dataloader(train_path, imgsz, batch_size, gs, opt,
                                            hyp=hyp, augment=True, cache=opt.cache, rect=opt.rect, rank=LOCAL_RANK,
                                            world_size=WORLD_SIZE, workers=workers,
                                            image_weights=opt.image_weights, quad=opt.quad, prefix=colorstr('train: '), kpt_label=kpt_label, multiloss=multilosses)

    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    if not multilosses:
        assert mlc < nc, f'Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}'
        
    # Process 0 
    # val_batch_size = int(batch_size / 4)
    val_batch_size = 2
    if RANK in {-1, 0}:
        val_loader = create_dataloader(val_path, imgsz, val_batch_size, gs, opt,  # testloader
                                       hyp=hyp, cache=opt.cache and not opt.notest, rect=True, rank=-1,
                                       world_size=WORLD_SIZE, workers=workers,
                                       pad=0.5, prefix=colorstr('val: '), kpt_label=kpt_label,
                                       multiloss=multilosses)[0]
        
        if not resume:
            # if not opt.noautoanchor:
            #     check_anchors(dataset, model=model, thr=hyp['anchor_t'], imgsz=imgsz)  # run AutoAnchor
            model.half().float()  # pre-reduce anchor precision

        callbacks.run('on_pretrain_routine_end', labels, names)
        
    # DDP mode
    if cuda and RANK != -1:
        model = smart_DDP(model)

    hyp['label_smoothing'] = opt.label_smoothing
    model.nc = nc  # attach number of classes to model
    model.hyp = hyp  # attach hyperparameters to model
    model.gr = 1.0  # iou loss ratio (obj_loss = 1.0 or iou)
    model.iou_loss = opt.iou_loss
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc  # attach class weights
    model.names = names
    
    # Start training
    t0 = time.time()
    nb = len(train_loader)  # number of batches
    if warmup:
        nw = max(round(hyp['warmup_epochs'] * nb), 100)  # number of warmup iterations, max(3 epochs, 100 iterations)
    last_opt_step = -1
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # P, R, mAP@.5, mAP@.5-.95, val_loss(box, obj, cls)
    scheduler.last_epoch = start_epoch - 1  # do not move
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    stopper, stop = EarlyStopping(patience=opt.patience), False
    
    # init loss function
    if basemodel == 'v7':
        if multilosses:
            loss_fn = {
                'IKeypoint' : ComputeLossV7(model, kpt_label=kpt_label, detect_layer='IKeypoint'),
                'IDetectHead' : ComputeLossV7(model, detect_layer='IDetectHead'),
                'IDetectBody' : ComputeLossV7(model, detect_layer='IDetectBody')
            }
        else:
            loss_fn = {
                detect_layer : ComputeLossV7(model, detect_layer=detect_layer, kpt_label=kpt_label) # Default
            }
    else:
        loss_fn = {
                detect_layer : ComputeLossV9(model) # Default
            }
        
    LOGGER.info(f'Image sizes {imgsz} train, {imgsz} val\n'
                f'Using {train_loader.num_workers * WORLD_SIZE} dataloader workers\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting training for {epochs} epochs...')
    
    return loss_fn, gs, imgsz, amp, model, data_dict, ema

def get_model(opt, callbacks=Callbacks()):
    # Checks
    if RANK in {-1, 0}:
        print_args(vars(opt))

    # Resume (from specified or most recent last.pt)
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / 'opt.yaml'  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors='ignore') as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location='cpu')['opt']
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = '', str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = \
            check_file(opt.data), check_yaml(opt.cfg), check_yaml(opt.hyp), str(opt.weights), str(opt.project)  # checks
        assert len(opt.cfg) or len(opt.weights), 'either --cfg or --weights must be specified'
        if opt.evolve:
            if opt.project == str(ROOT / 'runs/train'):  # if default project name, rename to runs/evolve
                opt.project = str(ROOT / 'runs/evolve')
            opt.exist_ok, opt.resume = opt.resume, False  # pass resume to exist_ok and disable resume
        if opt.name == 'cfg':
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = 'is not compatible with YOLO Multi-GPU DDP training'
        assert not opt.image_weights, f'--image-weights {msg}'
        assert not opt.evolve, f'--evolve {msg}'
        assert opt.batch_size != -1, f'AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size'
        assert opt.batch_size % WORLD_SIZE == 0, f'--batch-size {opt.batch_size} must be multiple of WORLD_SIZE'
        assert torch.cuda.device_count() > LOCAL_RANK, 'insufficient CUDA devices for DDP command'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # Train
    if not opt.evolve:
        loss_fn, gs, imgsz, amp, model, data_dict, ema = train(opt.hyp, opt, device, callbacks)

    # Evolve hyperparameters (optional)
    else:
        model = None
        print("Wrong hyperparam")
    
    return loss_fn, gs, imgsz, device, amp, model, data_dict, ema

