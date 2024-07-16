
import os
import math
import random
import yaml
import torch
from torch import nn
from typing import Union, Dict, Any

import torch.utils
import torch.utils.data


from yolov9.utils.loggers import Loggers
from yolov9.utils.callbacks import Callbacks
from yolov9.utils.general import (LOGGER, methods, init_seeds, check_dataset, check_suffix, intersect_dicts)
from yolov9.utils.torch_utils import torch_distributed_zero_first



LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None


class Trainer:
    def __init__(
        self,
        model,
        args,
        data_collator,
        train_dataset,
        eval_dataset,
        callbacks = None,
        optimizer = None,
        loss_fn = None
    ):
        self.args = args
        self.model = model
        self.callbacks= callbacks
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.keys_log = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95', 'val/box_loss', 'val/obj_loss', 'val/cls_loss')
        
        
        # check collator
        if data_collator is None:
            from multitasks.utils.datasets import LoadImagesAndLabels
            data_collator = LoadImagesAndLabels.collate_fn
        self.data_collator = data_collator
            
        # Initialize loss function 
        if loss_fn is None:
            loss_fn = self.wrapper_loss_context_manager()
        elif not isinstance(loss_fn, dict):
            self.loss_fn = loss_fn
        else:
            self.loss_fn = {self.args.detect_layer : loss_fn}
            
        # check callback
        if callbacks is None:
            callbacks = Callbacks()
        self.callbacks = callbacks
        
            
    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            return {self.args.detect_layer : data}

        return data
    
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train dataset")
        
        train_dataset = self.train_dataset
        data_collator = self.data_collator
        sampler = self.sampler
        
        dataloader_params = {
            "batch_size" : self.args.batch_size,
            "collate_fn" : data_collator,
            "num_workers" : self.args.workers,
            "pin_memory" : self.args.pin_memory,
            "sampler" : sampler,
            "shuffle" : True
        }
        
        return torch.utils.data.DataLoader(train_dataset, **dataloader_params)
    
    def get_eval_dataloader(self, eval_dataset):
        if eval_dataset is None or self.eval_dataset is None:
            raise ValueError("Traine: training require a eval_dataset")
        
        sampler = self.sampler 
        data_collator = self.data_collator
        
        dataloader_params = {
            "batch_size" : self.args.batch_size,
            "collate_fn" : data_collator,
            "num_workers" : self.args.workers,
            "pin_memory" : self.args.pin_memory,
            "sampler" : sampler,
            "shuffle" : False,
            "drop_last" : self.args.drop_last
        }
        
        return torch.utils.data.DataLoader(eval_dataset, **dataloader_params)
    
    def compute_loss(self, model, inputs, return_output=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        model_opt = model(inputs)
        
        if isinstance(self.loss_fn, dict):
            _loss, _loss_item = [], []
            labels = self._prepare_input(labels)
            
            for name, _loss_fn in self.loss_fn.items():
                if isinstance(labels, dict) and isinstance(model_opt, dict):
                    _ls, _ls_items = _loss_fn(model_opt[name], labels[name])
                else:
                    _ls, _ls_items = _loss_fn(model_opt, labels)
                
                _loss.append(_ls)
                _loss_item.append(_ls_items)
            loss =  _loss[0] if len(_loss) == 1 else sum(_loss) / len(_loss)
            loss_items = _loss_item[0] if len(_loss_item) == 1 else self.get_average_tensor(_loss_item)
        else:
            loss, loss_items = self.loss_fn(model_opt, labels)
            
        # if self.rank != -1:
        #     loss *= WORLD_SIZE
            
        # if self.args.quad:
        #     loss *= 4
        return loss, loss_items if not return_output else loss, loss_items, model_opt
    
    def wrapper_loss_context_manager(self):
        """
        A helper wrapper that creates an appropriate context manager for `autocast` while feeding it the desired
        arguments, depending on the situation.
        """
        if self.args.basemodel == 'v7':
            from multitasks.utils.loss import ComputeLoss as DynamicKPTSLoss
            loss_fn = DynamicKPTSLoss(self.model, **self.args)
        
        elif self.args.basemodel == 'v9':
            if self.args.method == 'gelanv9':
                from yolov9.utils.loss import ComputeLoss
                loss_fn = ComputeLoss(self.model, **self.args)
                
            elif self.args.method == 'dualv9':
                from yolov9.utils.loss_tal import ComputeLoss
                loss_fn = ComputeLoss(self.model, **self.args)
                
            elif self.args.method == 'taldualv9':
                from yolov9.utils.loss_tal_dual import ComputeLoss
                loss_fn = ComputeLoss(self.model, **self.args)
                
            elif self.args.method == 'taltripletv9':
                from yolov9.utils.loss_tal_triple import ComputeLoss
                loss_fn = ComputeLoss(self.model, **self.args)
        
        return loss_fn
    
    def wrapper_model_context_manager(self):
        check_suffix(weights, '.pt')  # check weights
        pretrained = weights.endswith('.pt')
    
        if pretrained:
            if self.opt.basemodel.lower() == 'v7':
                from multitasks.models.yolov7.yolo import ModelV7
                from multitasks.models.yolov7.experimental import attempt_load as loadV7
                
                headlayers = ['Detect', 'IDetect', 'IKeypoint', 'IDetectHead', 'IDetectBody']
                
                with torch_distributed_zero_first(LOCAL_RANK):
                    weights = loadV7(weights)  # download if not found locally
                ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
                model = ModelV7(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors'), headlayers=headlayers)  # create
                
            elif self.opt.basemodel.lower() == 'v9':
                from yolov9.models.yolo import Model as ModelV9
                from yolov9.models.experimental import attempt_download
                
                with torch_distributed_zero_first(LOCAL_RANK):
                    weights = attempt_download(weights)  # download if not found locally
                ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
                model = ModelV9(self.opt.cfg or ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')) # create
            else:
                raise NotImplemented
            
            exclude = ['anchor'] if (self.opt.cfg or hyp.get('anchors')) and not self.opt.resume else []  # exclude keys
            csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
            csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
            model.load_state_dict(csd, strict=False)  # load
            LOGGER.info(f'----- Transferred {len(csd)}/{len(model.state_dict())} items from {weights} -----')  # report
        else:
            if self.opt.basemodel.lower() == 'V7':
                model = ModelV7(self.opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'), headlayers=headlayers)  # create
            elif self.opt.basemodel.lower() == 'V9':
                model = ModelV9(self.opt.cfg, ch=3, nc=nc, anchors=hyp.get('anchors'))  # created
            else:
                raise NotImplemented
            
        return model
    
    def log(self):

        return
    
    def create_optimzier(self):
        return
    
    def create_scheduler(self):
        return
    
    def prepare_inputs(self, inputs):
        
        return
    
    def save_model(self, model:nn.Module):
        return
    
    def get_average_tensor(self):
        return
    
    def train(self):
        # Directories
        w = self.opt.save_dir / 'weights'  # weights dir
        (w.parent if self.opt.evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
        last, best = w / 'last.pt', w / 'best.pt'
        last_striped, best_striped = w / 'last_striped.pt', w / 'best_striped.pt'
        
        with open(self.opt.hyp) as f:
            hyp = yaml.safe_load(f)  # load hyps
    
        # Loggers
        data_dict = None
        if RANK in {-1, 0}:
            loggers = Loggers(self.opt.save_dir, weights, self.opt, hyp, LOGGER)  # loggers instance

            # Register actions
            for k in methods(loggers):
                self.callbacks.register_action(k, callback=getattr(loggers, k))

            # Process custom dataset artifact link
            data_dict = loggers.remote_dataset
            if self.optresume:  # If resuming runs from remote artifact
                weights, epochs, hyp, batch_size = self.opt.weights, self.opt.epochs, self.opt.hyp, self.opt.batch_size
                
        # Config
        plots = not self.opt.evolve and not self.opt.noplots  # create plots
        cuda = device.type != 'cpu'
        init_seeds(self.opt.seed + 1 + RANK, deterministic=True)
        with open(self.opt.data) as f:
            data_dict = yaml.safe_load(f)  # data dict
            
        with torch_distributed_zero_first(LOCAL_RANK):
            data_dict = data_dict or check_dataset(data)  # check if None
            
        train_path, val_path = data_dict['train'], data_dict['val']
        nc = 1 if self.opt.single_cls else int(data_dict['nc'])  # number of classes
        names = {0: 'item'} if self.opt.single_cls and len(data_dict['names']) != 1 else data_dict['names']  # class names
        is_coco = isinstance(val_path, str) and val_path.endswith('val2017.txt')  # COCO dataset
            

    def training_step(self, model:nn.Module, inputs:Union[torch.tensor, Dict[str, torch.tensor]]):
        model.train()
        
        return
    
    def evaluation_step(self):
        return
        
    def prediction_step(self):
        return