import torch
from torch.optim import lr_scheduler

from utils.torch_utils import smart_optimizer
from utils.general import one_cycle, one_flat_cycle

def create_optimizer(model, optimizer, hyp):
    optimizer = smart_optimizer(model, optimizer, hyp['lr0'], hyp['momentum'], hyp['weight_decay'])
    return optimizer

def create_scheduler(optimizer, hyp, epochs, cos_lr, flat_cos_lr, fixed_lr):

    assert sum([cos_lr, flat_cos_lr, fixed_lr]) <= 1
    if cos_lr:
        lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    elif flat_cos_lr:
        lf = one_flat_cycle(1, hyp['lrf'], epochs)  # flat cosine 1->hyp['lrf']        
    elif fixed_lr:
        lf = lambda x: 1.0
    else:
        lf = lambda x: (1 - x / epochs) * (1.0 - hyp['lrf']) + hyp['lrf']

    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler
