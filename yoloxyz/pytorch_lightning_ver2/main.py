import os
import sys
sys.path.append(os.path.join(os.getcwd(), "yoloxyz"))
import torch
from pytorch_lightning_ver2.arguments import training_arguments
from backbones.yolov9.models.experimental import attempt_load
from pytorch_lightning_ver2.CustomData import CustomDataModule
from pytorch_lightning_ver2.CustomModel import YOLOv9LightningModel


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))
GIT_INFO = None#check_git_info()

    
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


