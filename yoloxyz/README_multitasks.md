# YOLOmultitask
This tutorial give instruction how to run and setup Yolomultitask

## Installed package
```
pip install -r yoloxyz/multitasks/requirements/requirements.txt -q
```

## Data
Prepare the data follow by structure ([Download](https://drive.google.com/drive/folders/1vjAJUxpThYOlOp4bSLDg6rFZyYd4ZsHc?usp=sharing) data to test model)
```
|__ setup.py
|__ README.md
|__ yoloxyz
|__ datahub
   |__widerface
      |__images
      |   |__train
      |   |__val
      |__labels
         |__train
         |__val
```

## Download pretrain weights
- Prepare the checkpoint file in path (Download checkpoints [here](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt))
```
|__ setup.py
|__ README.md
|__ yoloxyz
|__ weights
   |__ yolov7-tiny.pt 
```

## Training
1. YoloV9
To run YoloV9, setup the hyperparameter in the config and download data | weights pretrain model in this [repo](https://github.com/WongKinYiu/yolov9)

```
CUDA_VISIBLE_DEVICES=0 python yoloxyz/train.py \
    --basemodel 'v9' \
    --weights weights/yolov9-c.pt \
    --cfg yoloxyz/cfg/architecture/yolov9-c.yaml \
    --hyp yoloxyz/cfg/hyp/hyp.scratch-high-v9.yaml \
    --data yoloxyz/cfg/data/food.yaml \
    --name finetune_v9 \
    --batch 4 \
    --epochs 2 \
    --imgsz 640 \
    --device 0 \
    --workers 2 \
    --close-mosaic 15 \
    --min-items 0 \
    --detect-layer 'DualDDetect'
```

2. YoloV7
- Currently, the training code for this project is still in progress