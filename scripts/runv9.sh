# python yoloxyz/train.py \
#   --basemodel 'v7' \
#   --epochs 10 \
#   --workers 8 \
#   --device 0 \
#   --batch-size 10 \
#   --data yoloxyz/multitasks/cfg/data/widerface.yaml \
#   --cfg yoloxyz/multitasks/cfg/training/yolov7-tiny-multitask.yaml \
#   --name yolov7-tiny-pretrain \
#   --hyp yoloxyz/multitasks/cfg/hyp/hyp.yolov7.tiny.yaml \
#   --weight weights/yolov7-tiny.pt \
#   --sync-bn \
#   --kpt-label 5 \
#   --iou-loss EIoU \
#   --multilosses \
#   --detect-layer 'IKeypoint'

# Finetune V9
python -m torch.distributed.launch --nproc_per_node 2 --master_port 9527 yoloxyz/train.py \
    --basemodel 'v9' \
    --weights weights/yolov9-c.pt \
    --cfg yoloxyz/cfg/achitecture/yolov9-c.yaml \
    --hyp yoloxyz/cfg/hyp/hyp.scratch-high-v9.yaml \
    --data yoloxyz/cfg/data/food.yaml \
    --name finetune_v9 \
    --batch 8 \
    --epochs 2 \
    --imgsz 640 \
    --device 0,1 \
    --workers 2 \
    --close-mosaic 15 \
    --sync-bn \
    --min-items 0 \
    --detect-layer 'DualDDetect'