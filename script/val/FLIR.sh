#!/bin/bash
name='FLIR_val_metrics'
imgsz=640
gpu_id=3,6
batch_size=8
weights='your weights path'


cd ../../
CUDA_VISIBLE_DEVICES=${gpu_id} python val.py \
--data FLIR.yaml \
--weights ${weights} \
--batch-size ${batch_size} \
--name ${name} \
--imgsz ${imgsz} \
--device=0,1 \
--augment \
--conf-thres=0.5 \
--iou-thres=0.6 \
--save-txt