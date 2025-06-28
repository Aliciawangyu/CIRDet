epoch=80
gpu_id=0,4,5,6
hyp='yourproj/data/hyps/flir.yaml'
name='test'
imgsz=640
bs=8
optimizer='AdamW'

cd ../../
CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
--epochs ${epoch} \
--data FLIR.yaml \
--weights yolov5l.pt \
--device=0,1,2,3 \
--hyp ${hyp} \
--name ${name} \
--imgsz ${imgsz} \
--batch-size ${bs} \
--optimizer ${optimizer} \
--cos-lr \
--sync-bn 