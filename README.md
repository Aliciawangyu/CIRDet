# TCSVT2025 - Confidence-driven Unimodal Interference Removal for Enhanced Multimodal Object Detection (CIRDet)

Paper Links: [*TCSVT Early Access*](https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=4358651&sortType=vol-only-newest&searchWithin=Confidence-driven)

  ## Requirements

1. Set up the environment:

```
conda create -n cirdet python=3.8
conda activate cirdet
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=[your CUDA version] -c pytorch
```

2. Clone the repo
   
```
git clone https://github.com/Aliciawangyu/CIRDet.git
```

3. Install requirements

```
cd CIRDet
pip install -r requirements.txt
```

  ## Dataset and pretrained model

1. Dataset preparation

    In the paper, we use 3 benchmarks: alighed FLIR, LLVIP, VEDAI.

   -[FLIR] [download](https://pan.xunlei.com/s/VOTzGQd_KOH5j3zDaaCgV87gA1?pwd=b5zv)

   -[LLVIP] [download](https://pan.xunlei.com/s/VOTzGbJtRILz3bf1Doc8I2rrA1?pwd=fhsx)

   -[VEDAI] [download](https://pan.xunlei.com/s/VOTzGY2UMce2SLPdEGNykFcLA1?pwd=briv)

2. Download the pretrained weights

   yolov5 weights (pre-train)

   -[yolov5l] [xunlei](https://pan.xunlei.com/s/VOTzO_qWJMhaCe7z_SV0BbxJA1?pwd=f3nf)

   CIRDet weights

   -[FLIR] [xunlei](https://pan.xunlei.com/s/VOTpKpYrjNCVYVS8cabIKRwNA1?pwd=uxzb)

   -[LLVIP] [xunlei](https://pan.xunlei.com/s/VOTpKyOfIfJBBF8bFq52XKUSA1?pwd=2t4x)

   -[VEDAI] [xunlei](https://pan.xunlei.com/s/VOTpL3QrvkmlrY_kRTMfRIzwA1?pwd=x8ax)

  ## Training and Evaluation
  
1. Training script.
```
cd script/train
sh FLIR.sh
```
2. Evaluation script.
```
cd script/val
sh FLIR.sh
```

