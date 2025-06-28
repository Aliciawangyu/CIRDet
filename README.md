## TCSVT2025 - Confidence-driven Unimodal Interference Removal for Enhanced Multimodal Object Detection (CIRDet)

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

  ## Reproducing results

1. Dataset preparation

In the paper, we use 3 benchmarks: alighed FLIR, LLVIP, VEDAI.

-[FLIR] [download](https://pan.xunlei.com/s/VOTqSh6QENsCYsDENKWr1Tb1A1?pwd=i8r7)

-[LLVIP] [download]()

-[VEDAI] [download]()

2. Download the pretrained weights

yolov5 weights (pre-train)

-[yolov5l] []

CIRDet weights

-[FLIR] [xunlei](https://pan.xunlei.com/s/VOTpKpYrjNCVYVS8cabIKRwNA1?pwd=uxzb)

-[LLVIP] [xunlei](https://pan.xunlei.com/s/VOTpKyOfIfJBBF8bFq52XKUSA1?pwd=2t4x)

-[VEDAI] [xunlei](https://pan.xunlei.com/s/VOTpL3QrvkmlrY_kRTMfRIzwA1?pwd=x8ax)

3. Evaluate the model

