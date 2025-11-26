# TCSVT2025 - Confidence-driven Unimodal Interference Removal for Enhanced Multimodal Object Detection (CIRDet)

Paper Links: [*CIRDet*]([https://ieeexplore.ieee.org/xpl/tocresult.jsp?isnumber=4358651&sortType=vol-only-newest&searchWithin=Confidence-driven](https://ieeexplore.ieee.org/document/11029233))

## Introduction

**Motivation of CIRDet**

<div align="left">
   <img src="https://github.com/Aliciawangyu/CIRDet/blob/main/img/Motivation.png" width="60%">
</div>

Infrared and visible modalities exhibit different levels of confidence under various conditions. Crossmodal cooperation is effective only when both modalities maintain the same confidence level and convey similar semantic information (e.g., “car”). We refer to these as consensus features. However, in scenarios where one modality is inferior, the descriptions of the same object may diverge (e.g., “pedestrian”), leading to conflict features. Aggregating information without considering the effectiveness of each modality may introduce noise and diminish the contribution of the superior modality.

## Architecture
**Overview of CIRDet**

<div align="center">
   <img src="https://github.com/Aliciawangyu/CIRDet/blob/main/img/Network.png" width="100%">
</div>

Overview of the proposed CIRDet for object detection. The input images are first processed by two-stream encoders to extract feature maps at three different scales. Then, UPD is used to explicitly decompose each modal features into two parts: consensus features, which are constrained by KL divergence in ECF, and conflict features. Besides, ECF performs an equitable credibility fusion of the consensus features from both modalities. Finally, we employ GCA and LCA to evaluate the confidence levels of conflict features and remove the unreliable modal interference. The fused features from three different scales are then fed into the subsequent networks for detection.

## Results

|Dataset|mAP@0.5|mAP@0.75|mAP|
|:---------:|:-----:|:-----------------:|:-------------:|
|FLIR|81.2(-1.0)|45.5(+8.1)|46.5(+4.2)|
|VEDAI|86.5(+1.2)|70.9(+5.0)|59.1(+3.1)|
|LLVIP|98.0(+0.5)|78.4(+5.2)|67.3(+3.4)|

## Citation
If you use this repo for your research, please cite our paper:
```
@ARTICLE{11029233,
  author={Wang, Yu and Wei, Shikui and Xu, Sen and Qin, Ying and Zhao, Yao},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Confidence-driven Unimodal Interference Removal for Enhanced Multimodal Object Detection}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Interference;Object detection;Visualization;Transformers;Noise;Streaming media;Detection algorithms;Training;Reliability;multi-modal fusion;multi-spectral object detection;unimodal interference;disentangled representation},
  doi={10.1109/TCSVT.2025.3578340}}

```

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

## Thanks to the Third Party Libs
https://docs.ultralytics.com/zh/yolov5/

