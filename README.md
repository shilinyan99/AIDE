<div align="center">
<br>
<h3>A Sanity Check for AI-generated Image Detection</h3>


<sup>1</sup>Xiaohongshu Inc. <sup>2</sup>University of Science and Technology of China <sup>3</sup>Shanghai Jiao Tong University


<p align="center">
  <a href='https://shilinyan99.github.io/AIDE'>
    <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'>
  </a>
  <a href='https://arxiv.org/abs/2406.19435'>
    <img src='https://img.shields.io/badge/Arxiv-2406.19435-A42C25?style=flat&logo=arXiv&logoColor=A42C25'>
  </a>
  <a href='https://arxiv.org/pdf/2406.19435'>
    <img src='https://img.shields.io/badge/Paper-PDF-yellow?style=flat&logo=arXiv&logoColor=yellow'>
  </a>
</p>
</div>


<!-- <div align="center">
<h1>
<b>
A Sanity Check for AI-generated Image Detection
</b>
</h1>
</div> -->
## 🔥 News
* [2024-12-29]🔥🔥🔥 We release the Chamelon dataset.
* [2024-06-20]🔥🔥🔥 We release the code and checkpoints of AIDE.


## 🔍 Chameleon 

**Comparison of `Chameleon` with existing benchmarks.**

<p align="center"><img src="docs/Chameleon.jpg" width="800"/></p>

We visualize two contemporary AI-generated image benchmarks, namely:

- **(a) AIGCDetect Benchmark** 
- **(b) GenImage Benchmark** 

where all images are generated from publicly available generators, such as ProGAN (GAN-based), SD v1.4 (DM-based), and Midjourney (commercial API). These images are generated by unconditional situations or conditioned on simple prompts (e.g., *photo of a plane*) without delicate manual adjustments, thereby inclined to generate obvious artifacts in consistency and semantics (marked with <span style="color:red">red boxes</span>).

In contrast, our **`Chameleon`** dataset in **(c)** aims to simulate real-world scenarios by collecting diverse images from online websites, where these online images are carefully adjusted by photographers and AI artists.


**License**:
```
Chameleon is only used for academic research. Commercial use in any form is prohibited.
```

🌟🌟🌟 If you need the Chameleon dataset, please send an email to **tattoo.ysl@gmail.com**. 🌟


## 👀 Overview

We conduct a sanity check on **"whether the task of AI-generated image detection has been solved"**. To start with, we present **Chameleon** dataset, consisting AI-generated images that are genuinely challenging for human perception. To quantify the generalization of existing methods, we evaluate 9 off-the-shelf AI-generated image detectors on **Chameleon** dataset. Upon analysis, almost all models classify AI-generated images as real ones. Later, we propose **AIDE**~(**A**I-generated **I**mage **DE**tector with Hybrid Features), which leverages multiple experts to simultaneously extract visual artifacts and noise patterns. 

<p align="center"><img src="docs/network.png" width="800"/></p>




## Requirements

We test the codes in the following environments, other versions may also be compatible:

- CUDA 11.8
- Python 3.10
- Pytorch 2.0.1


## Setup

First, clone the repository locally.

```
https://github.com/shilinyan99/AIDE
```

Then, install Pytorch 2.0.1 using the conda environment.
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 -c pytorch
```

Lastly, install the necessary packages and pycocotools.

```
pip install -r requirements.txt 
```


## Get Started

### Training

```
./scripts/train.sh  --data_path [/path/to/train_data] --eval_data_path [/path/to/eval_data] --resnet_path [/path/to/pretrained_resnet_path] --convnext_path [/path/to/pretrained_convnext_path] --output_dir [/path/to/output_dir] [other args]
```

For example, training on ProGAN, run the following command:

```
./scripts/train.sh --data_path dataset/progan/train --eval_data_path dataset/progan/eval --resnet_path pretrained_ckpts/resnet50.pth --convnext_path pretrained_ckpts/open_clip_pytorch_model.bin --output_dir results/progan_train
```

### Inference

Inference using the trained model.
```
./scripts/eval.sh --data_path [/path/to/train_data] --eval_data_path [/path/to/eval_data] --resume [/path/to/progan_train] --eval True --output_dir [/path/to/output_dir]
```

For example, evaluating the progan_train model, run the following command:

```
./scripts/eval.sh --data_path dataset/progan/train --eval_data_path dataset/progan/eval --resume results/progan_train/progan_train.pth --eval True --output_dir results/progan_train
```



## Dataset

### Training Set
We adopt the training set in [CNNSpot](https://github.com/peterwang512/CNNDetection) and [GenImage](https://github.com/Andrew-Zhu/GenImage).

### Test Set
The whole test set we used in our experiments can be downloaded from [AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file) and [GenImage](https://github.com/Andrew-Zhu/GenImage).


## Model Zoo

Our training checkpoints can be downloaded from [link](https://drive.google.com/drive/folders/1qvUz0MgrVwG1B1ntkUVcRuYY0864jqcy?usp=sharing).

## Acknowledgement

This repo is based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt-V2). We also refer to the repositories [CNNSpot](https://github.com/peterwang512/CNNDetection)、[AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file)、[GenImage](https://github.com/Andrew-Zhu/GenImage) and [DNF](https://github.com/YichiCS/DNF). Thanks for their wonderful works.

## Citation

```
@article{yan2024sanity,
  title={A Sanity Check for AI-generated Image Detection},
  author={Yan, Shilin and Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal={arXiv preprint arXiv:2406.19435},
  year={2024}
}
```

## Contact
If you have any question about this project, please feel free to contact tattoo.ysl@gmail.com.
