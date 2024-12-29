<div align="center">
<br>
<h3>A Sanity Check for AI-generated Image Detection</h3>

[Shilin Yan](https://scholar.google.com/citations?user=2VhjOykAAAAJ&hl=zh-CN&oi=ao)<sup>1†</sup>, Ouxiang Li<sup>1,2†</sup>, Jiayin Cai<sup>1†</sup>, [Xiaolong Jiang](https://scholar.google.com/citations?user=G0Ow8j8AAAAJ&hl=en&oi=ao)<sup>1</sup>, [Yao Hu](https://scholar.google.com/citations?user=LIu7k7wAAAAJ&hl=en)<sup>1</sup>, [Weidi Xie](https://scholar.google.com/citations?user=Vtrqj4gAAAAJ&hl=en)<sup>3‡</sup>

<div class="is-size-6 publication-authors">
  <p class="footnote">
    <span class="footnote-symbol"><sup>†</sup></span>Equal contribution
    <span class="footnote-symbol"><sup>‡</sup></span>Corresponding author
  </p>
</div>

<sup>1</sup>Xiaohongshu Inc. <sup>2</sup>University of Science and Technology of China <sup>3</sup>Shanghai Jiao Tong University


<p align="center">
  <a href='https://shilinyan99.github.io/AIDE'>
    <img src='https://img.shields.io/badge/Project-Page-pink?style=flat&logo=Google%20chrome&logoColor=pink'>
  </a>
  <a href='https://arxiv.org/pdf/2406.19435'>
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

- CUDA 11.3
- Python 3.10
- Pytorch 1.11.0


## Installation

Please refer to [install.md](docs/install.md) for installation.

## Get Started

Please see [Training.md](docs/Training.md) for details.


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
