# A Sanity Check for AI-generated Image Detection

Official implementation of ['A Sanity Check for AI-generated Image Detection']().

<!-- <div align="center">
<h1>
<b>
A Sanity Check for AI-generated Image Detection
</b>
</h1>
</div> -->

## Introduction

We conduct a sanity check on **"whether the task of AI-generated image detection has been solved"**. To start with, we present **Chameleon** dataset, consisting AI-generated images that are genuinely challenging for human perception. To quantify the generalization of existing methods, we evaluate 9 off-the-shelf AI-generated image detectors on **Chameleon** dataset. Upon analysis, almost all models classify AI-generated images as real ones. Later, we propose **AIDE**~(**A**I-generated **I**mage **DE**tector with Hybrid Features), which leverages multiple experts to simultaneously extract visual artifacts and noise patterns. 

<p align="center"><img src="docs/network.png" width="800"/></p>

## Update
* **TODO**: Release the Chamelon dataset 📌.
* We release the code and checkpoints of AIDE 🔥.

## Requirements

We test the codes in the following environments, other versions may also be compatible:

- CUDA 11.8
- Python 3.10
- Pytorch 2.0.1


## Installation


## Model Zoo and Results


## Acknowledgement

This repo is based on [ConvNeXt](https://github.com/facebookresearch/ConvNeXt-V2). We also refer to the repositories [CNNSpot](https://github.com/peterwang512/CNNDetection)、[AIGCDetectBenchmark](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file) and [DNF](https://github.com/YichiCS/DNF). Thanks for their wonderful works.


## Contact
If you have any question about this project, please feel free to contact tattoo.ysl@gmail.com.
