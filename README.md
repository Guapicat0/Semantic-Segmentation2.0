<div align="center">

# SSF - Semantic Segmentation Framework
**A Useful and Convenient Image Semantic Segmentation Framework**
</div>


## Brief Introduction

this code is about semantic segmentation framework,we collect some common Image segmentation work,like U-Net,pspnet,FPENet,SegNet etc. you can find them in ./nets/SSeg

this work is based on the original U-Net framework 🏠 [project page](https://github.com/bubbliiiing/unet-pytorch) 

## 🎈 How to use

## Contents <!-- omit in toc -->

- [Dataset](#dataset)
- [EFPENet Weights](#EFPENet)
- [Install](#install)
- [Evaluation](#evaluation)
- [Training](#training)
- [Licenses](#licenses)
- [Acknowledgement](#acknowledgement)

## Dataset

We present the [Dataset for Steel Surface Defect Segmentation Dataset](https://pan.baidu.com/s/1YmglHGPc_G_FTTNe4MgpXg), key-word: ejgd 
, which is a dataset for Steel Surface Defect Segmentation Dataset. We colloct 3 common steel surface defect class ：Patch(Pa) , Inclusion(IN）, Scratch(Sc）
download the dataset and place them at ./datasets/NEU-Seg


## EFPENet Weights

We release EPFENet model weights on [web](https://pan.baidu.com/s/1f2xWxixTbPpY5kSCcyx2ww) key-word: v8bz, which is trained on NEU-Seg dataset.

You can place the downloaded weight at ./weight 

## Install

Install SSF
```bash
git clone https://github.com/Guapicat0/Semantic-Segmentation2.0

cd Semantic-Segmentation2.0
# Creating conda environment
conda create -n SSeg python=3.7
conda activate SSeg

# Installing dependencies
pip install -r requirements.txt

```


## Evaluation

Make sure you have downloaded the weight and dataset. when you eval, please modification the ./utils/utils_predict.py  and ./predict

1. Modify the config1

in  ./utils/utils_predict.py , note that ：_defaults {model_path,num_classes,SSeg etc.} in  Image Segmentation，num_classes : class +1

if you wan to ues EFPENet, please modify the SSeg ：EFPENet

2. Modify the config2

Modification : name_classes :["background","CLASS1"]

3. Run the code

```bash
cd Semantic-Segmentation2.0
python3 predict.py

```




## Semantic Segmentation Training

1. modify the config

The parameters for running are all included in the `./train` file, similar to a config file. You only need to modify common hyperparameters such as `num_classes`, `SSeg`, and `dataset_path`.


2. The way of train

if you wan to fine-tune or Pre-train，please notice the config parameters about pretrained：{bool}，model_path

3. Training

```bash
cd Semantic-Segmentation2.0
python3 train.py

```

## Licenses


**Usage and License Notices**: The data, code, and checkpoint are intended and licensed for research use only.  The dataset and models trained using the dataset should not be used outside of research purposes.


## Acknowledgement

- [unet-pytorch](https://github.com/bubbliiiing/unet-pytorch): the codebase we built upon.


## Citation

If you find our model/code/data helpful, please  📝  star us ⭐️！

