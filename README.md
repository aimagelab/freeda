# Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation (CVPR 2024)

[Project Page](https://aimagelab.github.io/freeda/) | [Paper](https://arxiv.org/abs/2404.06542)

<div align="center">
<figure>
  <img alt="Qualitative results" src="./assets/qualitatives1.png">
</figure>
</div>

## Method

<div align="center">
<figure>
  <img alt="FreeDA method" src="./assets/inference.png">
</figure>
</div>

<br/>

<details>
<summary> Additional qualitative examples </summary>
<p align="center">
  <img alt="Additional qualitative results" src="./assets/qualitatives.png" width="800" />
</p>
</details>

<details>
<summary> Additional examples <i>in-the-wild</i> </summary>
<p align="center">
  <img alt="in-the-wild examples" src="./assets/into_the_wild.png" width="800" />
</p>
</details>

## Setup

Our setup is based on pytorch 1.13.1, mmcv 1.6.2 and mmsegmentation 0.27.0. To create the same environment that we used for our experiments:

```bash
python3 -m venv ./freeda
source ./freeda/bin/activate
pip install -U pip setuptools wheel
```

Install PyTorch:

```bash 
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Install other dependencies:

```bash
pip install -r requirements.txt
```

Download both the [prototype embeddings](https://drive.google.com/file/d/1U4d0exJuq29b0rLR6iOT20ErW3DAmgw0/view?usp=sharing) and the [faiss index](https://drive.google.com/file/d/1FHjpM0aqPf9OjiuG_341EMlEuq6hsh6L/view?usp=sharing), and decompress them
into `./data`:

```bash
cd ./data
mkdir "prototype_embeddings"
tar -xvzf prototype_embeddings.tar -C ./prototype_embeddings
unzip faiss_index.zip
```

## Datasets

This section is adapted from [TCL](https://github.com/kakaobrain/tcl) and [GroupViT](https://github.com/NVlabs/GroupViT#data-preparation) README.

The overall file structure is as follows:

```shell
src
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   ├── VOC2010
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClassContext
│   │   │   ├── ImageSets
│   │   │   │   ├── SegmentationContext
│   │   │   │   │   ├── train.txt
│   │   │   │   │   ├── val.txt
│   │   │   ├── trainval_merged.json
│   │   ├── VOCaug
│   │   │   ├── dataset
│   │   │   │   ├── cls
│   ├── ade
│   │   ├── ADEChallengeData2016
│   │   │   ├── annotations
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   │   │   ├── images
│   │   │   │   ├── training
│   │   │   │   ├── validation
│   ├── coco_stuff164k
│   │   ├── images
│   │   │   ├── train2017
│   │   │   ├── val2017
│   │   ├── annotations
│   │   │   ├── train2017
│   │   │   ├── val2017
```

Please download and setup [PASCAL VOC](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-voc)
, [PASCAL Context](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#pascal-context), [COCO-Stuff164k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#coco-stuff-164k)
, [Cityscapes](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#cityscapes), and [ADE20k](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#ade20k) datasets
following [MMSegmentation data preparation document](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md).

## Evaluation

Pascal VOC:

```
python -m torch.distributed.run main.py --eval --eval_cfg configs/pascal20/freeda_pascal20.yml --eval_base_cfg configs/pascal20/eval_pascal20.yml
```

Pascal Context:

```
python -m torch.distributed.run main.py --eval --eval_cfg configs/pascal59/freeda_pascal59.yml --eval_base_cfg configs/pascal59/eval_pascal59.yml
```

COCO-Stuff:

```
python -m torch.distributed.run main.py --eval --eval_cfg configs/cocostuff/freeda_cocostuff.yml --eval_base_cfg configs/cocostuff/eval_cocostuff.yml
```

Cityscapes:

```
python -m torch.distributed.run main.py --eval --eval_cfg configs/cityscapes/freeda_cityscapes.yml --eval_base_cfg configs/cityscapes/eval_cityscapes.yml
```

ADE20K:

```
python -m torch.distributed.run main.py --eval --eval_cfg configs/ade/freeda_ade.yml --eval_base_cfg configs/ade/eval_ade.yml
```

If you find FreeDA useful for your work please cite:
```
@inproceedings{barsellotti2024training
  title={Training-Free Open-Vocabulary Segmentation with Offline Diffusion-Augmented Prototype Generation},
  author={Barsellotti, Luca and Amoroso, Roberto and Cornia, Marcella and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```