# SpineTwin

Achieves 2D/3D consistent keypoint detection. Our architecture is as follows:

![Architecture Diagram](insert_your_image_path_here)

## 🚀 Environment Requirements

The following environment has been tested and is supported:

- `python` == 3.8
- `pytorch` == 2.1.1
- `cuda` == 11.8
- `cudnn` == 8.7.0

## 📁 Data Preparation

To train the model on your own data, please organize your dataset as follows:

```text
data/
└── spine1k/
    ├── ct/
    │   ├── imagesTr/
    │   │   ├── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 16 0000.nii.gz
    │   │   └── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 17 0000.nii.gz
    │   ├── labels/
    │   │   ├── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 16.nii.gz
    │   │   └── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 17.nii.gz
    │   └── shift/
    │       ├── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 16.txt
    │       └── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 17.txt
    ├── drr_heatmap/
    │   ├── ap/
    │   │   ├── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 16.pt
    │   │   └── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 17.pt
    │   └── lat/
    │       ├── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 16.pt
    │       └── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 17.pt
    ├── drr_imgs/
    │   ├── ap/
    │   │   ├── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 16.jpg
    │   │   └── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 17.jpg
    │   └── lat/
    │       ├── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 16.jpg
    │       └── 1.3.6.1.4.1.9328.50.4.0001 ct vertebra 17.jpg
    ├── train_2d.txt
    ├── train_3d.txt
    ├── val_2d.txt
    └── val_3d.txt
```
## 🚀 Training
Train (fusion)
```
bash train_spine1k.sh
```
Train (RL)
```
bash train_spine1k_rl.sh
```
## 👏 Acknowledgements
Our codebase is built upon [MedCoSS](https://github.com/yeerwen/MedCoSS).
