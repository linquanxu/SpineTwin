# SpineTwin

To train the model on your own data, please organize your dataset as follows:
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
