import os
import math
import cv2
import torch
import numpy as np
import SimpleITK as sitk
from torch.utils import data
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.transforms import (
    Compose, Rand3DElasticD, RandAffineD, RandGaussianNoiseD, RandGaussianSmoothD,
    RandAdjustContrastD, RandShiftIntensityD, RandScaleIntensityD,
    RandZoomD, RandFlipD, ToTensorD
)


def safe_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.clone().detach()
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        raise TypeError(f"Unsupported type: {type(x)}")

def pad_image(img, target_size):
    """Pad an image up to the target size."""
    if img.ndim == 4: # (C, D, H, W)
        c, d, h, w = img.shape
        target_d, target_h, target_w = target_size
        d_missing = max(target_d - d, 0)
        h_missing = max(target_h - h, 0)
        w_missing = max(target_w - w, 0)
        padding = ((0, 0), (0, d_missing), (0, h_missing), (0, w_missing))
        padded_img = np.pad(img, padding, mode='constant', constant_values=0)
    elif img.ndim == 3: # (D, H, W)
        d, h, w = img.shape
        # Handle case where target_size might be (D, H, W)
        rows_missing = math.ceil(target_size[1] - h)
        cols_missing = math.ceil(target_size[2] - w)
        dept_missing = math.ceil(target_size[0] - d)
        
        # Ensure non-negative
        rows_missing = max(rows_missing, 0)
        cols_missing = max(cols_missing, 0)
        dept_missing = max(dept_missing, 0)

        padded_img = np.pad(img, ((0, dept_missing), (0, rows_missing), (0, cols_missing)), 'constant')
    else:
        # Fallback for other shapes if necessary, keeping original logic structure
        padded_img = img 
    return padded_img

def truncate_ct(CT):
    min_HU = -325
    max_HU = 1500
    subtract = 0
    divide = 1500
    CT[np.where(CT <= min_HU)] = min_HU
    CT[np.where(CT >= max_HU)] = max_HU
    CT = CT - subtract
    CT = CT / divide
    return CT



class MOTS_Dataset_train_2d3d(data.Dataset):
    def __init__(self, root, list_path, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255, ratio_labels=1,
                 root_2d=None, list_path_2d=None, crop_size_2d=(256, 256), scale_2d=True,
                 mirror_2d=True, ignore_label_2d=255, ratio_labels_2d=1, split_2d="train", 
                 num_classes=5, mode='joint'):
        
        self.mode = mode  # Critical: Store the mode
        self.num_classes = num_classes
        
        # 3D Configs
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size # (D, H, W)
        self.crop_d, self.crop_h, self.crop_w = crop_size
        
        # 2D Configs
        self.root_2d = root_2d
        self.list_path_2d = list_path_2d
        self.crop_size_2d = crop_size_2d

        # --- Conditional Loading Logic ---
        self.files_2d_ap = []
        self.files_2d_lat = []
        self.files_3d = []

        # 1. Load 2D Data Lists
        if self.mode in ['only_2d_ap', 'only_2d_lat', 'joint', 'fusion']:
            if root_2d and list_path_2d:
                data_lines = open(self.list_path_2d, "r").readlines()
                for name in data_lines:
                    name = name.strip()
                    # Only add AP if needed
                    if self.mode in ['only_2d_ap', 'joint', 'fusion']:
                        self.files_2d_ap.append({
                            "image": os.path.join(self.root_2d, "drr_imgs/ap", name),
                            "name": name,
                            "label": os.path.join(self.root_2d, "drr_heatmap/ap", name[:-4]+'.pt')
                        })
                    # Only add Lat if needed
                    if self.mode in ['only_2d_lat', 'joint', 'fusion']:
                        self.files_2d_lat.append({
                            "image": os.path.join(self.root_2d, "drr_imgs/lat", name),
                            "name": name,
                            "label": os.path.join(self.root_2d, "drr_heatmap/lat", name[:-4]+'.pt')
                        })
                
                if len(self.files_2d_ap) > 0:
                    print(f'[{self.mode}] {len(self.files_2d_ap)} 2D AP images loaded!')
                if len(self.files_2d_lat) > 0:
                    print(f'[{self.mode}] {len(self.files_2d_lat)} 2D Lat images loaded!')

        # 2. Load 3D Data Lists
        self.dict_3d = {} 
        if self.mode in ['only_3d', 'joint', 'fusion']:
            if list_path:
                self.img_ids = [i_id.strip().split() for i_id in open(self.list_path)]
                for item in self.img_ids:
                    self.files_3d.append({"image": item})

                    name_without_ext = item[0].replace('.nii.gz', '') 
                    key = name_without_ext.rsplit('_', 1)[0]
                    self.dict_3d[key] = item[0]
                print(f'[{self.mode}] {len(self.files_3d)} 3D volumes loaded!')

           
           

    def __len__(self):
        # Return length based on primary modality
        if self.mode == 'only_3d':
            return len(self.files_3d)
        # For joint/fusion or 2D modes, we usually align with 2D length or the shorter one
        # Assuming 2D AP list is the anchor for 2D/Joint training
        if len(self.files_2d_ap) > 0:
            return len(self.files_2d_ap)
        elif len(self.files_2d_lat) > 0:
            return len(self.files_2d_lat)
        return len(self.files_3d)

    def __getitem__(self, index):
        result = {
            "name": "unknown", 
            "num_classes": self.num_classes,
            "ct_oir_size": np.array([0, 0, 0]),
            "valid_shape": np.array([self.crop_d, self.crop_h, self.crop_w], dtype=np.int32),
            "prop": {},
        }

        # --- Load 2D AP ---
        if self.mode in ['only_2d_ap', 'joint', 'fusion'] and len(self.files_2d_ap) > 0:
            datafiles = self.files_2d_ap[index]
            result["name"] = datafiles["name"]
            
            img = cv2.imread(datafiles["image"])
            img = img.transpose(2, 0, 1).astype(np.float32) # (3, H, W)
            lbl = torch.load(datafiles["label"]) # (C, H, W)
            
            lbl = torch.from_numpy(lbl)
            lbl = torch.flip(lbl, dims=[1]).numpy()
            img = torch.from_numpy(img)
            img = torch.flip(img, dims=[1]).numpy()
            
            result["image2d_ap"] = img
            result["label2d_ap"] = lbl

        # --- Load 2D Lat ---
        if self.mode in ['only_2d_lat', 'joint', 'fusion'] and len(self.files_2d_lat) > 0:
            datafiles = self.files_2d_lat[index]
            if result["name"] == "unknown": result["name"] = datafiles["name"]

            img = cv2.imread(datafiles["image"])
            img = img.transpose(2, 0, 1).astype(np.float32)
            lbl = torch.load(datafiles["label"])
            
            lbl = torch.from_numpy(lbl)
            lbl = torch.flip(lbl, dims=[1]).numpy()
            img = torch.from_numpy(img)
            img = torch.flip(img, dims=[1]).numpy()
            
            result["image2d_lat"] = img
            result["label2d_lat"] = lbl

        # --- Load 3D ---
        if self.mode in ['only_3d', 'joint', 'fusion'] and len(self.files_3d) > 0:
            # Handle index mismatch if 3D dataset size != 2D dataset size
            # idx_3d = index % len(self.files_3d)
            # datafiles = self.files_3d[idx_3d]
            
            # image_name = datafiles["image"][0]
            # print(image_name)
            if self.mode in ['joint', 'fusion']:
                key_name = result["name"].replace('_ap.png', '').replace('_lat.png', '').replace('.png', '').replace('.jpg', '')
                
                if key_name in self.dict_3d:
                    image_name = self.dict_3d[key_name]
                    shift_file = os.path.join(self.root, 'shift', image_name[:-12] + '.txt')
                    shift_val = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                    if os.path.exists(shift_file):
                        with open(shift_file, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                shift_val = np.array([float(x) for x in line.split()], dtype=np.float32)

                    result['prop']['shift'] = shift_val
                    
                else:
                    print(f"Warning: No matching CT for {result['name']}")
                    image_name = list(self.dict_3d.values())[0] 
            else:
                image_name = self.files_3d[index]["image"][0]
            
            # print(image_name)


            label_name = image_name[:-12] + '.nii.gz'
            
            ipath = os.path.join(self.root, 'imagesTr', image_name)
            lpath = os.path.join(self.root, 'labelsTr', label_name)
            
            ct_sitk = sitk.ReadImage(ipath)
            result['prop']['ori_size'] = np.array(ct_sitk.GetSize(), dtype=np.int32)
            result['prop']['ori_space'] = np.array(ct_sitk.GetSpacing(), dtype=np.float32)
            result['prop']['direction'] = ct_sitk.GetDirection()
            result['prop']['origin'] = ct_sitk.GetOrigin()

            image = sitk.GetArrayFromImage(ct_sitk) # (D, H, W)
            
            label_sitk = sitk.ReadImage(lpath)
            label = sitk.GetArrayFromImage(label_sitk)
            od, oh, ow = image.shape
            result["ct_oir_size"] = np.array([od, oh, ow])
            # Preprocessing
            image = pad_image(image, [self.crop_d, self.crop_h, self.crop_w])
            label = pad_image(label, [self.crop_d, self.crop_h, self.crop_w])
            
            image = truncate_ct(image)

            if self.mode in ['fusion']:
                valid_d = min(od, self.crop_d)
                valid_h = min(oh, self.crop_h)
                valid_w = min(ow, self.crop_w)
                result["valid_shape"] = np.array([valid_d, valid_h, valid_w], dtype=np.int32)

            image = image[np.newaxis, :].astype(np.float32) # (1, D, H, W)
            label = label.astype(np.float32) # (C, D, H, W) or (D, H, W) depending on your label

            result["image3d"] = image
            result["label3d"] = label

        result["mode"] = self.mode

        return result



class MOTS_Dataset_test_2d3d(data.Dataset):
    def __init__(self, root, list_path, crop_size=(64, 192, 192), 
                 root_2d=None, list_path_2d=None, crop_size_2d=(256, 256), 
                 num_classes=5, mode='joint'):
        
        self.mode = mode
        self.num_classes = num_classes
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.crop_d, self.crop_h, self.crop_w = crop_size
        self.root_2d = root_2d
        self.list_path_2d = list_path_2d

        self.files_2d_ap = []
        self.files_2d_lat = []
        self.files_3d = []

        # 1. Load 2D
        if self.mode in ['only_2d_ap', 'only_2d_lat', 'joint', 'fusion']:
            data_lines = open(self.list_path_2d, "r").readlines()
            for name in data_lines:
                name = name.strip()
                if self.mode in ['only_2d_ap', 'joint', 'fusion']:
                    self.files_2d_ap.append({
                        "image": os.path.join(self.root_2d, "drr_imgs/ap", name),
                        "name": name,
                        "label": os.path.join(self.root_2d, "drr_heatmap/ap", name[:-4]+'.pt')
                    })
                if self.mode in ['only_2d_lat', 'joint', 'fusion']:
                    self.files_2d_lat.append({
                        "image": os.path.join(self.root_2d, "drr_imgs/lat", name),
                        "name": name,
                        "label": os.path.join(self.root_2d, "drr_heatmap/lat", name[:-4]+'.pt')
                    })

        # 2. Load 3D
        self.dict_3d = {} 
        if self.mode in ['only_3d', 'joint', 'fusion']:
            self.img_ids = [i_id.strip().split() for i_id in open(self.list_path)]
            for item in self.img_ids:
                self.files_3d.append({"image": item})
                name_without_ext = item[0].replace('.nii.gz', '') 
                key = name_without_ext.rsplit('_', 1)[0]
                self.dict_3d[key] = item[0]

    def __len__(self):
        
        if self.mode == 'only_3d':
            return len(self.files_3d)
        if len(self.files_2d_ap) > 0:
            return len(self.files_2d_ap)
        elif len(self.files_2d_lat) > 0:
            return len(self.files_2d_lat)
        return len(self.files_3d)

    def __getitem__(self, index):
        result = {
            "name": "unknown", 
            "num_classes": self.num_classes,
            "ct_oir_size": np.array([0, 0, 0]),
            "prop": {}
        }

        # Load AP
        if self.mode in ['only_2d_ap', 'joint', 'fusion'] and len(self.files_2d_ap) > 0:
            datafiles = self.files_2d_ap[index]
            result["name"] = datafiles["name"]
            img = cv2.imread(datafiles["image"]).transpose(2, 0, 1).astype(np.float32)
            lbl = torch.load(datafiles["label"])
            
            lbl = torch.from_numpy(lbl)
            lbl = torch.flip(lbl, dims=[1]).numpy()
            img = torch.from_numpy(img)
            img = torch.flip(img, dims=[1]).numpy()
            
            result["image2d_ap"] = img
            result["label2d_ap"] = lbl

        # Load Lat
        if self.mode in ['only_2d_lat', 'joint', 'fusion'] and len(self.files_2d_lat) > 0:
            datafiles = self.files_2d_lat[index]
            if result["name"] == "unknown": result["name"] = datafiles["name"]
            img = cv2.imread(datafiles["image"]).transpose(2, 0, 1).astype(np.float32)
            lbl = torch.load(datafiles["label"])
            
            lbl = torch.from_numpy(lbl)
            lbl = torch.flip(lbl, dims=[1]).numpy()
            img = torch.from_numpy(img)
            img = torch.flip(img, dims=[1]).numpy()
            
            result["image2d_lat"] = img
            result["label2d_lat"] = lbl

        # Load 3D
        if self.mode in ['only_3d', 'joint', 'fusion'] and len(self.files_3d) > 0:
            if self.mode in ['joint', 'fusion']:
                key_name = result["name"].replace('_ap.png', '').replace('_lat.png', '').replace('.png', '').replace('.jpg', '')
                
                if key_name in self.dict_3d:
                    image_name = self.dict_3d[key_name]
                    shift_file = os.path.join(self.root, 'shift', image_name[:-12] + '.txt')
                    shift_val = np.array([0.0, 0.0, 0.0], dtype=np.float32)

                    if os.path.exists(shift_file):
                        with open(shift_file, 'r') as f:
                            line = f.readline().strip()
                            if line:
                                shift_val = np.array([float(x) for x in line.split()], dtype=np.float32)

                    result['prop']['shift'] = shift_val
                else:
                    print(f"Warning: No matching CT for {result['name']}")
                    image_name = list(self.dict_3d.values())[0] 
            else:
                idx_3d = index % len(self.files_3d)
                image_name = self.files_3d[idx_3d]["image"][0]
                
                shift_file = os.path.join(self.root, 'shift', image_name[:-12] + '.txt')
                shift_val = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                if os.path.exists(shift_file):
                    with open(shift_file, 'r') as f:
                        line = f.readline().strip()
                        if line:
                            shift_val = np.array([float(x) for x in line.split()], dtype=np.float32)
                result['prop']['shift'] = shift_val

            label_name = image_name[:-12]+'.nii.gz'
            
            ipath = os.path.join(self.root, 'imagesTr', image_name)
            lpath = os.path.join(self.root, 'labelsTr', label_name)
            
            ct_sitk = sitk.ReadImage(ipath)
            
            # Props
            result['prop']['ori_size'] = np.array(ct_sitk.GetSize(), dtype=np.int32)
            result['prop']['ori_space'] = np.array(ct_sitk.GetSpacing(), dtype=np.float32)
            result['prop']['direction'] = ct_sitk.GetDirection()
            result['prop']['origin'] = ct_sitk.GetOrigin()

            image = sitk.GetArrayFromImage(ct_sitk)
            label_sitk = sitk.ReadImage(lpath)
            label = sitk.GetArrayFromImage(label_sitk)

            od, oh, ow = image.shape
            result["ct_oir_size"] = np.array([od, oh, ow])

            image = pad_image(image, [self.crop_d, self.crop_h, self.crop_w])
            label = pad_image(label, [self.crop_d, self.crop_h, self.crop_w])
            image = truncate_ct(image)

            result["image3d"] = image[np.newaxis, :].astype(np.float32)
            result["label3d"] = label.astype(np.float32)

            valid_d = min(od, self.crop_d)
            valid_h = min(oh, self.crop_h)
            valid_w = min(ow, self.crop_w)
            result["valid_shape"] = np.array([valid_d, valid_h, valid_w], dtype=np.int32)
        
                

        return result
    

class NormalizePerImage(A.ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)

    def apply(self, image, **params):
        image = image.astype('float32')
        for c in range(image.shape[2]):
            mean = image[..., c].mean()
            std = image[..., c].std()
            std = std if std > 1e-6 else 1.0
            image[..., c] = (image[..., c] - mean) / std
        return image

def get_train_transform_2d_albu(patch_size=(256, 256), num_classes=5, mode='joint'):
    # if mode in ['only_2d_ap', 'only_2d_lat']:
    #     geo_transforms = A.Compose([
    #         A.HorizontalFlip(p=0.3),
    #         A.VerticalFlip(p=0.3),
    #         A.Resize(patch_size[0], patch_size[1]),
    #     ])
    # else:
    geo_transforms = A.Compose([
        A.Resize(patch_size[0], patch_size[1]),
    ])
    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=0.2),
        geo_transforms,
        NormalizePerImage(), # Assuming this class is defined as in your original code
        ToTensorV2()
    ], additional_targets={f'mask{i+1}': 'mask' for i in range(num_classes)})
    return transform

def get_train_transform_2d_albu_test(patch_size=(256, 256), num_classes=5):
    transform = A.Compose([
        A.Resize(patch_size[0], patch_size[1]),
        NormalizePerImage(),
        ToTensorV2()
    ], additional_targets={f'mask{i+1}': 'mask' for i in range(num_classes)})
    return transform

def get_train_transform_3d(patch_size, mode='joint'):
    transforms = []
    # if mode == 'only_3d':
    #     transforms.extend([
    #         Rand3DElasticD(
    #             keys=["image", "label"], sigma_range=(8, 10), magnitude_range=(0, 200), prob=0.1,
    #             rotate_range=(np.pi/12, np.pi/12, np.pi/12), scale_range=(0.05, 0.05, 0.05),
    #             mode=("bilinear", "nearest"), padding_mode="reflection"
    #         ),
    #         RandAffineD(
    #             keys=["image", "label"], prob=0.2, rotate_range=(np.pi/12, np.pi/12, np.pi/12),
    #             scale_range=(0.1, 0.1, 0.1), mode=("bilinear", "nearest"), padding_mode="reflection"
    #         ),
    #         RandZoomD(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, mode=("trilinear", "nearest"), 
    #                   padding_mode="edge", prob=0.2),
    #         RandFlipD(keys=["image", "label"], spatial_axis=[0, 1, 2], prob=0.5),
    #     ])
  
    transforms.extend([
        RandGaussianNoiseD(keys="image", prob=0.05, mean=0.0, std=0.01),
        RandGaussianSmoothD(keys="image", prob=0.05, sigma_x=(0.5, 1.0)),
        RandAdjustContrastD(keys="image", prob=0.15, gamma=(0.9, 1.1)),
        RandShiftIntensityD(keys="image", offsets=0.05, prob=0.1),
        RandScaleIntensityD(keys="image", factors=0.1, prob=0.1),
        ToTensorD(keys=["image", "label"])
    ])
    return Compose(transforms)

def call_aug_batch(image_batch, label_batch, name_list, num_classes, is_train=True, mode='joint'):
    """Generic 2D Batch Augmentation"""
    batch_size = len(image_batch)
    image_list, label_list, name_out = [], [], []
    
    if is_train:
        transform = get_train_transform_2d_albu(num_classes=num_classes, mode=mode)
    else:
        transform = get_train_transform_2d_albu_test(num_classes=num_classes)

    for i in range(batch_size):
        image = image_batch[i].transpose(1, 2, 0) # (H, W, 3)
        label = label_batch[i] # (C, H, W)
        masks = [label[j] for j in range(label.shape[0])]

        transform_input = {'image': image}
        for j in range(num_classes):
            transform_input[f'mask{j+1}'] = masks[j]

        augmented = transform(**transform_input)

        image_aug = safe_tensor(augmented['image'])
        label_aug = torch.stack([safe_tensor(augmented[f'mask{j+1}']) for j in range(num_classes)])

        image_list.append(image_aug)
        label_list.append(label_aug)
        name_out.append(name_list[i])

    return {
        'image': torch.stack(image_list),
        'label': torch.stack(label_list),
        'name': name_out
    }



def my_collate(batch):
    batch_out = {
        'r_2d_ap': None, 
        'r_2d_lat': None, 
        'r_3d': None, 
        'name': [], 
        'ct_oir_size': [],
        'valid_shapes': [],
        'prop': []
    }
    current_mode = batch[0]['mode'] 
    has_ap = 'image2d_ap' in batch[0]
    has_lat = 'image2d_lat' in batch[0]
    has_3d = 'image3d' in batch[0]
    num_classes = batch[0]['num_classes']

    imgs_ap, lbls_ap, w_ap = [], [], []
    imgs_lat, lbls_lat, w_lat = [], [], []
    imgs_3d, lbls_3d, w_3d = [], [], []
    
    for item in batch:
        batch_out['name'].append(item['name'])
        batch_out['ct_oir_size'].append(item['ct_oir_size'])
        if 'prop' in item:
            batch_out['prop'].append(item['prop'])

        if 'valid_shape' in item:
            batch_out['valid_shapes'].append(item['valid_shape'])
        
        if has_ap:
            imgs_ap.append(item['image2d_ap'])
            lbls_ap.append(item['label2d_ap'])
        if has_lat:
            imgs_lat.append(item['image2d_lat'])
            lbls_lat.append(item['label2d_lat'])
        if has_3d:
            imgs_3d.append(item['image3d'])
            lbls_3d.append(item['label3d'])

    batch_out['ct_oir_size'] = np.stack(batch_out['ct_oir_size'], 0)
    batch_out['name'] = np.array(batch_out['name'])

    if len(batch_out['valid_shapes']) > 0:
        batch_out['valid_shapes'] = torch.from_numpy(np.stack(batch_out['valid_shapes'], 0))
    else:
        batch_out['valid_shapes'] = None


    if has_ap:
        batch_out['r_2d_ap'] = call_aug_batch(imgs_ap, lbls_ap, batch_out['name'], num_classes, is_train=True, mode=current_mode)


    if has_lat:
        batch_out['r_2d_lat'] = call_aug_batch(imgs_lat, lbls_lat, batch_out['name'], num_classes, is_train=True, mode=current_mode)


    if has_3d:
        img_3d_np = np.stack(imgs_3d, 0) 
        lbl_3d_np = np.stack(lbls_3d, 0) 

        if not hasattr(my_collate, "_tr_3d"):
            patch_size = lbl_3d_np[0].shape[1:] 
            my_collate._tr_3d = get_train_transform_3d(patch_size, mode=current_mode)
            
        batch_aug_imgs = []
        batch_aug_lbls = []
        
        batch_size = img_3d_np.shape[0]
        
        for i in range(batch_size):
            data_dict = {
                "image": img_3d_np[i], # (C, D, H, W)
                "label": lbl_3d_np[i]  # (C, D, H, W)
            }
            
            aug_data = my_collate._tr_3d(data_dict)
            
            batch_aug_imgs.append(aug_data["image"])
            batch_aug_lbls.append(aug_data["label"])
        
        batch_out['r_3d'] = {
            "image": torch.stack(batch_aug_imgs),
            "label": torch.stack(batch_aug_lbls),
        }

    return batch_out




def my_collate_test(batch):
    batch_out = {
        'r_2d_ap': None, 'r_2d_lat': None, 'r_3d': None, 
        'name': [], 'ct_oir_size': [], 'prop': [], 'valid_shapes': [],
        'shifts': []
    }
    
    has_ap = 'image2d_ap' in batch[0]
    has_lat = 'image2d_lat' in batch[0]
    has_3d = 'image3d' in batch[0]
    num_classes = batch[0]['num_classes']

    imgs_ap, lbls_ap, w_ap = [], [], []
    imgs_lat, lbls_lat, w_lat = [], [], []
    imgs_3d, lbls_3d, w_3d = [], [], []

    for item in batch:
        batch_out['name'].append(item['name'])
        batch_out['ct_oir_size'].append(item['ct_oir_size'])
        
        if 'prop' in item: 
            batch_out['prop'].append(item['prop'])
            if 'shift' in item['prop']:
                batch_out['shifts'].append(item['prop']['shift'])
        
        if has_ap:
            imgs_ap.append(item['image2d_ap'])
            lbls_ap.append(item['label2d_ap'])
        if has_lat:
            imgs_lat.append(item['image2d_lat'])
            lbls_lat.append(item['label2d_lat'])
        if has_3d:
            imgs_3d.append(item['image3d'])
            lbls_3d.append(item['label3d'])
        if 'valid_shape' in item:
            batch_out['valid_shapes'].append(item['valid_shape'])
    
    if len(batch_out['valid_shapes']) > 0:
        batch_out['valid_shapes'] = torch.from_numpy(np.stack(batch_out['valid_shapes'], 0))
    else:
        batch_out['valid_shapes'] = None

    if len(batch_out['shifts']) > 0:
        batch_out['shifts'] = torch.from_numpy(np.stack(batch_out['shifts'], 0))
    else:
        batch_out['shifts'] = None

    batch_out['ct_oir_size'] = np.stack(batch_out['ct_oir_size'], 0)
    batch_out['name'] = np.array(batch_out['name'])

    if has_ap:
        batch_out['r_2d_ap'] = call_aug_batch(imgs_ap, lbls_ap, batch_out['name'], num_classes, is_train=False)
        
    if has_lat:
        batch_out['r_2d_lat'] = call_aug_batch(imgs_lat, lbls_lat, batch_out['name'], num_classes, is_train=False)
        

    if has_3d:
        img_3d_np = np.stack(imgs_3d, 0) # (B, C, D, H, W)
        lbl_3d_np = np.stack(lbls_3d, 0) # (B, C, D, H, W)
        
        
        batch_out['r_3d'] = {
            "image": torch.from_numpy(img_3d_np), 
            "label": torch.from_numpy(lbl_3d_np), 
            "prop": batch_out['prop']
        }

    return batch_out