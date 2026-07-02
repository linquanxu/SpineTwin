import argparse
import os, sys
sys.path.append("..")
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import timeit, time
from utils.ParaFlop import print_model_parm_nums, print_model_parm_flops, torch_summarize_df
start = timeit.default_timer()
import matplotlib.pyplot as plt
import os
from collections import namedtuple
import math
sys.path.append("..") 
from model.project import DiffDRR_Projector

def resize_pos_embed(state_dict, model, patch_size=16, patch_size_2d=16):

    target_grid_3d = getattr(model, 'feat_size', None) 
    target_grid_2d = getattr(model, 'feat_size_2d', None)
    target_time_len = None

    if hasattr(model, 'video_embed'):
        if hasattr(model.video_embed, 'embeddings_st_pos_3D'):
            target_time_len = model.video_embed.embeddings_st_pos_3D.max_frames
        elif hasattr(model.video_embed, 'embeddings_st_pos_2D'):
            target_time_len = model.video_embed.embeddings_st_pos_2D.max_frames

    keys = list(state_dict.keys())
    for key in keys:
        if 'pos_embed' not in key:
            continue
        chk_emb = state_dict[key] 
        is_flat_param = False
        if chk_emb.dim() == 2:
            chk_emb = chk_emb.unsqueeze(0)
            is_flat_param = True
        
        if chk_emb.dim() != 3:
            continue

        n_tokens = chk_emb.shape[1]
        embed_dim = chk_emb.shape[2]
        if key in ['decoder_pos_embed_2D', 'decoder_pos_embed_3D']:
            continue
        if 'temporal_pos_embed' in key:
            if target_time_len is None:
                continue
            
            if n_tokens != target_time_len:
                print(f"  [Temporal Interpolation] Resizing {key}: {n_tokens} -> {target_time_len}")
                chk_emb_trans = chk_emb.transpose(1, 2)
                new_emb_trans = F.interpolate(
                    chk_emb_trans,
                    size=(target_time_len),
                    mode='linear',
                    align_corners=False
                )
                
                new_emb = new_emb_trans.transpose(1, 2)
                
                if is_flat_param:
                    new_emb = new_emb.squeeze(0)
                
                state_dict[key] = new_emb
            continue

        if target_grid_3d is not None and ('3D' in key or '3d' in key) and '2D' not in key:
            target_tokens_3d = target_grid_3d[0] * target_grid_3d[1] * target_grid_3d[2]
            
            if n_tokens != target_tokens_3d:
                print(f"  [3D Spatial Interpolation] Resizing {key}: {n_tokens} -> {target_tokens_3d}")
                
                d_old = int(round(n_tokens ** (1/3)))
                if d_old**3 == n_tokens:
                    old_grid = (d_old, d_old, d_old)
                else:
                    print(f"    Warning: {n_tokens} is not a perfect cube. Assuming 16x16x16.")
                    old_grid = (16, 16, 16)

                chk_emb_grid = chk_emb.transpose(1, 2).reshape(1, embed_dim, old_grid[0], old_grid[1], old_grid[2])
                new_emb_grid = F.interpolate(chk_emb_grid, size=target_grid_3d, mode='trilinear', align_corners=False)
                new_emb = new_emb_grid.flatten(2).transpose(1, 2)
                
                if is_flat_param:
                    new_emb = new_emb.squeeze(0)
                state_dict[key] = new_emb
                continue 
        if target_grid_2d is not None and ('2D' in key or '2d' in key or 'ap' in key or 'lat' in key):
            target_tokens_2d = target_grid_2d[0] * target_grid_2d[1]
            
            if n_tokens != target_tokens_2d:
                print(f"  [2D Spatial Interpolation] Resizing {key}: {n_tokens} -> {target_tokens_2d}")

                h_old = int(math.sqrt(n_tokens))
                if h_old * h_old != n_tokens:
                     print(f"    Warning: {n_tokens} is not a perfect square.")
                old_grid = (h_old, h_old)
                
                chk_emb_grid = chk_emb.transpose(1, 2).reshape(1, embed_dim, old_grid[0], old_grid[1])
                new_emb_grid = F.interpolate(chk_emb_grid, size=target_grid_2d, mode='bilinear', align_corners=False)
                new_emb = new_emb_grid.flatten(2).transpose(1, 2)
                
                if is_flat_param:
                    new_emb = new_emb.squeeze(0)
                state_dict[key] = new_emb
    return state_dict

def load_pretrained_weights(model, checkpoint_path):
    print(f"Loading weights from {checkpoint_path} with auto-interpolation (Spatial & Temporal)...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    state_dict = resize_pos_embed(state_dict, model)
    
    msg = model.load_state_dict(state_dict, strict=False)
    
    print(f"Weights loaded successfully.")
    print(f"  Missing keys: {len(msg.missing_keys)}")
    print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    if len(msg.missing_keys) > 0:
        print("  Example missing keys:", msg.missing_keys[:3])

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def extract_coordinates_unbiased(heatmaps):
    K, D, H, W = heatmaps.shape
    coords = []

    for k in range(K):
        hm = heatmaps[k]
        flat_idx = np.argmax(hm)
        z, y, x = np.unravel_index(flat_idx, (D, H, W))
        if 1 < z < D-2 and 1 < y < H-2 and 1 < x < W-2:
            try:
                def get_offset(v_minus, v_center, v_plus):
                    l = np.log(max(v_minus, 1e-10))
                    c = np.log(max(v_center, 1e-10))
                    r = np.log(max(v_plus, 1e-10))
                    first = (r - l) / 2
                    second = r - 2*c + l
                    if abs(second) < 1e-6: return 0.0
                    return -first / second

                dz = get_offset(hm[z-1, y, x], hm[z, y, x], hm[z+1, y, x])
                dy = get_offset(hm[z, y-1, x], hm[z, y, x], hm[z, y+1, x])
                dx = get_offset(hm[z, y, x-1], hm[z, y, x], hm[z, y, x+1])
                
                dz = np.clip(dz, -0.5, 0.5)
                dy = np.clip(dy, -0.5, 0.5)
                dx = np.clip(dx, -0.5, 0.5)
                
                coords.append([z + dz, y + dy, x + dx])
            except:
                coords.append([z, y, x])
        else:
            coords.append([z, y, x])

    return np.array(coords)

def extract_coordinates_2d_unbiased(heatmaps):
    K, H, W = heatmaps.shape
    coords = []

    for k in range(K):
        hm = heatmaps[k]
        flat_idx = np.argmax(hm)
        y, x = np.unravel_index(flat_idx, (H, W))
        
        if 1 < y < H-2 and 1 < x < W-2:
            try:
                def get_offset(v_minus, v_center, v_plus):
                    l = np.log(max(v_minus, 1e-10))
                    c = np.log(max(v_center, 1e-10))
                    r = np.log(max(v_plus, 1e-10))
                    
                    first = (r - l) / 2
                    second = r - 2*c + l
                    
                    if abs(second) < 1e-6: 
                        return 0.0
                    return -first / second

                dy = get_offset(hm[y-1, x], hm[y, x], hm[y+1, x])
                dx = get_offset(hm[y, x-1], hm[y, x], hm[y, x+1])
                dy = np.clip(dy, -0.5, 0.5)
                dx = np.clip(dx, -0.5, 0.5)
                
                coords.append([x + dx, y + dy])
            except:
                coords.append([x, y])
        else:
            coords.append([x, y])

    return np.array(coords)

def restart_from_checkpoint(ckp_path, run_variables=None, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return
    print("Found checkpoint at {}".format(ckp_path))

    checkpoint = torch.load(ckp_path, map_location="cpu")
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded {} from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded {} from checkpoint '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))
        else:
            print("=> failed to load {} from checkpoint '{}'".format(key, ckp_path))

    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]

def norm_state(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    m = x.mean()
    s = x.std()
    return (x - m) / (s + 1e-6)

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def enforce_determinism():
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False 
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True 
    torch.use_deterministic_algorithms(True, warn_only=True)

def set_monai_determinism(seed: int = 42):
    from monai.utils import set_determinism
    set_determinism(seed=seed)

def validate(args, input_size, model, ValLoader, num_classes, engine, input_size_2d=None, writer=None, epoch=None):
    fold = 'fold0'
    mode_s = 'fusion'
    fname_ap = f'results/{mode_s}/{fold}/ap_error.txt'
    fname_lat = f'results/{mode_s}/{fold}/lat_error.txt'
    fname_ct = f'results/{mode_s}/{fold}/ct_error.txt'
    fname_all = f'results/{mode_s}/{fold}/all_error.txt'
    con_ap = f'results/{mode_s}/{fold}/con_ap_error.txt'
    con_lat = f'results/{mode_s}/{fold}/con_lat_error.txt'
    fname_id = f'results/{mode_s}/{fold}/id_error.txt'
    mode = args.mode
    
    list_3d_gt, list_3d_pred_no_rl = [], []
    list_2d_ap_gt, list_2d_ap_pred_no_rl = [], []
    list_2d_lat_gt, list_2d_lat_pred_no_rl = [], []
    valid_shapes = None

    model[0].eval()

    for index, batch in enumerate(ValLoader):
        r_2d_ap = batch.get('r_2d_ap')
        r_2d_lat = batch.get('r_2d_lat')
        r_3d = batch.get('r_3d')
        ct_oir_size = batch['ct_oir_size']
        
        image2d_ap, image2d_lat, images_3d = None, None, None
        
        if r_2d_ap is not None:
            image2d_ap = r_2d_ap['image'].cuda(non_blocking=True)
            # label is (B, C, H, W)
            label2d_ap = r_2d_ap['label'] 

        if r_2d_lat is not None:
            image2d_lat = r_2d_lat['image'].cuda(non_blocking=True)
            label2d_lat = r_2d_lat['label']

        if r_3d is not None:
            # r_3d['image'] is (C, D, H, W), need unsqueeze for Batch -> (1, C, D, H, W)
            images_3d = r_3d["image"].cuda(non_blocking=True)
            labels_3d = r_3d["label"] # Keep on CPU for metric calc
        
        
        if args.mode in [ 'fusion']:
            valid_shapes = batch.get('valid_shapes').cuda(non_blocking=True)
        
        data = {
            "data_2d_ap": image2d_ap, 
            "data_2d_lat": image2d_lat,
            "data_3d": images_3d, 
            'ct_oir_size': ct_oir_size,
            'valid_shapes': valid_shapes
        }

        with torch.no_grad():
            outputs = model[0](data) 
            if args.mode in ['only_2d_ap', 'only_2d_lat', 'only_3d']:
                pred_2d_ap_out, pred_2d_lat_out, pred_3d_out = outputs
            
            if args.mode in ['joint','fusion']:
                pred_2d_ap_out, pred_2d_lat_out, pred_3d_out = outputs['pred_2d_ap'], outputs['pred_2d_lat'], outputs['pred_3d']

            if mode in ['only_2d_ap', 'joint', 'fusion'] and image2d_ap is not None:
                pred_2d_ap_sig = torch.sigmoid(pred_2d_ap_out)
                
                coords_pred = extract_coordinates_2d_unbiased(pred_2d_ap_sig[0].cpu().numpy())
                coords_gt = extract_coordinates_2d_unbiased(label2d_ap[0].cpu().numpy())
                
                list_2d_ap_pred_no_rl.append(coords_pred) 
                list_2d_ap_gt.append(coords_gt)

            if mode in ['only_2d_lat', 'joint', 'fusion'] and image2d_lat is not None:
                pred_2d_lat_sig = torch.sigmoid(pred_2d_lat_out)
                
                coords_pred = extract_coordinates_2d_unbiased(pred_2d_lat_sig[0].cpu().numpy())
                coords_gt = extract_coordinates_2d_unbiased(label2d_lat[0].cpu().numpy())
                
                list_2d_lat_pred_no_rl.append(coords_pred) 
                list_2d_lat_gt.append(coords_gt)

            if mode in ['only_3d', 'fusion', 'joint'] and images_3d is not None:
                pred_3d_sig = torch.sigmoid(pred_3d_out)
                
                coords_pred = extract_coordinates_unbiased(pred_3d_sig[0].detach().cpu().numpy())
                coords_gt = extract_coordinates_unbiased(labels_3d[0].detach().cpu().numpy())
                
                list_3d_pred_no_rl.append(coords_pred)
                list_3d_gt.append(coords_gt)

    metrics_all = {'MRE_mean': 1000.0} # Default high value
    
    has_ap = len(list_2d_ap_gt) > 0
    has_lat = len(list_2d_lat_gt) > 0
    has_3d = len(list_3d_gt) > 0

    if mode == 'only_2d_ap' and has_ap:
        metrics_all = compute_metrics(list_2d_ap_gt, list_2d_ap_pred_no_rl, norm_factor=50)
        print_modal_metrics("AP", metrics_all)
        
    elif mode == 'only_2d_lat' and has_lat:
        metrics_all = compute_metrics(list_2d_lat_gt, list_2d_lat_pred_no_rl, norm_factor=50)
        print_modal_metrics("LAT", metrics_all)

    elif mode == 'only_3d' and has_3d:
        metrics_all = compute_metrics(list_3d_gt, list_3d_pred_no_rl, norm_factor=30)
        print_modal_metrics("CT (3D)", metrics_all)

    elif mode in ['joint', 'fusion']:
        errors_list = []
        id_count_total = 0
        landmarks_total = 0
        
        fid = open(fname_id, 'w')
        if has_ap:
            m_ap = compute_metrics(list_2d_ap_gt, list_2d_ap_pred_no_rl, norm_factor=50)
            print_modal_metrics("AP", m_ap)
            errors_list.append(m_ap['errors'])
            id_count_total += m_ap['id_count']
            landmarks_total += m_ap['total_landmarks']
            with open(fname_ap, 'w') as f:
                for error in m_ap['errors']:
                    f.write(f"{error}\n")
            fid.write(f"ap:{m_ap['id_count'], m_ap['total_landmarks']}\n")

        if has_lat:
            m_lat = compute_metrics(list_2d_lat_gt, list_2d_lat_pred_no_rl, norm_factor=50)
            print_modal_metrics("LAT", m_lat)
            errors_list.append(m_lat['errors'])
            id_count_total += m_lat['id_count']
            landmarks_total += m_lat['total_landmarks']
            with open(fname_lat, 'w') as f:
                for error in m_lat['errors']:
                    f.write(f"{error}\n")
            fid.write(f"lat:{m_lat['id_count'], m_lat['total_landmarks']}\n")
        
        if has_3d:
            m_3d = compute_metrics(list_3d_gt, list_3d_pred_no_rl, norm_factor=30)
            print_modal_metrics("CT", m_3d)
            errors_list.append(m_3d['errors'])
            id_count_total += m_3d['id_count']
            landmarks_total += m_3d['total_landmarks']
            with open(fname_ct, 'w') as f:
                for error in m_3d['errors']:
                    f.write(f"{error}\n")
            fid.write(f"ct:{m_3d['id_count'], m_3d['total_landmarks']}\n")
        
        if errors_list:
            errors_all = np.concatenate(errors_list, axis=0)
            metrics_all = summarize_errors(errors_all, norm_factor=30, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
            metrics_all['ID_rate'] = comput_ID(id_count_total, landmarks_total)
            with open(fname_all, 'w') as f:
                for error in errors_all:
                    f.write(f"{error}\n")
            fid.write(f"all:{id_count_total}, {landmarks_total}\n")
        else:
            print("Warning: No data found for Joint/Fusion evaluation.")

    if 'MRE_mean' in metrics_all and metrics_all['MRE_mean'] != 1000.0:
        print(f"[{mode.upper()}] Final Results:")
        print(f"MRE_all: {metrics_all['MRE_mean']:.3f}±{metrics_all.get('MRE_std', 0):.3f}")
        print(f"NME_all: {metrics_all['NME_mean']:.3f}±{metrics_all.get('NME_std', 0):.3f}")
        print(f"SR@2mm:  {metrics_all['SR@2.0_mean']:.3f}±{metrics_all.get('SR@2.0_std', 0):.3f}")
        print(f"SR@4mm:  {metrics_all['SR@4.0_mean']:.3f}±{metrics_all.get('SR@4.0_std', 0):.3f}")
        print(f"AUC={metrics_all['AUC_mean']:.3f}",)
        print(f"ID_rate: {metrics_all.get('ID_rate', 0):.3f}")
    
    metrics_re = {'total': metrics_all['MRE_mean']}

    if mode in ['joint', 'fusion']:
        if has_ap:
            metrics_re['ap'] = m_ap['MRE_mean']
        if has_lat:
            metrics_re['lat'] = m_lat['MRE_mean']
        if has_3d:
            metrics_re['ct'] = m_3d['MRE_mean']

    if mode in ['joint','fusion']:
        device = images_3d.device
        fusion_metrics = val_ccd_fusion(
                args=args, 
                model=model[0], 
                ValLoader=ValLoader, 
                device=device, 
                input_size_3d=(64, 160, 160),
                con_ap = con_ap,
                con_lat = con_lat
            )
    return metrics_re

def print_modal_metrics(name, m):
    print(
        f"{name}: "
        f"MRE={m['MRE_mean']:.3f}±{m['MRE_std']:.3f}, "
        f"NME={m['NME_mean']:.3f}±{m['NME_std']:.3f}, "
        f"SR@2mm={m['SR@2.0_mean']:.3f}±{m['SR@2.0_std']:.3f}, "
        f"SR@4mm={m['SR@4.0_mean']:.3f}±{m['SR@4.0_std']:.3f}, "
        f"AUC={m['AUC_mean']:.3f}",
        f"ID_rate={m['ID_rate']:.3f}"
    )
    
def summarize_errors(
    errors_all,
    norm_factor=None,
    sr_thresholds=(2.0, ),
    auc_max_threshold=10.0,
):
    errors = np.asarray(errors_all, dtype=np.float32).ravel()
    num_points = errors.shape[0]
    metrics = {}
    metrics["MRE_mean"] = float(errors.mean())
    metrics["MRE_std"] = float(errors.std(ddof=1))

    if norm_factor is not None:
        nme_vals = errors / float(norm_factor)
        metrics["NME_mean"] = float(nme_vals.mean())
        metrics["NME_std"] = float(nme_vals.std(ddof=1))
    else:
        metrics["NME_mean"] = None
        metrics["NME_std"] = None

    for thr in sr_thresholds:
        mask = errors < thr 
        sr_mean = float(mask.mean())
        sr_std  = float(mask.std(ddof=1))
        metrics[f"SR@{thr}_mean"] = sr_mean
        metrics[f"SR@{thr}_std"] = sr_std

    thresholds = np.linspace(0, auc_max_threshold, 100)
    sr_curve = np.array([(errors < t).mean() for t in thresholds], dtype=np.float32)
    auc = np.trapz(sr_curve, thresholds) / float(auc_max_threshold)
    metrics["AUC_mean"] = float(auc)  
    metrics["AUC_std"] = float(0.0) 
    return metrics

def comput_ID(ID_count, ID_landmarkers):
    ID_rate =  float(ID_count) / float(ID_landmarkers)
    return ID_rate

def compute_metrics(gt_list,
                    pred_list,
                    norm_factor=None,
                    sr_threshold=2.0,
                    auc_max_threshold=10.0,
                    space3d=None,
                    num_landmarks_per_sample=None,
                    id_threshold=20.0):
    gt = np.array(gt_list)
    pred = np.array(pred_list)

    if gt.ndim == 3:
        num_samples, K, coord_dim = gt.shape
        if num_landmarks_per_sample is None:
            num_landmarks_per_sample = K
        else:
            assert num_landmarks_per_sample == K, \
                f"num_landmarks_per_sample={num_landmarks_per_sample} does not match gt.shape[1]={K}"
    elif gt.ndim == 2:
        num_points, coord_dim = gt.shape
        num_samples = num_points // num_landmarks_per_sample
        gt = gt.reshape(num_samples, num_landmarks_per_sample, coord_dim)
        pred = pred.reshape(num_samples, num_landmarks_per_sample, coord_dim)
    else:
        raise ValueError(f"Unsupported gt shape: {gt.shape}")

    num_samples, K, coord_dim = gt.shape

    if space3d is None:
        space3d = np.array([1.5, 0.8, 0.8])

    if coord_dim == 3:
        spacing = np.array(space3d) 
    else:
        target = 0.8/(1000/600)
        spacing = np.array([target, target])
    gt_mm = gt * spacing
    pred_mm = pred * spacing
    gt_flat   = gt_mm.reshape(-1, coord_dim)
    pred_flat = pred_mm.reshape(-1, coord_dim)

    errors = np.linalg.norm(gt_flat - pred_flat, axis=1)

    per_axis_errors_mm = np.abs(gt_flat - pred_flat)   # [N*K, d]

    mre_mean = np.mean(errors)
    mre_std  = np.std(errors, ddof=1)

    if norm_factor is not None:
        nme_vals = errors / norm_factor
        nme_mean = np.mean(nme_vals)
        nme_std  = np.std(nme_vals, ddof=1)
    else:
        nme_mean, nme_std = None, None

    sr_indicator = (errors < sr_threshold).astype(np.float32)
    sr_mean = np.mean(sr_indicator)
    sr_std  = np.std(sr_indicator, ddof=1)

    sr_indicator_4 = (errors < 4.0).astype(np.float32)
    sr_mean_4 = np.mean(sr_indicator_4)
    sr_std_4  = np.std(sr_indicator_4, ddof=1)

    thresholds = np.linspace(0, auc_max_threshold, 100)
    sr_curve = np.array([np.mean(errors < t) for t in thresholds])
    auc_val = np.trapz(sr_curve, thresholds) / auc_max_threshold
    auc_mean = auc_val
    auc_std  = 0.0  

    per_axis_mre_mean = np.mean(per_axis_errors_mm, axis=0).tolist()
    per_axis_mre_std  = np.std(per_axis_errors_mm, axis=0, ddof=1).tolist()

    id_count = 0

    for j in range(num_samples):  
        gt_j   = gt_mm[j]    
        pred_j = pred_mm[j] 

        for i in range(num_landmarks_per_sample): 
            pred_i = pred_j[i] 
            dists = np.linalg.norm(gt_j - pred_i, axis=1)
            k_star = np.argmin(dists)
            if (k_star == i) and (dists[i] <= id_threshold):
                id_count += 1

    total_landmarks = num_samples * num_landmarks_per_sample
    id_rate = id_count / total_landmarks if total_landmarks > 0 else 0.0

    return {
        "MRE_mean": mre_mean,
        "MRE_std":  mre_std,
        "NME_mean": nme_mean,
        "NME_std":  nme_std,
        f"SR@{sr_threshold}_mean": sr_mean,
        f"SR@{sr_threshold}_std":  sr_std,
        f"SR@{4.0}_mean": sr_mean_4,
        f"SR@{4.0}_std":  sr_std_4,
        "AUC_mean": auc_mean,
        "AUC_std":  auc_std,
        # "per_axis_MRE_mean": per_axis_mre_mean,
        # "per_axis_MRE_std":  per_axis_mre_std,
        # "ID_count": int(id_count),
        "ID_rate":  float(id_rate),
        "errors": errors.tolist(),
        "id_count": int(id_count),
        "total_landmarks": int(total_landmarks),
    }

def spatial_log_softmax(tensor):
    # tensor: [B, C, D, H, W] or [B, C, H, W]
    b, c, *dims = tensor.shape
    tensor_flat = tensor.view(b, c, -1)
    tensor_log_soft = F.log_softmax(tensor_flat, dim=-1)
    return tensor_log_soft.view(b, c, *dims)

def linear_normalize_gt(tensor, epsilon=1e-6):
    # tensor: [B, C, D, H, W] or [B, C, H, W]
    b, c = tensor.shape[:2]
    flat_tensor = tensor.view(b, c, -1)
    sum_val = flat_tensor.sum(dim=-1, keepdim=True)
    
    norm_tensor = flat_tensor / (sum_val + epsilon)
    
    return norm_tensor.view_as(tensor)

def run_lr_finder(args, model, trainloader, optimizer, device, awl, scaler, kl_criterion, start_lr=1e-7, end_lr=50, num_iter=1000):
    import math

    print(f"Running LR Finder: {start_lr} -> {end_lr} over {num_iter} steps...")

    mult = (end_lr / start_lr) ** (1 / num_iter)

    model.train()
    
    lrs = []
    losses = []
    best_loss = float('inf')
    avg_loss = 0
    beta = 0.98 
    
    current_lr = start_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    train_iter = iter(trainloader)
    
    for i in range(num_iter):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            batch = next(train_iter)

        r_2d_ap = batch.get('r_2d_ap')
        r_2d_lat = batch.get('r_2d_lat')
        r_3d = batch.get('r_3d')
        ct_oir_size = batch['ct_oir_size']
        
        images_2d_ap, labels_2d_ap = None, None
        images_2d_lat, labels_2d_lat = None, None
        images, labels = None, None
        valid_shapes = None
        if args.mode in [ 'fusion']:
            valid_shapes = batch.get('valid_shapes').cuda(non_blocking=True)

        if r_2d_ap is not None:
            images_2d_ap = r_2d_ap['image'].cuda(non_blocking=True)
            labels_2d_ap = r_2d_ap['label'].cuda(non_blocking=True)
        
        if r_2d_lat is not None:
            images_2d_lat = r_2d_lat['image'].cuda(non_blocking=True)
            labels_2d_lat = r_2d_lat['label'].cuda(non_blocking=True)
        
        if r_3d is not None:
            images = r_3d["image"].cuda(non_blocking=True)
            labels = r_3d["label"].cuda(non_blocking=True)
        
        data = {
            "data_2d_ap": images_2d_ap, "labels_2d_ap": labels_2d_ap, 
            "data_2d_lat": images_2d_lat, "labels_2d_lat": labels_2d_lat,
            "data_3d": images, "labels_3d": labels, 'ct_oir_size': ct_oir_size,
            'valid_shapes': valid_shapes
        }

        optimizer.zero_grad()
        use_amp = (scaler is not None)
        
        def compute_loss():
            if args.mode in ['fusion']:
                outputs = model(data) 
                pred_2d_ap, pred_2d_lat, pred_3d = outputs['pred_2d_ap'], outputs['pred_2d_lat'], outputs['pred_3d']
                coords_3d, coords_ap, coords_lat = outputs['coords_3d'], outputs['coords_ap'], outputs['coords_lat']
            else:       
                pred_2d_ap, pred_2d_lat, pred_3d = model(data) 
                    
            total_loss = 0.0
            
            if args.mode in ['only_2d_ap', 'joint', 'fusion'] and images_2d_ap is not None:
                sig_pred_2d_ap = torch.sigmoid(pred_2d_ap)
                mse_2d_ap = F.mse_loss(sig_pred_2d_ap, labels_2d_ap) 
                log_prob_2d_ap = spatial_log_softmax(pred_2d_ap)
                with torch.no_grad():
                    prob_gt_2d_ap = linear_normalize_gt(labels_2d_ap)
                kl_2d_ap = kl_criterion(log_prob_2d_ap, prob_gt_2d_ap)
                total_loss += mse_2d_ap + 0.01 * kl_2d_ap

            if args.mode in ['only_2d_lat', 'joint', 'fusion'] and images_2d_lat is not None:
                sig_pred_2d_lat = torch.sigmoid(pred_2d_lat)
                mse_2d_lat = F.mse_loss(sig_pred_2d_lat, labels_2d_lat)
                log_prob_2d_lat = spatial_log_softmax(pred_2d_lat)
                with torch.no_grad():
                    prob_gt_2d_lat = linear_normalize_gt(labels_2d_lat)
                kl_2d_lat = kl_criterion(log_prob_2d_lat, prob_gt_2d_lat)
                total_loss += mse_2d_lat + 0.01 * kl_2d_lat

            if args.mode in ['only_3d', 'joint', 'fusion'] and images is not None:
                sig_pred_3d = torch.sigmoid(pred_3d)
                mse_3d = F.mse_loss(sig_pred_3d, labels)
                log_prob_3d = spatial_log_softmax(pred_3d)
                with torch.no_grad():
                    prob_gt_3d = linear_normalize_gt(labels)
                kl_3d = kl_criterion(log_prob_3d, prob_gt_3d)
                total_loss += mse_3d + 0.01 * kl_3d
            
            return total_loss

        if use_amp:
            with torch.cuda.amp.autocast():
                loss = compute_loss()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = compute_loss()
            loss.backward()
            optimizer.step()

        current_loss = loss.item()

        if math.isnan(current_loss):
            print(f"Loss became NaN at step {i}, stopping early.")
            break

        avg_loss = beta * avg_loss + (1 - beta) * current_loss
        smoothed_loss = avg_loss / (1 - beta**(i + 1))
            
        if smoothed_loss < best_loss or i == 0:
            best_loss = smoothed_loss
            
        lrs.append(current_lr)
        losses.append(smoothed_loss)
        
        if (i + 1) % 10 == 0:
            print(f"Step {i+1}/{num_iter}: LR = {current_lr:.8f}, Loss = {smoothed_loss:.4f}")
        current_lr *= mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
    if len(lrs) > 0:
        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        plt.figure()
        plt.plot(lrs, losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title(f'LR Finder (Mode: {args.mode})')
        save_path = os.path.join(args.snapshot_dir, 'lr_finder_curve.png')
        plt.savefig(save_path)
        print(f"LR Finder curve saved to {save_path}")
        
        min_loss_idx = losses.index(min(losses))
        suggested_lr = lrs[min_loss_idx]
        print(f"Minimum Loss found at LR: {suggested_lr:.8f}")
        print(f"Suggested LR (Min/10): {suggested_lr/10:.8f}")
    else:
        print("No steps completed, cannot plot.")

def val_cce(args, models_dict, ValLoader, device, input_size_3d=(64, 160, 160)):
    fold = 'fold4'
    con_ap = f'/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/{fold}/con_ap_error.txt'
    con_lat = f'/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/{fold}/con_lat_error.txt'
    print(f"\n>>> Starting Cross-Consistency Evaluation (CCE) on {len(ValLoader)} samples (Strict Alignment Mode)...")
    
    drr_projector = DiffDRR_Projector(sdd=1000, height=256, delx=0.8, device=device)
    
    for key, model in models_dict.items():
        model.eval()
        model.to(device)
    dist_ap_list = []
    dist_lat_list = []
    drr_height = 256.0


    with torch.no_grad():
        for index, batch in enumerate(ValLoader):
            r_2d_ap = batch.get('r_2d_ap')
            r_2d_lat = batch.get('r_2d_lat')
            r_3d = batch.get('r_3d')
            batch_props = batch.get('prop', [])
            
            if r_2d_ap is None or r_2d_lat is None or r_3d is None:
                continue

            image2d_ap = r_2d_ap['image'].cuda(non_blocking=True)
            image2d_lat = r_2d_lat['image'].cuda(non_blocking=True)
            images_3d = r_3d["image"].cuda(non_blocking=True)

            data_ap = {"data_2d_ap": image2d_ap}
            data_lat = {"data_2d_lat": image2d_lat}
            data_ct = {"data_3d": images_3d}
            
            out_tuple_ap = models_dict['ap'](data_ap)
            out_tuple_lat = models_dict['lat'](data_lat)
            out_tuple_ct = models_dict['ct'](data_ct)

            pred_2d_ap = out_tuple_ap[0]
            pred_2d_lat = out_tuple_lat[1]
            pred_3d = out_tuple_ct[2] 

            if pred_3d is None or pred_2d_ap is None or pred_2d_lat is None:
                continue
            pred_2d_ap_sig = torch.sigmoid(pred_2d_ap)[0].cpu().numpy()
            pred_2d_lat_sig = torch.sigmoid(pred_2d_lat)[0].cpu().numpy()
            pred_3d_sig = torch.sigmoid(pred_3d)[0].cpu().numpy()
            
            coords_3d_np = extract_coordinates_unbiased(pred_3d_sig) 
            coords_ap_np = extract_coordinates_2d_unbiased(pred_2d_ap_sig)
            coords_lat_np = extract_coordinates_2d_unbiased(pred_2d_lat_sig)
            
            coords_3d_vox = torch.tensor(coords_3d_np, device=device, dtype=torch.float32).unsqueeze(0)
            coords_ap_pix = torch.tensor(coords_ap_np, device=device, dtype=torch.float32).unsqueeze(0)
            coords_lat_pix = torch.tensor(coords_lat_np, device=device, dtype=torch.float32).unsqueeze(0)

            proj_ap_pix, proj_lat_pix = drr_projector(coords_3d_vox, batch_props)
            
            proj_ap_pix_flipped = proj_ap_pix.clone()
            proj_ap_pix_flipped[..., 1] = 255.0 - proj_ap_pix_flipped[..., 1]
            
            proj_lat_pix_flipped = proj_lat_pix.clone()
            proj_lat_pix_flipped[..., 1] = 255.0 - proj_lat_pix_flipped[..., 1]
            diff_ap = torch.norm(proj_ap_pix_flipped - coords_ap_pix, dim=-1)
            diff_lat = torch.norm(proj_lat_pix_flipped - coords_lat_pix, dim=-1)
            
            proj_ap_norm = proj_ap_pix_flipped / drr_height
            valid_mask_ap = (proj_ap_norm >= 0) & (proj_ap_norm <= 1)
            valid_mask_ap = valid_mask_ap.all(dim=-1) # (B, N)
            
            proj_lat_norm = proj_lat_pix_flipped / drr_height
            valid_mask_lat = (proj_lat_norm >= 0) & (proj_lat_norm <= 1)
            valid_mask_lat = valid_mask_lat.all(dim=-1)

            if valid_mask_ap.any():
                dist_ap_list.extend(diff_ap[valid_mask_ap].cpu().numpy().tolist())
            
            if valid_mask_lat.any():
                dist_lat_list.extend(diff_lat[valid_mask_lat].cpu().numpy().tolist())

    final_metrics = {}
    with open(con_ap, 'w') as f:
        for error in dist_ap_list:
            f.write(f"{error}\n")
    with open(con_lat, 'w') as f:
        for error in dist_lat_list:
            f.write(f"{error}\n")
    
    mean_ap_dist = np.mean(dist_ap_list) if len(dist_ap_list) > 0 else 0.0
    std_ap_dist = np.std(dist_ap_list) if len(dist_ap_list) > 0 else 0.0
    
    mean_lat_dist = np.mean(dist_lat_list) if len(dist_lat_list) > 0 else 0.0
    std_lat_dist = np.std(dist_lat_list) if len(dist_lat_list) > 0 else 0.0
    
    print("\n=== CCE Results (Pixels @ 256x256) ===")
    print(f"{'AP Consistency':<20}: {mean_ap_dist:.4f}±{std_ap_dist:.4f} px")
    print(f"{'Lat Consistency':<20}: {mean_lat_dist:.4f}±{std_lat_dist:.4f} px")
    
    pixel_spacing = 0.48 
    print(f"{'AP Cons (mm)':<20}: {mean_ap_dist * pixel_spacing:.4f} mm")
    print(f"{'Lat Cons (mm)':<20}: {mean_lat_dist * pixel_spacing:.4f} mm")
    
    total_cons = mean_ap_dist + mean_lat_dist
    print(f"{'Total':<20}: {total_cons:.4f} px")
    print("==========================================================")
    
    final_metrics['cons_ap_pix'] = mean_ap_dist
    final_metrics['cons_lat_pix'] = mean_lat_dist
    final_metrics['total_consistency'] = total_cons

    return final_metrics

def val_metrics_baseline_mean(args, models_dict, ValLoader, device):
    print("\n>>> Starting Multi-Model Evaluation for AP, LAT, and 3D...")
    fold = 'fold4'
    fname_ap = f'/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/{fold}/ap_error.txt'
    fname_lat = f'/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/{fold}/lat_error.txt'
    fname_ct = f'/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/{fold}/ct_error.txt'
    fname_all = f'/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/{fold}/all_error.txt'
    fname_id = f'/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/{fold}/id_error.txt'
    
    for key, model in models_dict.items():
        model.eval()
        model.to(device)
        
    list_3d_gt, list_3d_pred = [], []
    list_2d_ap_gt, list_2d_ap_pred = [], []
    list_2d_lat_gt, list_2d_lat_pred = [], []

    with torch.no_grad():
        for index, batch in enumerate(ValLoader):
            r_2d_ap = batch.get('r_2d_ap')
            r_2d_lat = batch.get('r_2d_lat')
            r_3d = batch.get('r_3d')

            if r_2d_ap is not None:
                image2d_ap = r_2d_ap['image'].cuda(non_blocking=True)
                label2d_ap = r_2d_ap['label']
                out_ap = models_dict['ap']({"data_2d_ap": image2d_ap})
                pred_2d_ap = out_ap[0] if isinstance(out_ap, tuple) else out_ap['pred_2d_ap']
                
                pred_sig = torch.sigmoid(pred_2d_ap)[0].cpu().numpy()
                coords_pred = extract_coordinates_2d_unbiased(pred_sig)
                coords_gt = extract_coordinates_2d_unbiased(label2d_ap[0].cpu().numpy())
                
                list_2d_ap_pred.append(coords_pred)
                list_2d_ap_gt.append(coords_gt)

            if r_2d_lat is not None:
                image2d_lat = r_2d_lat['image'].cuda(non_blocking=True)
                label2d_lat = r_2d_lat['label']
                out_lat = models_dict['lat']({"data_2d_lat": image2d_lat})
                pred_2d_lat = out_lat[1] if isinstance(out_lat, tuple) else out_lat['pred_2d_lat']
                
                pred_sig = torch.sigmoid(pred_2d_lat)[0].cpu().numpy()
                coords_pred = extract_coordinates_2d_unbiased(pred_sig)
                coords_gt = extract_coordinates_2d_unbiased(label2d_lat[0].cpu().numpy())
                
                list_2d_lat_pred.append(coords_pred)
                list_2d_lat_gt.append(coords_gt)

            if r_3d is not None:
                images_3d = r_3d["image"].cuda(non_blocking=True)
                labels_3d = r_3d["label"]
                out_ct = models_dict['ct']({"data_3d": images_3d, "ct_oir_size": batch.get('ct_oir_size')})
                pred_3d = out_ct[2] if isinstance(out_ct, tuple) else out_ct['pred_3d']
                
                pred_sig = torch.sigmoid(pred_3d)[0].cpu().numpy()
                coords_pred = extract_coordinates_unbiased(pred_sig)
                coords_gt = extract_coordinates_unbiased(labels_3d[0].cpu().numpy())
                
                list_3d_pred.append(coords_pred)
                list_3d_gt.append(coords_gt)

    errors_list = []
    id_count_total = 0
    landmarks_total = 0
    
    print("\n" + "="*50)
    print("Multi-Model Evaluation Detailed Results")
    print("="*50)
    fid = open(fname_id, 'w')
    if len(list_2d_ap_gt) > 0:
        m_ap = compute_metrics(list_2d_ap_gt, list_2d_ap_pred, norm_factor=50)
        err_ap = m_ap['errors']
        errors_list.append(err_ap)
        id_count_total += m_ap['id_count']
        landmarks_total += m_ap['total_landmarks']
        
        metrics_ap = summarize_errors(err_ap, norm_factor=50, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
        id_rate_ap = comput_ID(m_ap['id_count'], m_ap['total_landmarks'])
        mre_mean_ap = np.mean(err_ap)
        mre_std_ap = np.std(err_ap, ddof=1)
        
        print(f"[AP] Results:")
        print(f"  MRE:     {mre_mean_ap:.3f}±{mre_std_ap:.3f} mm")
        print(f"  NME:     {metrics_ap.get('NME_mean', 0):.3f}±{metrics_ap.get('NME_std', 0):.3f}")
        print(f"  SR@2mm:  {metrics_ap.get('SR@2.0_mean', 0):.3f}±{metrics_ap.get('SR@2.0_std', 0):.3f}")
        print(f"  SR@4mm:  {metrics_ap.get('SR@4.0_mean', 0):.3f}±{metrics_ap.get('SR@4.0_std', 0):.3f}")
        print(f"  AUC:     {metrics_ap.get('AUC_mean', 0):.3f}")
        print(f"  ID_rate: {id_rate_ap:.3f}")
        print("-" * 30)
        
        with open(fname_ap, 'w') as f:
            for error in m_ap['errors']:
                f.write(f"{error}\n")
        fid.write(f"ap:{m_ap['id_count'], m_ap['total_landmarks']}\n")

    if len(list_2d_lat_gt) > 0:
        m_lat = compute_metrics(list_2d_lat_gt, list_2d_lat_pred, norm_factor=50)
        err_lat = m_lat['errors']
        errors_list.append(err_lat)
        id_count_total += m_lat['id_count']
        landmarks_total += m_lat['total_landmarks']
        
        metrics_lat = summarize_errors(err_lat, norm_factor=50, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
        id_rate_lat = comput_ID(m_lat['id_count'], m_lat['total_landmarks'])
        mre_mean_lat = np.mean(err_lat)
        mre_std_lat = np.std(err_lat, ddof=1)
        
        print(f"[LAT] Results:")
        print(f"  MRE:     {mre_mean_lat:.3f}±{mre_std_lat:.3f} mm")
        print(f"  NME:     {metrics_lat.get('NME_mean', 0):.3f}±{metrics_lat.get('NME_std', 0):.3f}")
        print(f"  SR@2mm:  {metrics_lat.get('SR@2.0_mean', 0):.3f}±{metrics_lat.get('SR@2.0_std', 0):.3f}")
        print(f"  SR@4mm:  {metrics_lat.get('SR@4.0_mean', 0):.3f}±{metrics_lat.get('SR@4.0_std', 0):.3f}")
        print(f"  AUC:     {metrics_lat.get('AUC_mean', 0):.3f}")
        print(f"  ID_rate: {id_rate_lat:.3f}")
        print("-" * 30)
        
        with open(fname_lat, 'w') as f:
            for error in m_lat['errors']:
                f.write(f"{error}\n")
        fid.write(f"lat:{m_lat['id_count'], m_lat['total_landmarks']}\n")

    if len(list_3d_gt) > 0:
        m_3d = compute_metrics(list_3d_gt, list_3d_pred, norm_factor=30)
        err_3d = m_3d['errors']
        errors_list.append(err_3d)
        id_count_total += m_3d['id_count']
        landmarks_total += m_3d['total_landmarks']
        
        metrics_3d = summarize_errors(err_3d, norm_factor=30, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
        id_rate_3d = comput_ID(m_3d['id_count'], m_3d['total_landmarks'])
        mre_mean_3d = np.mean(err_3d)
        mre_std_3d = np.std(err_3d, ddof=1)
        
        print(f"[CT] Results:")
        print(f"  MRE:     {mre_mean_3d:.3f}±{mre_std_3d:.3f} mm")
        print(f"  NME:     {metrics_3d.get('NME_mean', 0):.3f}±{metrics_3d.get('NME_std', 0):.3f}")
        print(f"  SR@2mm:  {metrics_3d.get('SR@2.0_mean', 0):.3f}±{metrics_3d.get('SR@2.0_std', 0):.3f}")
        print(f"  SR@4mm:  {metrics_3d.get('SR@4.0_mean', 0):.3f}±{metrics_3d.get('SR@4.0_std', 0):.3f}")
        print(f"  AUC:     {metrics_3d.get('AUC_mean', 0):.3f}")
        print(f"  ID_rate: {id_rate_3d:.3f}")
        print("-" * 30)
        with open(fname_ct, 'w') as f:
            for error in m_3d['errors']:
                f.write(f"{error}\n")
        fid.write(f"ct:{m_3d['id_count'], m_3d['total_landmarks']}\n")

    if len(errors_list) > 0:
        all_errors = np.concatenate(errors_list, axis=0)
        metrics_all = summarize_errors(all_errors, norm_factor=30, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
        metrics_all['ID_rate'] = comput_ID(id_count_total, landmarks_total)
        
        total_mean = np.mean(all_errors)
        total_std = np.std(all_errors, ddof=1)
        
        metrics_all['MRE_mean'] = total_mean
        metrics_all['MRE_std'] = total_std

        print(f"[BASELINE TOTAL] Final Results:")
        print(f"  MRE_all: {metrics_all['MRE_mean']:.3f}±{metrics_all['MRE_std']:.3f} mm")
        print(f"  NME_all: {metrics_all.get('NME_mean', 0):.3f}±{metrics_all.get('NME_std', 0):.3f}")
        print(f"  SR@2mm:  {metrics_all.get('SR@2.0_mean', 0):.3f}±{metrics_all.get('SR@2.0_std', 0):.3f}")
        print(f"  SR@4mm:  {metrics_all.get('SR@4.0_mean', 0):.3f}±{metrics_all.get('SR@4.0_std', 0):.3f}")
        print(f"  AUC:     {metrics_all.get('AUC_mean', 0):.3f}")
        print(f"  ID_rate: {metrics_all.get('ID_rate', 0):.3f}")
        print("=" * 50)
        with open(fname_all, 'w') as f:
            for error in all_errors:
                f.write(f"{error}\n")
        fid.write(f"all:{id_count_total}, {landmarks_total}\n")
    
        metrics_re = {'total': metrics_all['MRE_mean']}
        if len(list_2d_ap_gt) > 0:
            metrics_re['ap'] = mre_mean_ap
        if len(list_2d_lat_gt) > 0:
            metrics_re['lat'] = mre_mean_lat
        if len(list_3d_gt) > 0:
            metrics_re['ct'] = mre_mean_3d
            
        return metrics_re
    else:
        print("Warning: No valid data evaluated.")
        return {'total': 1000.0}

def val_ccd_fusion(args, model, ValLoader, device, input_size_3d=(64, 160, 160), con_ap = None, con_lat = None):
    drr_projector = DiffDRR_Projector(sdd=1000, height=256, delx=0.8, device=device)
    
    model.eval()
    print(f">>> Starting CCD Fusion Validation (Strict Alignment Mode with Unbiased Decoding)...")
    dist_ap_list = []
    dist_lat_list = []
    
    drr_height = 256.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(ValLoader):
            r_2d_ap = batch.get('r_2d_ap')
            r_2d_lat = batch.get('r_2d_lat')
            r_3d = batch.get('r_3d')
            batch_props = batch.get('prop', [])

            if r_2d_ap is None or r_2d_lat is None or r_3d is None:
                continue

            images_2d_ap = r_2d_ap['image'].to(device)
            images_2d_lat = r_2d_lat['image'].to(device)
            images_3d = r_3d['image'].to(device)
            
            valid_shapes = batch.get('valid_shapes')
            if valid_shapes is not None:
                valid_shapes = valid_shapes.to(device)

            data = {
                "data_2d_ap": images_2d_ap, 
                "data_2d_lat": images_2d_lat,
                "data_3d": images_3d,
                "valid_shapes": valid_shapes,
                "ct_oir_size": batch.get('ct_oir_size')
            }

            outputs = model(data)
            heatmaps_3d = outputs.get('pred_3d')
            heatmaps_ap = outputs.get('pred_2d_ap')
            heatmaps_lat = outputs.get('pred_2d_lat')
            
            if heatmaps_3d is None or heatmaps_ap is None or heatmaps_lat is None:
                continue

            pred_3d_sig = torch.sigmoid(heatmaps_3d)[0].cpu().numpy()
            pred_ap_sig = torch.sigmoid(heatmaps_ap)[0].cpu().numpy()
            pred_lat_sig = torch.sigmoid(heatmaps_lat)[0].cpu().numpy()

            coords_3d_np = extract_coordinates_unbiased(pred_3d_sig) 
            coords_ap_np = extract_coordinates_2d_unbiased(pred_ap_sig)
            coords_lat_np = extract_coordinates_2d_unbiased(pred_lat_sig)
            
            coords_3d_vox = torch.tensor(coords_3d_np, device=device, dtype=torch.float32).unsqueeze(0)
            pred_ap_pix = torch.tensor(coords_ap_np, device=device, dtype=torch.float32).unsqueeze(0)
            pred_lat_pix = torch.tensor(coords_lat_np, device=device, dtype=torch.float32).unsqueeze(0)

            proj_ap_pix, proj_lat_pix = drr_projector(coords_3d_vox, batch_props)
            
            proj_ap_pix_flipped = proj_ap_pix.clone()
            proj_ap_pix_flipped[..., 1] = 255.0 - proj_ap_pix_flipped[..., 1]
            
            proj_lat_pix_flipped = proj_lat_pix.clone()
            proj_lat_pix_flipped[..., 1] = 255.0 - proj_lat_pix_flipped[..., 1]

            diff_ap = torch.norm(proj_ap_pix_flipped - pred_ap_pix, dim=-1)
            diff_lat = torch.norm(proj_lat_pix_flipped - pred_lat_pix, dim=-1)
            
            proj_ap_norm = proj_ap_pix_flipped / drr_height
            valid_mask_ap = (proj_ap_norm >= 0) & (proj_ap_norm <= 1)
            valid_mask_ap = valid_mask_ap.all(dim=-1) # (B, N)
            
            proj_lat_norm = proj_lat_pix_flipped / drr_height
            valid_mask_lat = (proj_lat_norm >= 0) & (proj_lat_norm <= 1)
            valid_mask_lat = valid_mask_lat.all(dim=-1)

            if valid_mask_ap.any():
                dist_ap_list.extend(diff_ap[valid_mask_ap].cpu().numpy().tolist())
            
            if valid_mask_lat.any():
                dist_lat_list.extend(diff_lat[valid_mask_lat].cpu().numpy().tolist())

    results = {}

    with open(con_ap, 'w') as f:
        for error in dist_ap_list:
            f.write(f"{error}\n")
    with open(con_lat, 'w') as f:
        for error in dist_lat_list:
            f.write(f"{error}\n")
    
    mean_ap_dist = np.mean(dist_ap_list) if len(dist_ap_list) > 0 else 0.0
    std_ap_dist = np.std(dist_ap_list) if len(dist_ap_list) > 0 else 0.0
    
    mean_lat_dist = np.mean(dist_lat_list) if len(dist_lat_list) > 0 else 0.0
    std_lat_dist = np.std(dist_lat_list) if len(dist_lat_list) > 0 else 0.0
    
    print("\n=== CCD Fusion Consistency Metrics (Pixels @ 256x256) ===")
    print(f"{'AP Consistency':<20}: {mean_ap_dist:.4f}±{std_ap_dist:.4f} px")
    print(f"{'Lat Consistency':<20}: {mean_lat_dist:.4f}±{std_lat_dist:.4f} px")
    
    pixel_spacing = 0.48 
    print(f"{'AP Cons (mm)':<20}: {mean_ap_dist * pixel_spacing:.4f} mm")
    print(f"{'Lat Cons (mm)':<20}: {mean_lat_dist * pixel_spacing:.4f} mm")
    
    results['cons_ap_pix'] = mean_ap_dist
    results['cons_lat_pix'] = mean_lat_dist
    results['total_consistency'] = mean_ap_dist + mean_lat_dist

    return results

def get_arguments():
    parser = argparse.ArgumentParser(description="Downstream segmentation tasks")
    parser.add_argument("--arch", type=str, default='none')
    parser.add_argument("--train_dir_2d", type=str, default='')
    parser.add_argument("--train_list_2d", type=str, default='')
    parser.add_argument("--train_dir_3d", type=str, default='')
    parser.add_argument("--train_list_3d", type=str, default='')
    parser.add_argument("--snapshot_dir", type=str, default='snapshots/tmp/')
    parser.add_argument("--nnUNet_preprocessed", type=str)
    parser.add_argument("--test_dir_2d", type=str, default='')
    parser.add_argument("--test_list_2d", type=str, default='')
    parser.add_argument("--test_dir_3d", type=str, default='')
    parser.add_argument("--test_list_3d", type=str, default='')
    parser.add_argument("--reload_from_pretrained", type=str2bool, default=False)
    parser.add_argument("--pretrained_path", type=str, default='../snapshots/xx/checkpoint.pth')
    parser.add_argument("--input_size", type=str, default='64,128,128')
    parser.add_argument("--input_size_2d", type=str, default='224,224')
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=False)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.00003)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--not_restore_last", action="store_true")
    parser.add_argument("--save_num_images", type=int, default=2)

    # data aug.
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_mirror_2d", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_scale_2d", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)

    # others
    parser.add_argument("--gpu", type=str, default='None')
    parser.add_argument("--recurrence", type=int, default=1)
    parser.add_argument("--ft", type=str2bool, default=False)
    parser.add_argument("--ohem", type=str2bool, default='False')
    parser.add_argument("--ohem_thres", type=float, default=0.6)
    parser.add_argument("--ohem_keep", type=int, default=200000)
    parser.add_argument("--model_name", type=str, default='None')
    parser.add_argument("--ratio_labels_2d", type=float, default=1.0)

    ### sturcture prior
    parser.add_argument("--is_prior", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default='fusion', choices=['only_2d_ap', 'only_2d_lat','only_3d', 'joint', 'fusion','cce'], help="Training mode")

    return parser