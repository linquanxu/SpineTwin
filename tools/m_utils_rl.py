import argparse
import os, sys
sys.path.append("..")
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import timeit, time
start = timeit.default_timer()
import math
sys.path.append("..") 
from model.project import DiffDRR_Projector
from collections import namedtuple
from model.rl_env import KeypointEnv

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

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_anneal_steps=10000):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.beta_anneal_steps = beta_anneal_steps
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.step = 0

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        prios = self.priorities[:len(self.buffer)]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.step += 1
        if self.step < self.beta_anneal_steps:
            self.beta = self.beta_start + (1 - self.beta_start) * (self.step / self.beta_anneal_steps) 
        return samples, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio + 1e-5  

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

def convert_voxel_to_world(pred_3d_vox, batch_props, device, shifts=None):
    B, N, _ = pred_3d_vox.shape
    origins, spacings, ori_sizes, directions = [], [], [], []
    
    for i in range(B):
        p = batch_props[i] 
        origin_lps = np.array(p['origin'])
        spacing = p['ori_space']
        ori_size = p['ori_size']
        
        d_lps = np.array(p['direction'])
        if d_lps.shape == (9,):
            d_lps = d_lps.reshape(3, 3)
            
        flip_mat = np.diag([-1, -1, 1]).astype(np.float32)
        d_ras = flip_mat @ d_lps
        origin_ras = flip_mat @ origin_lps
        
        origins.append(origin_ras)
        spacings.append(spacing)
        ori_sizes.append(ori_size)
        directions.append(d_ras)
        
    origins = torch.tensor(np.stack(origins), device=device).float()
    spacings = torch.tensor(np.stack(spacings), device=device).float()
    ori_sizes = torch.tensor(np.stack(ori_sizes), device=device).float()
    directions = torch.tensor(np.stack(directions), device=device).float()
    points_vox_xyz = torch.stack([
        pred_3d_vox[:, :, 2], 
        pred_3d_vox[:, :, 1], 
        pred_3d_vox[:, :, 0]  
    ], dim=-1)

    points_scaled = points_vox_xyz * spacings.unsqueeze(1)
    points_rotated = torch.bmm(directions, points_scaled.transpose(1, 2)).transpose(1, 2)
    points_ras_xyz = origins.unsqueeze(1) + points_rotated 
    
    center_vox_xyz = (ori_sizes - 1) / 2.0 
    center_scaled = center_vox_xyz * spacings 
    center_rotated = torch.bmm(directions, center_scaled.unsqueeze(-1)).squeeze(-1) 
    volume_center = origins + center_rotated 
    
    points_centered = points_ras_xyz - volume_center.unsqueeze(1)
    
    if shifts is not None:
        if shifts.device != device:
            shifts = shifts.to(device)
        shifts_ras = shifts.clone()
        shifts_ras[:, 0] = -shifts_ras[:, 0]
        shifts_ras[:, 1] = -shifts_ras[:, 1]
        points_centered = points_centered + shifts_ras.unsqueeze(1)
        
    return points_centered

def calculate_triangulation_mtre(pts_2d_ap, pts_2d_lat, pts_3d_gt, K, E_ap, E_lat):
    if len(pts_2d_ap) == 0 or len(pts_2d_lat) == 0:
        return np.nan, [], None

    P_ap = np.dot(K, E_ap[:3, :])
    P_lat = np.dot(K, E_lat[:3, :])

    pts_2d_ap_t = pts_2d_ap.T.astype(np.float64)
    pts_2d_lat_t = pts_2d_lat.T.astype(np.float64)

    points_4d_homo = cv2.triangulatePoints(P_ap, P_lat, pts_2d_ap_t, pts_2d_lat_t)

    points_3d_pred = (points_4d_homo[:3, :] / points_4d_homo[3, :]).T

    if isinstance(pts_3d_gt, torch.Tensor):
        pts_3d_gt = pts_3d_gt.detach().cpu().numpy()
        
    distances = np.linalg.norm(points_3d_pred - pts_3d_gt, axis=1)
    mtre = np.mean(distances)

    return mtre, distances.tolist(), points_3d_pred

def save_image_sitk(tensor, save_path, spacing=(1.0, 1.0, 1.0), origin=(0, 0, 0), direction=None):
    import numpy as np
    import torch
    import SimpleITK as sitk

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()

    if tensor.ndim == 4:
        tensor = tensor[0]

    sitk_image = sitk.GetImageFromArray(tensor)

    if spacing is not None:
        spacing = [float(s) for s in spacing]
        sitk_image.SetSpacing(spacing)
        
    if origin is not None:
        origin = [float(o) for o in origin]
        sitk_image.SetOrigin(origin)
        
    if direction is not None:
        if isinstance(direction, torch.Tensor):
            direction = direction.flatten().tolist()
        elif isinstance(direction, np.ndarray):
            direction = direction.flatten().tolist()
            
        direction = [float(d) for d in direction]
        sitk_image.SetDirection(direction)

    sitk.WriteImage(sitk_image, save_path)

def validate(args, input_size, model, ValLoader, num_classes, engine, input_size_2d=None, writer=None, epoch=None):
    fold = 'fold0'
    mode_s = 'rl'
    fname_ap = f'results/{mode_s}/{fold}/ap_error.txt'
    fname_lat = f'results/{mode_s}/{fold}/lat_error.txt'
    fname_ct = f'results/{mode_s}/{fold}/ct_error.txt'  
    fname_all = f'results/{mode_s}/{fold}/all_error.txt'
    con_ap = f'results/{mode_s}/{fold}/con_ap_error.txt'
    con_lat = f'results/{mode_s}/{fold}/con_lat_error.txt'
    fname_id = f'results/{mode_s}/{fold}/id_error.txt'
    fname_mtre = f'results/{mode_s}/{fold}/mtre_error.txt'
    
    mode = args.mode
    list_3d_gt, list_3d_pred_no_rl, list_3d_pred = [], [], []
    list_2d_ap_gt, list_2d_ap_pred_no_rl, list_2d_ap_pred = [], [], []
    list_2d_lat_gt, list_2d_lat_pred_no_rl, list_2d_lat_pred = [], [], []
    valid_shapes = None
    device = next(model[0].parameters()).device
    drr_projector = DiffDRR_Projector(sdd=1000.0, height=256, delx=0.8, device=device)
    K_np = drr_projector.K.cpu().numpy()
    E_ap_np = drr_projector.view_matrix_ap.cpu().numpy()
    E_lat_np = drr_projector.view_matrix_lat.cpu().numpy()
    triangulation_errors = [] 
    model[0].eval()

    for index, batch in enumerate(ValLoader):
        r_2d_ap = batch.get('r_2d_ap')
        r_2d_lat = batch.get('r_2d_lat')
        r_3d = batch.get('r_3d')
        ct_oir_size = batch['ct_oir_size']
        
        batch_props = batch.get('prop', None) 
        shifts = batch.get('shifts', None)
        
        image2d_ap, image2d_lat, images_3d = None, None, None
        
        if r_2d_ap is not None:
            image2d_ap = r_2d_ap['image'].cuda(non_blocking=True)
            label2d_ap = r_2d_ap['label'] 

        if r_2d_lat is not None:
            image2d_lat = r_2d_lat['image'].cuda(non_blocking=True)
            label2d_lat = r_2d_lat['label']

        if r_3d is not None:
            images_3d = r_3d["image"].cuda(non_blocking=True)
            labels_3d = r_3d["label"] 
        
        if args.mode in ['fusion']:
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
                
                if args.use_rl_in_test:
                    heatmaps_sig_ap = torch.sigmoid(pred_2d_ap_out)[0].cpu() 
                    keypoints = model[0].refine_keypoints_rl(
                        image2d_ap[0].cpu(), coords_pred, is_3d=False,
                        max_steps=args.rl_test_steps_2d, args=args,
                        heatmap=heatmaps_sig_ap)                        
                    list_2d_ap_pred.append(keypoints)
                                    
            if mode in ['only_2d_lat', 'joint', 'fusion'] and image2d_lat is not None:
                pred_2d_lat_sig = torch.sigmoid(pred_2d_lat_out)
                
                coords_pred = extract_coordinates_2d_unbiased(pred_2d_lat_sig[0].cpu().numpy())
                coords_gt = extract_coordinates_2d_unbiased(label2d_lat[0].cpu().numpy())
                
                list_2d_lat_pred_no_rl.append(coords_pred) 
                list_2d_lat_gt.append(coords_gt)
                
                if args.use_rl_in_test:
                    heatmaps_sig_lat = torch.sigmoid(pred_2d_lat_out)[0].cpu()
                    keypoints_lat = model[0].refine_keypoints_rl(
                        image2d_lat[0].cpu(), coords_pred, is_3d=False,
                        max_steps=args.rl_test_steps_2d, args=args,
                        heatmap=heatmaps_sig_lat)
                    list_2d_lat_pred.append(keypoints_lat)

            if mode in ['only_3d', 'fusion', 'joint'] and images_3d is not None:
                pred_3d_sig = torch.sigmoid(pred_3d_out)
                
                coords_pred = extract_coordinates_unbiased(pred_3d_sig[0].detach().cpu().numpy())
                coords_gt = extract_coordinates_unbiased(labels_3d[0].detach().cpu().numpy())
                
                list_3d_pred_no_rl.append(coords_pred)
                list_3d_gt.append(coords_gt)
                list_3d_pred.append(coords_pred)
                
                if batch_props is not None:
                    coords_gt_tensor = torch.tensor(coords_gt, device=device, dtype=torch.float32).unsqueeze(0)
                    pts_3d_world = convert_voxel_to_world(
                        coords_gt_tensor, 
                        batch_props=[batch_props] if isinstance(batch_props, dict) else batch_props, 
                        device=device, 
                        shifts=shifts
                    )
                    
                    pts_3d_world = pts_3d_world.squeeze(0)

                    if image2d_ap is not None:
                        pts_2d_ap = list_2d_ap_pred[-1] if args.use_rl_in_test else list_2d_ap_pred_no_rl[-1]
                        pts_2d_ap = np.array(pts_2d_ap, dtype=np.float32)
                        pts_2d_ap_pnp = np.zeros_like(pts_2d_ap)
                        pts_2d_ap_pnp[:, 0] = 255.0 - pts_2d_ap[:, 1]  
                        pts_2d_ap_pnp[:, 1] = 255.0 - pts_2d_ap[:, 0]  

                    if image2d_lat is not None:
                        pts_2d_lat = list_2d_lat_pred[-1] if args.use_rl_in_test else list_2d_lat_pred_no_rl[-1]
                        pts_2d_lat = np.array(pts_2d_lat, dtype=np.float32)
                        pts_2d_lat_pnp = np.zeros_like(pts_2d_lat)
                        pts_2d_lat_pnp[:, 0] = 255.0 - pts_2d_lat[:, 1]  # X = 255 - col
                        pts_2d_lat_pnp[:, 1] = 255.0 - pts_2d_lat[:, 0]  # Y = 256 - row                  
                    
                    if pts_2d_ap_pnp is not None and pts_2d_lat_pnp is not None:
                        mtre, dists, pred_3d = calculate_triangulation_mtre(
                            pts_2d_ap_pnp, 
                            pts_2d_lat_pnp, 
                            pts_3d_world, 
                            K_np, E_ap_np, E_lat_np
                        )
                        if not np.isnan(mtre):
                            triangulation_errors.extend(dists)
                


    metrics_all = {'MRE_mean': 1000.0} 
    metrics_all_RL = {'MRE_mean': 1000.0} 
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

        errors_list_RL = []
        id_count_total_RL = 0
        landmarks_total_RL = 0
        
        fid = open(fname_id, 'w')
        
        if has_ap:
            m_ap = compute_metrics(list_2d_ap_gt, list_2d_ap_pred_no_rl, norm_factor=50)
            print_modal_metrics("AP", m_ap)
            errors_list.append(m_ap['errors'])
            id_count_total += m_ap['id_count']
            landmarks_total += m_ap['total_landmarks']

            if args.use_rl_in_test:
                metrics_2d_ap_rl = compute_metrics(list_2d_ap_gt, list_2d_ap_pred, norm_factor=50)
                print_modal_metrics("AP_RL", metrics_2d_ap_rl)
                errors_list_RL.append(metrics_2d_ap_rl['errors'])
                id_count_total_RL += metrics_2d_ap_rl['id_count']
                landmarks_total_RL += metrics_2d_ap_rl['total_landmarks']
                with open(fname_ap, 'w') as f:
                    for error in metrics_2d_ap_rl['errors']:
                        f.write(f"{error}\n")
                fid.write(f"ap:{metrics_2d_ap_rl['id_count'], metrics_2d_ap_rl['total_landmarks']}\n")

        if has_lat:
            m_lat = compute_metrics(list_2d_lat_gt, list_2d_lat_pred_no_rl, norm_factor=50)
            print_modal_metrics("LAT", m_lat)
            errors_list.append(m_lat['errors'])
            id_count_total += m_lat['id_count']
            landmarks_total += m_lat['total_landmarks']

            if args.use_rl_in_test:
                metrics_2d_lat_rl = compute_metrics(list_2d_lat_gt, list_2d_lat_pred, norm_factor=50)
                print_modal_metrics("LAT_RL", metrics_2d_lat_rl)
                errors_list_RL.append(metrics_2d_lat_rl['errors'])
                id_count_total_RL += metrics_2d_lat_rl['id_count']
                landmarks_total_RL += metrics_2d_lat_rl['total_landmarks']
                with open(fname_lat, 'w') as f:
                    for error in metrics_2d_lat_rl['errors']:
                        f.write(f"{error}\n")
                fid.write(f"lat:{metrics_2d_lat_rl['id_count'], metrics_2d_lat_rl['total_landmarks']}\n")
                
        
        if has_3d:
            m_3d = compute_metrics(list_3d_gt, list_3d_pred_no_rl, norm_factor=30)
            print_modal_metrics("CT", m_3d)
            errors_list.append(m_3d['errors'])
            id_count_total += m_3d['id_count']
            landmarks_total += m_3d['total_landmarks']


            if args.use_rl_in_test:
                metrics_3d_rl = compute_metrics(list_3d_gt, list_3d_pred, norm_factor=30)
                print_modal_metrics("CT_RL", metrics_3d_rl)
                errors_list_RL.append(metrics_3d_rl['errors'])
                id_count_total_RL += metrics_3d_rl['id_count']
                landmarks_total_RL += metrics_3d_rl['total_landmarks']
                with open(fname_ct, 'w') as f:
                    for error in metrics_3d_rl['errors']:
                        f.write(f"{error}\n")
                fid.write(f"ct:{metrics_3d_rl['id_count'], metrics_3d_rl['total_landmarks']}\n")
        
        if errors_list:
            errors_all = np.concatenate(errors_list, axis=0)
            metrics_all = summarize_errors(errors_all, norm_factor=30, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
            metrics_all['ID_rate'] = comput_ID(id_count_total, landmarks_total)

            if args.use_rl_in_test and errors_list_RL:
                errors_all_RL = np.concatenate(errors_list_RL, axis=0)
                metrics_all_RL = summarize_errors(errors_all_RL, norm_factor=30, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
                metrics_all_RL['ID_rate'] = comput_ID(id_count_total_RL, landmarks_total_RL)
                with open(fname_all, 'w') as f:
                    for error in errors_all_RL:
                        f.write(f"{error}\n")
            fid.write(f"all:{id_count_total_RL}, {landmarks_total_RL}\n")
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

    if args.use_rl_in_test and 'MRE_mean' in metrics_all_RL and metrics_all_RL['MRE_mean'] != 1000.0:
        print(f"[{mode.upper()}] Final Results RL:")
        print(f"MRE_all: {metrics_all_RL['MRE_mean']:.3f}±{metrics_all_RL.get('MRE_std', 0):.3f}")
        print(f"NME_all: {metrics_all_RL['NME_mean']:.3f}±{metrics_all_RL.get('NME_std', 0):.3f}")
        print(f"SR@2mm:  {metrics_all_RL['SR@2.0_mean']:.3f}±{metrics_all_RL.get('SR@2.0_std', 0):.3f}")
        print(f"SR@4mm:  {metrics_all_RL['SR@4.0_mean']:.3f}±{metrics_all_RL.get('SR@4.0_std', 0):.3f}")
        print(f"AUC={metrics_all_RL['AUC_mean']:.3f}",)
        print(f"ID_rate: {metrics_all_RL.get('ID_rate', 0):.3f}")

   
    if len(triangulation_errors) > 0:
        tri_mtre_mean = np.mean(triangulation_errors)
        tri_mtre_std = np.std(triangulation_errors)
        print(f"--- Triangulation (2D+2D -> 3D) ---")
        print(f"mTRE (3D Distance): {tri_mtre_mean:.3f} mm ± {tri_mtre_std:.3f} mm")
        with open(fname_mtre, 'w') as f:
            for error in triangulation_errors:
                f.write(f"{error}\n")
    
    metrics_re = {'total': metrics_all['MRE_mean']}
    metrics_re_RL = {'total': metrics_all_RL['MRE_mean']}

    if mode in ['joint', 'fusion']:
        if has_ap:
            metrics_re['ap'] = m_ap['MRE_mean']
            if args.use_rl_in_test:
                metrics_re_RL['ap'] = metrics_2d_ap_rl['MRE_mean']
        if has_lat:
            metrics_re['lat'] = m_lat['MRE_mean']
            if args.use_rl_in_test:
                metrics_re_RL['lat'] = metrics_2d_lat_rl['MRE_mean']
        if has_3d:
            metrics_re['ct'] = m_3d['MRE_mean']
            if args.use_rl_in_test:
                metrics_re_RL['ct'] = metrics_3d_rl['MRE_mean']

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
            
    if args.use_rl_in_test:
        return metrics_re_RL
    else:
        return metrics_re

def draw_points_2d(image_tensor, pts_gt, pts_pred, save_path, flip_ud=True, proj=False):
    img_np = image_tensor.cpu().numpy()
    img_np = np.squeeze(img_np)
    if img_np.ndim == 3:
        if img_np.shape[0] < img_np.shape[-1]:
            img_np = img_np[0]
        else:
            img_np = img_np[..., 0]
    H, W = img_np.shape
    if flip_ud:
        img_np = np.flipud(img_np)
            
    img_np = ((img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8) * 255).astype(np.uint8)
    img_color = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    for i, (pt_gt, pt_pred) in enumerate(zip(pts_gt, pts_pred)):
        x_gt, y_gt = float(pt_gt[0]), float(pt_gt[1])
        x_pred, y_pred = float(pt_pred[0]), float(pt_pred[1])
        if flip_ud:
            y_gt = (H - 1) - y_gt
            y_pred = (H - 1) - y_pred
            
        center_gt = (int(x_gt), int(y_gt))
        center_pred = (int(x_pred), int(y_pred))
        
        cv2.line(img_color, center_gt, center_pred, color=(255, 200, 50), thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(img_color, center_gt, radius=3, color=(8, 186, 255), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        cv2.drawMarker(img_color, center_pred, color=(0, 0, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=5, thickness=1, line_type=cv2.LINE_AA)
        
    cv2.imwrite(save_path, img_color)
           
def validate_vision(args, input_size, model, ValLoader, num_classes, engine, input_size_2d=None, writer=None, epoch=None):
    fold = 'fold2'
    mode_s = 'rl'
    base_dir = f'results/{mode_s}/{fold}'
    
    fname_ap = f'{base_dir}/ap_error.txt'
    fname_lat = f'{base_dir}/lat_error.txt'
    fname_ct = f'{base_dir}/ct_error.txt'
    fname_all = f'{base_dir}/all_error.txt'
    con_ap = f'{base_dir}/con_ap_error.txt'
    con_lat = f'{base_dir}/con_lat_error.txt'
    fname_id = f'{base_dir}/id_error.txt'
    fname_mtre = f'{base_dir}/mtre_error.txt'

    vis_dir = f'{base_dir}/vis'
    args.visualize = True
    if getattr(args, 'visualize', False):
        os.makedirs(vis_dir, exist_ok=True)
    
    mode = args.mode
    
    list_3d_gt, list_3d_pred_no_rl, list_3d_pred = [], [], []
    list_2d_ap_gt, list_2d_ap_pred_no_rl, list_2d_ap_pred = [], [], []
    list_2d_lat_gt, list_2d_lat_pred_no_rl, list_2d_lat_pred = [], [], []
    valid_shapes = None
    
    device = next(model[0].parameters()).device
    drr_projector = DiffDRR_Projector(sdd=1000.0, height=256, delx=0.8, device=device)
    K_np = drr_projector.K.cpu().numpy()
    E_ap_np = drr_projector.view_matrix_ap.cpu().numpy()
    E_lat_np = drr_projector.view_matrix_lat.cpu().numpy()
    
    pnp_errors_ap = {'rot': [], 'trans': []}
    pnp_errors_lat = {'rot': [], 'trans': []}
    
    triangulation_errors = [] 

    model[0].eval()
    def save_fcsv(pts_world, save_path, prefix="pt", is_lps_to_ras=True):
            with open(save_path, 'w') as f:
                f.write("# Markups fiducial file version = 4.11\n")
                f.write("# CoordinateSystem = RAS\n") 
                f.write("# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n")
                
                for i, pt in enumerate(pts_world):
                    x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
                    if is_lps_to_ras:
                        x = -x
                        y = -y
                        
                    f.write(f"vtkMRMLMarkupsFiducialNode_{i},{x},{y},{z},0,0,0,1,1,1,0,,,\n")
            

    for index, batch in enumerate(ValLoader):
        r_2d_ap = batch.get('r_2d_ap')
        r_2d_lat = batch.get('r_2d_lat')
        r_3d = batch.get('r_3d')
        ct_oir_size = batch['ct_oir_size']
        
        batch_props = batch.get('prop', None) 
        shifts = batch.get('shifts', None)
        
        image2d_ap, image2d_lat, images_3d = None, None, None
        
        if r_2d_ap is not None:
            image2d_ap = r_2d_ap['image'].cuda(non_blocking=True)
            label2d_ap = r_2d_ap['label'] 

        if r_2d_lat is not None:
            image2d_lat = r_2d_lat['image'].cuda(non_blocking=True)
            label2d_lat = r_2d_lat['label']

        if r_3d is not None:
            images_3d = r_3d["image"].cuda(non_blocking=True)
            labels_3d = r_3d["label"] 
        
        if args.mode in ['fusion']:
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
                
                if args.use_rl_in_test:
                    heatmaps_sig_ap = torch.sigmoid(pred_2d_ap_out)[0].cpu()
                    keypoints = model[0].refine_keypoints_rl(
                        image2d_ap[0].cpu(), coords_pred, is_3d=False,
                        max_steps=args.rl_test_steps_2d, args=args,
                        heatmap=heatmaps_sig_ap)
                    list_2d_ap_pred.append(keypoints)
                
                if getattr(args, 'visualize', False):
                    vis_pred_ap = list_2d_ap_pred[-1] if args.use_rl_in_test else list_2d_ap_pred_no_rl[-1]
                    draw_points_2d(image2d_ap[0], coords_gt, vis_pred_ap, os.path.join(vis_dir, f'ap_{batch.get("name", [str(index)])[0]}.png'), flip_ud=True)
                                    
            if mode in ['only_2d_lat', 'joint', 'fusion'] and image2d_lat is not None:
                pred_2d_lat_sig = torch.sigmoid(pred_2d_lat_out)
                
                coords_pred = extract_coordinates_2d_unbiased(pred_2d_lat_sig[0].cpu().numpy())
                coords_gt = extract_coordinates_2d_unbiased(label2d_lat[0].cpu().numpy())
                
                list_2d_lat_pred_no_rl.append(coords_pred) 
                list_2d_lat_gt.append(coords_gt)
                
                if args.use_rl_in_test:
                    heatmaps_sig_lat = torch.sigmoid(pred_2d_lat_out)[0].cpu()
                    keypoints_lat = model[0].refine_keypoints_rl(
                        image2d_lat[0].cpu(), coords_pred, is_3d=False,
                        max_steps=args.rl_test_steps_2d, args=args,
                        heatmap=heatmaps_sig_lat)
                    list_2d_lat_pred.append(keypoints_lat)

                if getattr(args, 'visualize', False):
                    vis_pred_lat = list_2d_lat_pred[-1] if args.use_rl_in_test else list_2d_lat_pred_no_rl[-1]
                    draw_points_2d(image2d_lat[0], coords_gt, vis_pred_lat, os.path.join(vis_dir, f'lat_{batch.get("name", [str(index)])[0]}.png'), flip_ud=True)

            if mode in ['only_3d', 'fusion', 'joint'] and images_3d is not None:
                pred_3d_sig = torch.sigmoid(pred_3d_out)
                
                coords_pred = extract_coordinates_unbiased(pred_3d_sig[0].detach().cpu().numpy())
                coords_gt = extract_coordinates_unbiased(labels_3d[0].detach().cpu().numpy())
                
                list_3d_pred_no_rl.append(coords_pred)
                list_3d_gt.append(coords_gt)
                list_3d_pred.append(coords_pred)


                if getattr(args, 'visualize', False):
                    vis_pred_ct = list_3d_pred[-1] if args.use_rl_in_test else list_3d_pred_no_rl[-1]
                    
                    img_3d_np = images_3d[0].detach().cpu().numpy()
                    img_3d_np = np.squeeze(img_3d_np)
                    
                    if valid_shapes is not None:
                        valid_d, valid_h, valid_w = valid_shapes[0].cpu().numpy().astype(int)
                        img_3d_unpad = img_3d_np[:valid_d, :valid_h, :valid_w]
                    else:
                        img_3d_unpad = img_3d_np
                    
                    affine = np.eye(4)
                    if batch_props is not None:
                        p = batch_props[0] if isinstance(batch_props, list) else batch_props
                        if 'affine' in p:
                            affine = p['affine']
                            if isinstance(affine, torch.Tensor):
                                affine = affine.cpu().numpy()
                        else:
                            spacing = p.get('ori_space', [1.0, 1.0, 1.0])
                            origin = p.get('origin', [0.0, 0.0, 0.0])
                            direction = p.get('direction', np.eye(3))
                            if isinstance(direction, (list, np.ndarray)) and len(np.array(direction).flatten()) == 9:
                                direction = np.array(direction).reshape(3, 3)
                            affine[:3, :3] = np.array(direction) * np.array(spacing)
                            affine[:3, 3] = np.array(origin)

                    base_name = batch.get("name", [str(index)])[0]                 
                    save_image_sitk(img_3d_unpad, os.path.join(vis_dir, f'ct_img_{base_name}.nii.gz'), spacing=batch_props[0]['ori_space'],origin=batch_props[0]['origin'],direction=batch_props[0]['direction'])

                    if batch_props is not None:
                        p = batch_props[0] if isinstance(batch_props, list) else batch_props
                        origin_lps = np.array(p['origin'], dtype=float)
                        spacing = np.array(p['ori_space'], dtype=float)
                        direction_lps = np.array(p['direction'], dtype=float)
                        if direction_lps.shape == (9,):
                            direction_lps = direction_lps.reshape(3, 3)
                            
                        current_shift = shifts[0].cpu().numpy() if shifts is not None else np.array([0.0, 0.0, 0.0])

                        def vox_to_slicer_lps(vox_zyx):
                            if len(vox_zyx) == 0:
                                return []
                            vox_xyz = np.array(vox_zyx)[:, ::-1].astype(float)
                            scaled = vox_xyz * spacing
                            rotated = (direction_lps @ scaled.T).T
                            lps_pts = origin_lps + rotated + current_shift
                            return lps_pts.tolist()

                        pts_3d_world_gt = vox_to_slicer_lps(coords_gt)
                        pts_3d_world_pred = vox_to_slicer_lps(vis_pred_ct)
                        
                        save_fcsv(pts_3d_world_gt, os.path.join(vis_dir, f'gt_{base_name}.fcsv'), is_lps_to_ras=True)
                        save_fcsv(pts_3d_world_pred, os.path.join(vis_dir, f'pred_{base_name}.fcsv'), is_lps_to_ras=True)
                
                if batch_props is not None:
                    coords_gt_tensor = torch.tensor(coords_gt, device=device, dtype=torch.float32).unsqueeze(0)
                    pts_3d_world = convert_voxel_to_world(
                        coords_gt_tensor, 
                        batch_props=[batch_props] if isinstance(batch_props, dict) else batch_props, 
                        device=device, 
                        shifts=shifts
                    )
                    
                    pts_3d_world = pts_3d_world.squeeze(0)

                    if image2d_ap is not None:
                        pts_2d_ap = list_2d_ap_pred[-1] if args.use_rl_in_test else list_2d_ap_pred_no_rl[-1]
                        pts_2d_ap = np.array(pts_2d_ap, dtype=np.float32)
                        
                        pts_2d_ap_pnp = np.zeros_like(pts_2d_ap)
                        pts_2d_ap_pnp[:, 0] = 255.0 - pts_2d_ap[:, 1] 
                        pts_2d_ap_pnp[:, 1] = 255.0 - pts_2d_ap[:, 0]

                    if image2d_lat is not None:
                        pts_2d_lat = list_2d_lat_pred[-1] if args.use_rl_in_test else list_2d_lat_pred_no_rl[-1]
                        pts_2d_lat = np.array(pts_2d_lat, dtype=np.float32)
                        
                        pts_2d_lat_pnp = np.zeros_like(pts_2d_lat)
                        pts_2d_lat_pnp[:, 0] = 255.0 - pts_2d_lat[:, 1]  # X = 255 - col
                        pts_2d_lat_pnp[:, 1] = 255.0 - pts_2d_lat[:, 0]  # Y = 256 - row
                    
                    if pts_2d_ap_pnp is not None and pts_2d_lat_pnp is not None:
                        mtre, dists, pred_3d = calculate_triangulation_mtre(
                            pts_2d_ap_pnp, 
                            pts_2d_lat_pnp, 
                            pts_3d_world, 
                            K_np, E_ap_np, E_lat_np
                        )
                        if not np.isnan(mtre):
                            triangulation_errors.extend(dists)
                
    metrics_all = {'MRE_mean': 1000.0} 
    metrics_all_RL = {'MRE_mean': 1000.0} 
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

        errors_list_RL = []
        id_count_total_RL = 0
        landmarks_total_RL = 0
        
        fid = open(fname_id, 'w')
        
        if has_ap:
            m_ap = compute_metrics(list_2d_ap_gt, list_2d_ap_pred_no_rl, norm_factor=50)
            print_modal_metrics("AP", m_ap)
            errors_list.append(m_ap['errors'])
            id_count_total += m_ap['id_count']
            landmarks_total += m_ap['total_landmarks']

            if args.use_rl_in_test:
                metrics_2d_ap_rl = compute_metrics(list_2d_ap_gt, list_2d_ap_pred, norm_factor=50)
                print_modal_metrics("AP_RL", metrics_2d_ap_rl)
                errors_list_RL.append(metrics_2d_ap_rl['errors'])
                id_count_total_RL += metrics_2d_ap_rl['id_count']
                landmarks_total_RL += metrics_2d_ap_rl['total_landmarks']
                with open(fname_ap, 'w') as f:
                    for error in metrics_2d_ap_rl['errors']:
                        f.write(f"{error}\n")
                fid.write(f"ap:{metrics_2d_ap_rl['id_count'], metrics_2d_ap_rl['total_landmarks']}\n")

        if has_lat:
            m_lat = compute_metrics(list_2d_lat_gt, list_2d_lat_pred_no_rl, norm_factor=50)
            print_modal_metrics("LAT", m_lat)
            errors_list.append(m_lat['errors'])
            id_count_total += m_lat['id_count']
            landmarks_total += m_lat['total_landmarks']

            if args.use_rl_in_test:
                metrics_2d_lat_rl = compute_metrics(list_2d_lat_gt, list_2d_lat_pred, norm_factor=50)
                print_modal_metrics("LAT_RL", metrics_2d_lat_rl)
                errors_list_RL.append(metrics_2d_lat_rl['errors'])
                id_count_total_RL += metrics_2d_lat_rl['id_count']
                landmarks_total_RL += metrics_2d_lat_rl['total_landmarks']
                with open(fname_lat, 'w') as f:
                    for error in metrics_2d_lat_rl['errors']:
                        f.write(f"{error}\n")
                fid.write(f"lat:{metrics_2d_lat_rl['id_count'], metrics_2d_lat_rl['total_landmarks']}\n")
                
        if has_3d:
            m_3d = compute_metrics(list_3d_gt, list_3d_pred_no_rl, norm_factor=30)
            print_modal_metrics("CT", m_3d)
            errors_list.append(m_3d['errors'])
            id_count_total += m_3d['id_count']
            landmarks_total += m_3d['total_landmarks']

            if args.use_rl_in_test:
                metrics_3d_rl = compute_metrics(list_3d_gt, list_3d_pred, norm_factor=30)
                print_modal_metrics("CT_RL", metrics_3d_rl)
                errors_list_RL.append(metrics_3d_rl['errors'])
                id_count_total_RL += metrics_3d_rl['id_count']
                landmarks_total_RL += metrics_3d_rl['total_landmarks']
                with open(fname_ct, 'w') as f:
                    for error in metrics_3d_rl['errors']:
                        f.write(f"{error}\n")
                fid.write(f"ct:{metrics_3d_rl['id_count'], metrics_3d_rl['total_landmarks']}\n")
        
        if errors_list:
            errors_all = np.concatenate(errors_list, axis=0)
            metrics_all = summarize_errors(errors_all, norm_factor=30, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
            metrics_all['ID_rate'] = comput_ID(id_count_total, landmarks_total)

            if args.use_rl_in_test and errors_list_RL:
                errors_all_RL = np.concatenate(errors_list_RL, axis=0)
                metrics_all_RL = summarize_errors(errors_all_RL, norm_factor=30, sr_thresholds=(2.0, 4.0), auc_max_threshold=10.0)
                metrics_all_RL['ID_rate'] = comput_ID(id_count_total_RL, landmarks_total_RL)
                with open(fname_all, 'w') as f:
                    for error in errors_all_RL:
                        f.write(f"{error}\n")
            fid.write(f"all:{id_count_total_RL}, {landmarks_total_RL}\n")
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

    if args.use_rl_in_test and 'MRE_mean' in metrics_all_RL and metrics_all_RL['MRE_mean'] != 1000.0:
        print(f"[{mode.upper()}] Final Results RL:")
        print(f"MRE_all: {metrics_all_RL['MRE_mean']:.3f}±{metrics_all_RL.get('MRE_std', 0):.3f}")
        print(f"NME_all: {metrics_all_RL['NME_mean']:.3f}±{metrics_all_RL.get('NME_std', 0):.3f}")
        print(f"SR@2mm:  {metrics_all_RL['SR@2.0_mean']:.3f}±{metrics_all_RL.get('SR@2.0_std', 0):.3f}")
        print(f"SR@4mm:  {metrics_all_RL['SR@4.0_mean']:.3f}±{metrics_all_RL.get('SR@4.0_std', 0):.3f}")
        print(f"AUC={metrics_all_RL['AUC_mean']:.3f}",)
        print(f"ID_rate: {metrics_all_RL.get('ID_rate', 0):.3f}")
        
    if len(triangulation_errors) > 0:
        tri_mtre_mean = np.mean(triangulation_errors)
        tri_mtre_std = np.std(triangulation_errors)
        print(f"--- Triangulation (2D+2D -> 3D) ---")
        print(f"mTRE (3D Distance): {tri_mtre_mean:.3f} mm ± {tri_mtre_std:.3f} mm")
        with open(fname_mtre, 'w') as f:
            for error in triangulation_errors:
                f.write(f"{error}\n")
    
    metrics_re = {'total': metrics_all['MRE_mean']}
    metrics_re_RL = {'total': metrics_all_RL['MRE_mean']}

    if mode in ['joint', 'fusion']:
        if has_ap:
            metrics_re['ap'] = m_ap['MRE_mean']
            if args.use_rl_in_test:
                metrics_re_RL['ap'] = metrics_2d_ap_rl['MRE_mean']
        if has_lat:
            metrics_re['lat'] = m_lat['MRE_mean']
            if args.use_rl_in_test:
                metrics_re_RL['lat'] = metrics_2d_lat_rl['MRE_mean']
        if has_3d:
            metrics_re['ct'] = m_3d['MRE_mean']
            if args.use_rl_in_test:
                metrics_re_RL['ct'] = metrics_3d_rl['MRE_mean']

    if mode in ['joint','fusion']:
        device = images_3d.device
        fusion_metrics = val_ccd_fusion(
                args=args, 
                model=model[0], 
                ValLoader=ValLoader, 
                device=device, 
                input_size_3d=(64, 160, 160),
                con_ap = con_ap,
                con_lat = con_lat,
                vision_dir=vis_dir,
            )
            
    if args.use_rl_in_test:
        return metrics_re_RL
    else:
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
        mask = errors < thr          # bool, shape [num_points]
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

def val_ccd_fusion(args, model, ValLoader, device, input_size_3d=(64, 160, 160),con_ap = None, con_lat = None, vision_dir=None):
    drr_projector = DiffDRR_Projector(sdd=1000, height=256, delx=0.8, device=device)
    model.eval()
    print(f">>> Starting CCD Fusion Validation (Strict Alignment Mode with Unbiased Decoding)...")
    dist_ap_list_before = []
    dist_lat_list_before = []
    dist_ap_list_after = []
    dist_lat_list_after = []
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
            
            pred_ap_pix_before = torch.tensor(coords_ap_np, device=device, dtype=torch.float32).unsqueeze(0)
            pred_lat_pix_before = torch.tensor(coords_lat_np, device=device, dtype=torch.float32).unsqueeze(0)
            
            pred_ap_pix_after = pred_ap_pix_before.clone()
            pred_lat_pix_after = pred_lat_pix_before.clone()

            if getattr(args, 'use_rl_in_test', False):
                init_coords_ap = pred_ap_pix_before[0].cpu().numpy().tolist()
                refined_ap = model.refine_keypoints_rl(
                    images_2d_ap[0].cpu(), init_coords_ap, is_3d=False, 
                    max_steps=args.rl_test_steps_2d, args=args, heatmap=heatmaps_ap[0]
                )
                pred_ap_pix_after = torch.tensor(refined_ap, device=device, dtype=torch.float32).unsqueeze(0)

                init_coords_lat = pred_lat_pix_before[0].cpu().numpy().tolist()
                refined_lat = model.refine_keypoints_rl(
                    images_2d_lat[0].cpu(), init_coords_lat, is_3d=False, 
                    max_steps=args.rl_test_steps_2d, args=args, heatmap=heatmaps_lat[0]
                )
                pred_lat_pix_after = torch.tensor(refined_lat, device=device, dtype=torch.float32).unsqueeze(0)

            proj_ap_pix_before, proj_lat_pix_before = drr_projector(coords_3d_vox, batch_props)

            proj_ap_pix_after, proj_lat_pix_after = drr_projector(coords_3d_vox, batch_props)
            
            proj_ap_pix_before_flipped = proj_ap_pix_before.clone()
            proj_ap_pix_before_flipped[..., 1] = 255.0 - proj_ap_pix_before_flipped[..., 1]
            
            proj_lat_pix_before_flipped = proj_lat_pix_before.clone()
            proj_lat_pix_before_flipped[..., 1] = 255.0 - proj_lat_pix_before_flipped[..., 1]

            proj_ap_pix_after_flipped = proj_ap_pix_after.clone()
            proj_ap_pix_after_flipped[..., 1] = 255.0 - proj_ap_pix_after_flipped[..., 1]
            
            proj_lat_pix_after_flipped = proj_lat_pix_after.clone()
            proj_lat_pix_after_flipped[..., 1] = 255.0 - proj_lat_pix_after_flipped[..., 1]
            
            if getattr(args, 'visualize', False):
                draw_points_2d(images_2d_ap[0], proj_ap_pix_after_flipped[0], pred_ap_pix_after[0], os.path.join(vision_dir, f'ap_{batch.get("name", [str(batch_idx)])[0]}_cons.png'), flip_ud=True, proj=True)
                draw_points_2d(images_2d_lat[0], proj_lat_pix_after_flipped[0], pred_lat_pix_after[0], os.path.join(vision_dir, f'lat_{batch.get("name", [str(batch_idx)])[0]}_cons.png'), flip_ud=True, proj=True)

            diff_ap_before = torch.norm(proj_ap_pix_before_flipped - pred_ap_pix_before, dim=-1)
            diff_lat_before = torch.norm(proj_lat_pix_before_flipped - pred_lat_pix_before, dim=-1)
            
            diff_ap_after = torch.norm(proj_ap_pix_after_flipped - pred_ap_pix_after, dim=-1)
            diff_lat_after = torch.norm(proj_lat_pix_after_flipped - pred_lat_pix_after, dim=-1)

            proj_ap_norm = proj_ap_pix_before_flipped / drr_height
            valid_mask_ap = (proj_ap_norm >= 0) & (proj_ap_norm <= 1)
            valid_mask_ap = valid_mask_ap.all(dim=-1) # (B, N)
            
            proj_lat_norm = proj_lat_pix_before_flipped / drr_height
            valid_mask_lat = (proj_lat_norm >= 0) & (proj_lat_norm <= 1)
            valid_mask_lat = valid_mask_lat.all(dim=-1)

            if valid_mask_ap.any():
                dist_ap_list_before.extend(diff_ap_before[valid_mask_ap].cpu().numpy().tolist())
            if valid_mask_lat.any():
                dist_lat_list_before.extend(diff_lat_before[valid_mask_lat].cpu().numpy().tolist())
                
            if valid_mask_ap.any():
                dist_ap_list_after.extend(diff_ap_after[valid_mask_ap].cpu().numpy().tolist())
            if valid_mask_lat.any():
                dist_lat_list_after.extend(diff_lat_after[valid_mask_lat].cpu().numpy().tolist())


    results = {}
    pixel_spacing = 0.48 
    with open(con_ap, 'w') as f:
        for error in dist_ap_list_after:
            f.write(f"{error}\n")
    with open(con_lat, 'w') as f:
        for error in dist_lat_list_after:
            f.write(f"{error}\n")

    mean_ap_dist_before = np.mean(dist_ap_list_before) if len(dist_ap_list_before) > 0 else 0.0
    std_ap_dist_before = np.std(dist_ap_list_before) if len(dist_ap_list_before) > 0 else 0.0
    mean_lat_dist_before = np.mean(dist_lat_list_before) if len(dist_lat_list_before) > 0 else 0.0
    std_lat_dist_before = np.std(dist_lat_list_before) if len(dist_lat_list_before) > 0 else 0.0

    mean_ap_dist_after = np.mean(dist_ap_list_after) if len(dist_ap_list_after) > 0 else 0.0
    std_ap_dist_after = np.std(dist_ap_list_after) if len(dist_ap_list_after) > 0 else 0.0
    mean_lat_dist_after = np.mean(dist_lat_list_after) if len(dist_lat_list_after) > 0 else 0.0
    std_lat_dist_after = np.std(dist_lat_list_after) if len(dist_lat_list_after) > 0 else 0.0
    
    print(f"\n=== CCD Fusion Consistency Metrics (Pixels @ 256x256) ===")
    
    print(f"--- [BEFORE RL Refinement] ---")
    print(f"{'AP Consistency':<20}: {mean_ap_dist_before:.4f}±{std_ap_dist_before:.4f} px")
    print(f"{'Lat Consistency':<20}: {mean_lat_dist_before:.4f}±{std_lat_dist_before:.4f} px")
    print(f"{'AP Cons (mm)':<20}: {mean_ap_dist_before * pixel_spacing:.4f} mm")
    print(f"{'Lat Cons (mm)':<20}: {mean_lat_dist_before * pixel_spacing:.4f} mm")
    
    if getattr(args, 'use_rl_in_test', False):
        print(f"\n--- [AFTER RL Refinement (2D & 3D)] ---")
        print(f"{'AP Consistency':<20}: {mean_ap_dist_after:.4f}±{std_ap_dist_after:.4f} px")
        print(f"{'Lat Consistency':<20}: {mean_lat_dist_after:.4f}±{std_lat_dist_after:.4f} px")
        print(f"{'AP Cons (mm)':<20}: {mean_ap_dist_after * pixel_spacing:.4f} mm")
        print(f"{'Lat Cons (mm)':<20}: {mean_lat_dist_after * pixel_spacing:.4f} mm")
    else:
        print("\n--- [AFTER RL Refinement] ---")
        print("RL refinement was disabled (results are identical to BEFORE).")
    
    results['cons_ap_pix'] = mean_ap_dist_after
    results['cons_lat_pix'] = mean_lat_dist_after
    results['total_consistency'] = mean_ap_dist_after + mean_lat_dist_after
    
    results['cons_ap_pix_before'] = mean_ap_dist_before
    results['cons_lat_pix_before'] = mean_lat_dist_before

    return results

def get_arguments():
    """Parse all the arguments provided from the CLI.
    Returns:
      A list of parsed arguments.
    """
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

    
    ### rl
    parser.add_argument("--use_rl", type=str2bool, default=True)
    parser.add_argument("--rl_gamma", type=float, default=0.95)
    parser.add_argument("--rl_epsilon", type=float, default=0.5)
    parser.add_argument("--rl_lambda", type=float, default=0.1)
    parser.add_argument("--rl_episodes_per_batch", type=int, default=2)
    parser.add_argument("--rl_max_steps", type=int, default=50)
    parser.add_argument("--rl_buffer_size", type=int, default=3000)
    parser.add_argument("--rl_batch_size", type=int, default=128)
    parser.add_argument("--rl_epsilon_decay", type=bool, default=True)
    parser.add_argument("--rl_epsilon_min", type=float, default=0.05)
    parser.add_argument("--rl_reward_scale", type=float, default=1.0 / 192.0) 
    parser.add_argument("--use_rl_in_test", type=str2bool, default=True)
    parser.add_argument("--rl_max_steps_3d", type=int, default=15)
    parser.add_argument("--rl_max_steps_2d", type=int, default=10)
    parser.add_argument("--rl_test_steps_3d", type=int, default=15)
    parser.add_argument("--rl_test_steps_2d", type=int, default=10)
    parser.add_argument("--rl_min_test_steps_3d", type=int, default=5)
    parser.add_argument("--rl_min_test_steps_2d", type=int, default=3)
    parser.add_argument("--rl_start_epoch", type=int, default=0) 
    parser.add_argument("--rl_update_freq", type=int, default=5)

    ### sturcture prior
    parser.add_argument("--is_prior", type=str2bool, default=False)
    parser.add_argument("--mode", type=str, default='fusion', choices=['only_2d_ap', 'only_2d_lat','only_3d', 'joint', 'fusion','cce'], help="Training mode")

    return parser

def run_batched_rl_episodes(envs, states, q_net, epsilon, max_steps, replay_buffer, epoch_rewards, epoch_qvalues):
    if not envs:
        return
        
    active_indices = list(range(len(envs)))
    local_steps = [0] * len(envs)
    
    episode_transitions = [[] for _ in range(len(envs))]
    episode_rewards_list = [[] for _ in range(len(envs))]
    episode_qvalues_list = [[] for _ in range(len(envs))]

    q_net.eval()

    while active_indices:
        batch_states = torch.stack([states[i] for i in active_indices]).cuda(non_blocking=True)
        with torch.no_grad():
            q_values_batch = q_net(batch_states)
            best_actions = torch.argmax(q_values_batch, dim=1).cpu().numpy()
            avg_qs = q_values_batch.mean(dim=1).cpu().numpy()

        next_active_indices = []
        
        for i, env_idx in enumerate(active_indices):
            env = envs[env_idx]
            
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = best_actions[i]

            episode_qvalues_list[env_idx].append(float(avg_qs[i]))

            next_state, reward, done, _ = env.step(action)

            state_store = states[env_idx]
            next_state_store = next_state
            reward_store = float(reward)

            episode_transitions[env_idx].append((state_store, action, reward_store, next_state_store, done))
            episode_rewards_list[env_idx].append(reward_store)

            states[env_idx] = next_state
            local_steps[env_idx] += 1

            if not done and local_steps[env_idx] < max_steps:
                next_active_indices.append(env_idx)

        active_indices = next_active_indices

    for env_idx in range(len(envs)):
        for trans in episode_transitions[env_idx]:
            replay_buffer.push(*trans)
        if episode_rewards_list[env_idx]:
            epoch_rewards.append(sum(episode_rewards_list[env_idx]) / len(episode_rewards_list[env_idx]))
        if episode_qvalues_list[env_idx]:
            epoch_qvalues.append(sum(episode_qvalues_list[env_idx]) / len(episode_qvalues_list[env_idx]))