import torch
import torch.nn as nn
import numpy as np

class DiffDRR_Projector(nn.Module):
    def __init__(self, sdd=1000.0, height=256, delx=2.0, device='cuda'):
        super().__init__()
        self.sdd = sdd
        self.height = height
        self.delx = delx
        self.device = device
        self.fc = 600.0        
        u0 = height / 2.0
        v0 = height / 2.0 
        f = sdd

        
        self.K = torch.tensor([
            [f / delx, 0.0,      u0],
            [0.0,      f / delx, v0],
            [0.0,      0.0,      1.0]
        ], device=device)
        self.view_matrix_ap = self._build_matrix(view='ap')
        self.view_matrix_lat = self._build_matrix(view='lat')

    def _build_matrix(self, view):
        if view == 'ap':
            
            E = torch.zeros((4, 4), device=self.device)
            
            E[0, 2] = -1.0
            E[1, 0] = 1.0
            E[2, 1] = -1.0
            
            E[2, 3] = self.fc
            
            E[3, 3] = 1.0
            
            return E

        elif view == 'lat':
            E = torch.zeros((4, 4), device=self.device)
            
            # Rotation part
            E[0, 2] = -1.0   # X_cam = -Z_world
            E[1, 1] = 1.0    # Y_cam = Y_world
            E[2, 0] = 1.0    # Z_cam = X_world ...
            
            # Translation part
            E[2, 3] = self.fc  # ... + 600
            
            # Homogeneous
            E[3, 3] = 1.0
            
            return E
            
    def forward(self, pred_3d_norm, batch_props, shifts=None):

        B, N, _ = pred_3d_norm.shape
        
        origins = []
        spacings = []
        ori_sizes = []
        directions = []
        
        for i in range(B):
            p = batch_props[i] 
            origin_lps = np.array(p['origin'])    # (3,) usually (x, y, z)
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
 
        origins = torch.tensor(np.stack(origins), device=self.device).float()       # (B, 3)
        spacings = torch.tensor(np.stack(spacings), device=self.device).float()     # (B, 3)
        ori_sizes = torch.tensor(np.stack(ori_sizes), device=self.device).float()   # (B, 3)
        directions = torch.tensor(np.stack(directions), device=self.device).float() # (B, 3, 3)
        
        crop_size = torch.tensor([64, 160, 160], device=self.device).float()
        
        points_vox_zyx = pred_3d_norm
        
        points_vox_xyz = torch.stack([
            points_vox_zyx[:, :, 2], # X
            points_vox_zyx[:, :, 1], # Y
            points_vox_zyx[:, :, 0]  # Z
        ], dim=-1) # (B, N, 3)

        points_scaled = points_vox_xyz * spacings.unsqueeze(1) # (B, N, 3)
        
        points_rotated = torch.bmm(directions, points_scaled.transpose(1, 2))
        points_rotated = points_rotated.transpose(1, 2)
        points_ras_xyz = origins.unsqueeze(1) + points_rotated # (B, N, 3)
        center_vox_xyz = (ori_sizes-1) / 2.0 
        center_scaled = center_vox_xyz * spacings # (B, 3)
        center_rotated = torch.bmm(directions, center_scaled.unsqueeze(-1)).squeeze(-1) # (B, 3)
        volume_center = origins + center_rotated # (B, 3)
        
        points_centered = points_ras_xyz - volume_center.unsqueeze(1)
        if shifts is not None:
            if shifts.device != self.device:
                shifts = shifts.to(self.device)
            shifts_ras = shifts.clone()
            shifts_ras[:, 0] = -shifts_ras[:, 0] # L -> R
            shifts_ras[:, 1] = -shifts_ras[:, 1] # P -> A
            points_centered = points_centered + shifts_ras.unsqueeze(1)
        proj_ap = self._project_view(points_centered, self.view_matrix_ap)
        proj_lat = self._project_view(points_centered, self.view_matrix_lat)
        return proj_ap, proj_lat

    def _project_view(self, points_world, view_matrix):
        B, N, _ = points_world.shape
        ones = torch.ones(B, N, 1, device=self.device)
        points_homo = torch.cat([points_world, ones], dim=-1) # (B, N, 4)
        extrinsic = view_matrix.t()
        points_cam = torch.matmul(points_homo, extrinsic)    
        points_cam_3 = points_cam[:, :, :3]
        points_pix_homo = torch.matmul(points_cam_3, self.K.t())
        
        z = points_pix_homo[:, :, 2:3]
        z = torch.where(torch.abs(z) < 1e-6, torch.tensor(1e-6, device=self.device), z)
        
        xy = points_pix_homo[:, :, :2] / z
        xy[..., 1] = self.height - xy[..., 1]

        return xy.flip(-1) # (B, N, 2) -> (y, x)