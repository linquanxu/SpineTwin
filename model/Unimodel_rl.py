import torch
from torch import nn
from model.Base_module import TransformerEncoderLayer, VideoBaseEmbedding, TokenBaseEmbedding
from functools import partial
from timm.models.vision_transformer import Block
import numpy as np
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform
import torch.nn.functional as F
from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

from model.Unimodel_utils import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rl_nets import *
from model.rl_env import KeypointEnv

class FeatureDRR_Projector(nn.Module):
    def __init__(self, channels, proj_dim, num_samples=16):
        super().__init__()
        self.proj_dim = proj_dim 
        self.num_samples = num_samples
        
        self.ray_attn = nn.Sequential(
            nn.Conv3d(channels, channels // 4, kernel_size=1),
            nn.BatchNorm3d(channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x_3d, S):
        B, C, D, H, W = x_3d.shape
        device = x_3d.device
        
        if self.proj_dim == 3: 
            z_d, y_ray, x_d = torch.meshgrid(
                torch.linspace(-1, 1, D, device=device),
                torch.linspace(-1, 1, self.num_samples, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij'
            )
            scale = (y_ray + S) / (1.0 + S)
            x_grid = x_d * scale
            z_grid = z_d * scale
            y_grid = y_ray
            grid = torch.stack([x_grid, y_grid, z_grid], dim=-1) 
            grid = grid.permute(1, 0, 2, 3) 
            
        elif self.proj_dim == 4: 
            z_d, y_d, x_ray = torch.meshgrid(
                torch.linspace(-1, 1, D, device=device),
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, self.num_samples, device=device),
                indexing='ij'
            )
            scale = (x_ray + S) / (1.0 + S)
            y_grid = y_d * scale
            z_grid = z_d * scale
            x_grid = x_ray
            grid = torch.stack([x_grid, y_grid, z_grid], dim=-1) 
            grid = grid.permute(2, 0, 1, 3) 

        grid = grid.unsqueeze(0).expand(B, -1, -1, -1, -1) 
        sampled_features = F.grid_sample(x_3d, grid, align_corners=True) 
        
        attn_weights = self.ray_attn(sampled_features)
        attn_weights = attn_weights / (attn_weights.sum(dim=2, keepdim=True) + 1e-6)
        
        projected_2d = (sampled_features * attn_weights).sum(dim=2) 
        return projected_2d


class FeatureDRR_BackProjector(nn.Module):
    def __init__(self, proj_dim):
        super().__init__()
        self.proj_dim = proj_dim

    def forward(self, x_2d_padded, target_shape, S):
        B, C, D, H, W = target_shape
        device = x_2d_padded.device
        
        z_3d, y_3d, x_3d = torch.meshgrid(
            torch.linspace(-1, 1, D, device=device),
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij'
        )
        
        if self.proj_dim == 3: 
            scale = (1.0 + S) / (y_3d + S)
            x_proj = x_3d * scale
            z_proj = z_3d * scale
            grid_2d = torch.stack([x_proj, z_proj], dim=-1) 
            
        elif self.proj_dim == 4: 
            scale = (1.0 + S) / (x_3d + S)
            y_proj = y_3d * scale
            z_proj = z_3d * scale
            grid_2d = torch.stack([y_proj, z_proj], dim=-1) 

        grid_2d = grid_2d.unsqueeze(0).expand(B, -1, -1, -1, -1) 
        grid_flatten = grid_2d.reshape(B, D * H, W, 2)
        
        sampled_2d = F.grid_sample(x_2d_padded, grid_flatten, align_corners=True)
        lifted_3d = sampled_2d.view(B, C, D, H, W)
        
        return lifted_3d


class PerspectiveGeometryAwareFusion(nn.Module):
    def __init__(self, channels, ct_max_size=(64, 160, 160), img_2d_size=(256, 256), bottleneck_dim=128):
        super().__init__()
        self.ct_max_d, self.ct_max_h, self.ct_max_w = ct_max_size
        self.img_2d_h, self.img_2d_w = img_2d_size

        self.S_ap = nn.Parameter(torch.tensor(20.0))
        self.S_lat = nn.Parameter(torch.tensor(20.0))

        self.compress_3d = nn.Conv3d(channels, bottleneck_dim, 1)
        self.compress_ap = nn.Conv2d(channels, bottleneck_dim, 1)
        self.compress_lat = nn.Conv2d(channels, bottleneck_dim, 1)
        
        self.proj_to_ap = FeatureDRR_Projector(bottleneck_dim, proj_dim=3, num_samples=16)
        self.proj_to_lat = FeatureDRR_Projector(bottleneck_dim, proj_dim=4, num_samples=16)
        self.backproj_from_ap = FeatureDRR_BackProjector(proj_dim=3)
        self.backproj_from_lat = FeatureDRR_BackProjector(proj_dim=4)

        self.fuse_conv_ap = nn.Sequential(
            nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, 3, 1, 1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.fuse_conv_lat = nn.Sequential(
            nn.Conv2d(bottleneck_dim * 2, bottleneck_dim, 3, 1, 1),
            nn.BatchNorm2d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.reduce_3d = nn.Sequential(
            nn.Conv3d(bottleneck_dim * 3, bottleneck_dim, 1, 1, 0),
            nn.BatchNorm3d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.gate_3d = nn.Sequential(
            nn.Conv3d(bottleneck_dim, bottleneck_dim // 4, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(bottleneck_dim // 4, bottleneck_dim, 1, 1, 0),
            nn.Sigmoid() 
        )
        self.fuse_conv_3d = nn.Sequential(
            nn.Conv3d(bottleneck_dim, bottleneck_dim, 3, 1, 1),
            nn.BatchNorm3d(bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        
        self.expand_3d = nn.Conv3d(bottleneck_dim, channels, 1)
        self.expand_ap = nn.Conv2d(bottleneck_dim, channels, 1)
        self.expand_lat = nn.Conv2d(bottleneck_dim, channels, 1)
        
        self.gamma_3d = nn.Parameter(torch.zeros(1)) 

    def forward(self, x_ap, x_lat, x_3d, valid_shapes):
        B, C_orig, D3, H3, W3 = x_3d.shape
        _, _, H2, W2 = x_ap.shape

        x_ap_comp = self.compress_ap(x_ap)   
        x_lat_comp = self.compress_lat(x_lat) 
        x_3d_comp = self.compress_3d(x_3d)   

        feat_3d_proj_ap = self.proj_to_ap(x_3d_comp, self.S_ap)   
        feat_3d_proj_lat = self.proj_to_lat(x_3d_comp, self.S_lat) 

        aligned_3d_to_ap_list = []
        aligned_3d_to_lat_list = []

        for b in range(B):
            scale_d = D3 / self.ct_max_d
            scale_h = H3 / self.ct_max_h
            scale_w = W3 / self.ct_max_w
            
            valid_d = int(valid_shapes[b, 0] * scale_d)
            valid_h = int(valid_shapes[b, 1] * scale_h)
            valid_w = int(valid_shapes[b, 2] * scale_w)

            crop_ap = feat_3d_proj_ap[b:b+1, :, :valid_d, :valid_w] 
            crop_lat = feat_3d_proj_lat[b:b+1, :, :valid_d, :valid_h]
            
            resized_ap = F.interpolate(crop_ap, size=(H2, W2), mode='bilinear', align_corners=False)
            resized_lat = F.interpolate(crop_lat, size=(H2, W2), mode='bilinear', align_corners=False)
            
            aligned_3d_to_ap_list.append(resized_ap)
            aligned_3d_to_lat_list.append(resized_lat)

        aligned_3d_to_ap = torch.cat(aligned_3d_to_ap_list, dim=0)
        aligned_3d_to_lat = torch.cat(aligned_3d_to_lat_list, dim=0)

        fused_ap_comp = self.fuse_conv_ap(torch.cat([x_ap_comp, aligned_3d_to_ap], dim=1))
        fused_lat_comp = self.fuse_conv_lat(torch.cat([x_lat_comp, aligned_3d_to_lat], dim=1))

        feat_ap_padded_list = []
        feat_lat_padded_list = []

        for b in range(B):
            scale_d = D3 / self.ct_max_d
            scale_h = H3 / self.ct_max_h
            scale_w = W3 / self.ct_max_w
            valid_d = int(valid_shapes[b, 0] * scale_d)
            valid_h = int(valid_shapes[b, 1] * scale_h)
            valid_w = int(valid_shapes[b, 2] * scale_w)

            feat_ap_resized = F.interpolate(x_ap_comp[b:b+1], size=(valid_d, valid_w), mode='bilinear', align_corners=False)
            feat_lat_resized = F.interpolate(x_lat_comp[b:b+1], size=(valid_d, valid_h), mode='bilinear', align_corners=False)

            pad_ap = (0, W3 - valid_w, 0, D3 - valid_d)
            feat_ap_padded = F.pad(feat_ap_resized, pad_ap) 

            pad_lat = (0, H3 - valid_h, 0, D3 - valid_d)
            feat_lat_padded = F.pad(feat_lat_resized, pad_lat) 

            feat_ap_padded_list.append(feat_ap_padded)
            feat_lat_padded_list.append(feat_lat_padded)
        
        batch_ap_padded = torch.cat(feat_ap_padded_list, dim=0)
        batch_lat_padded = torch.cat(feat_lat_padded_list, dim=0)

        target_shape = (B, x_3d_comp.shape[1], D3, H3, W3)
        
        aligned_ap_to_3d = self.backproj_from_ap(batch_ap_padded, target_shape, self.S_ap)
        aligned_lat_to_3d = self.backproj_from_lat(batch_lat_padded, target_shape, self.S_lat)

        cat_feat = torch.cat([x_3d_comp, aligned_ap_to_3d, aligned_lat_to_3d], dim=1)

        fused_feat = self.reduce_3d(cat_feat)
        fused_feat = self.fuse_conv_3d(fused_feat)
        
        gate = self.gate_3d(fused_feat)
        fused_3d_comp = self.gamma_3d * (gate * fused_feat)
                    
        out_ap = x_ap + self.expand_ap(fused_ap_comp)
        out_lat = x_lat + self.expand_lat(fused_lat_comp)
        out_3d = x_3d + self.expand_3d(fused_3d_comp)

        return out_ap, out_lat, out_3d

class SoftArgmax(nn.Module):
    def __init__(self, normalized_coordinates=True):
        super().__init__()
        self.normalized_coordinates = normalized_coordinates

    def _build_grid(self, shape, device):
        dims = len(shape)
        coords_per_dim = []
        for s in shape:
            if self.normalized_coordinates:
                coords = torch.linspace(-1, 1, s, device=device)
            else:
                coords = torch.arange(0, s, device=device, dtype=torch.float32)
            coords_per_dim.append(coords)
        
        grid = torch.meshgrid(*coords_per_dim, indexing='ij')
        grid = torch.stack(grid, dim=-1)
        grid = grid.view(-1, dims) 
        
        return grid

    def forward(self, x):
        B, C = x.shape[:2]
        spatial_shape = x.shape[2:]
        device = x.device
        
        x_flat = x.view(B, C, -1)
        probs = F.softmax(x_flat, dim=-1) # [B, C, N_pixels]
        grid = self._build_grid(spatial_shape, device) # [N_pixels, dims]
        coords = torch.matmul(probs, grid)
        
        return coords

   
class Unified_Model(nn.Module):
    def __init__(self, now_3D_input_size, in_chans=1,in_chans_2d=3, patch_size=16, num_classes=2, pre_trained=False, pre_trained_weight=None,
        now_2D_input_size=None,patch_size_2d=16,mode='joint',use_rl=True):
        super(Unified_Model, self).__init__()
        self.num_head = 12
        self.patch_size = patch_size
        feature_size = 16
        norm_name = "instance"
        res_block = True
        self.hidden_size = 768
        self.feat_size = [now_3D_input_size[0] // self.patch_size, now_3D_input_size[1] // self.patch_size, now_3D_input_size[2] // self.patch_size]
        conv_block = True
        self.mode = mode
        self.soft_argmax_layer = SoftArgmax(normalized_coordinates=True)
        self.video_embed = VideoBaseEmbedding(input_size_3D=(256, 256, 256), input_size_2D=(512, 512))
        self.use_rl = use_rl


        if self.mode in ['fusion'] and self.use_rl:          
            self.patch_size_rl = 32


            self.q_net_2d = DuelingQNetwork(action_dim=5, is_3d=False)   # ★ was 5
            self.target_q_net_2d = DuelingQNetwork(action_dim=5, is_3d=False)
            self.target_q_net_2d.load_state_dict(self.q_net_2d.state_dict())

            self.target_update_freq = 100
            self.tau_2d = 0.005
            self.tau_3d = 0.01
            
        if self.mode in ['only_3d', 'joint', 'fusion']:
            self.fused_encoder = Encoder()
            self.encoder1 = UnetrBasicBlock(
                spatial_dims=3,
                in_channels=in_chans,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder2 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=self.hidden_size,
                out_channels=feature_size * 2,
                num_layer=2,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder3 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=self.hidden_size,
                out_channels=feature_size * 4,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder4 = UnetrPrUpBlock(
                spatial_dims=3,
                in_channels=self.hidden_size,
                out_channels=feature_size * 8,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.decoder5 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=self.hidden_size,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2 = UnetrUpBlock(
                spatial_dims=3,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=num_classes)  
 
        if self.mode in ['only_2d_ap', 'joint', 'fusion']:
            self.patch_size_2d = patch_size_2d
            self.feat_size_2d = [now_2D_input_size[0] // self.patch_size_2d, now_2D_input_size[1] // self.patch_size_2d]
            spatial_dims = 2
            
            self.fused_encoder_2d_ap = Encoder()
            self.encoder1_2d_ap = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_chans_2d,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder2_2d_ap = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 2,
                num_layer=2,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder3_2d_ap = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 4,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder4_2d_ap = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 8,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.decoder5_2d_ap = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4_2d_ap = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3_2d_ap = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2_2d_ap = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.out_2d_ap = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=num_classes)
            
        if self.mode in ['only_2d_lat', 'joint', 'fusion']:
            self.patch_size_2d = patch_size_2d
            self.feat_size_2d = [now_2D_input_size[0] // self.patch_size_2d, now_2D_input_size[1] // self.patch_size_2d]
            spatial_dims = 2

            self.fused_encoder_2d_lat = Encoder()
            self.encoder1_2d_lat = UnetrBasicBlock(
                spatial_dims=spatial_dims,
                in_channels=in_chans_2d,
                out_channels=feature_size,
                kernel_size=3,
                stride=1,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.encoder2_2d_lat = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 2,
                num_layer=2,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder3_2d_lat = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 4,
                num_layer=1,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.encoder4_2d_lat = UnetrPrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 8,
                num_layer=0,
                kernel_size=3,
                stride=1,
                upsample_kernel_size=2,
                norm_name=norm_name,
                conv_block=conv_block,
                res_block=res_block,
            )
            self.decoder5_2d_lat = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=self.hidden_size,
                out_channels=feature_size * 8,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder4_2d_lat = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 8,
                out_channels=feature_size * 4,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder3_2d_lat = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 4,
                out_channels=feature_size * 2,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.decoder2_2d_lat = UnetrUpBlock(
                spatial_dims=spatial_dims,
                in_channels=feature_size * 2,
                out_channels=feature_size,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            self.out_2d_lat = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=num_classes)

        if self.mode == 'fusion':
            self.fusion_layers = nn.ModuleList([
                PerspectiveGeometryAwareFusion(channels=self.hidden_size, 
                                    ct_max_size=now_3D_input_size, 
                                    img_2d_size=now_2D_input_size)
                for _ in range(4) 
            ])
            self.fusion_indices = [3, 6, 9, 11]
            # self.fusion_indices = [9, 11]

        self.initialize_weights()

        if pre_trained:
            print("load parameters from ", pre_trained_weight)
            model_dict = self.state_dict()
            pre_dict = torch.load(pre_trained_weight, map_location='cpu')["model"]
            pre_dict_update = {k: v for k, v in pre_dict.items() if k in model_dict}

            pre_dict_no_update = [k for k in pre_dict.keys() if k not in model_dict]
            print("no update: ", pre_dict_no_update)
            print("[pre_%d/mod_%d]: %d shared layers" % (len(pre_dict), len(model_dict), len(pre_dict_update)))
            model_dict.update(pre_dict_update)
            self.load_state_dict(model_dict)

    def init_weights_embedding(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _tokenize(self, data):
        emb_data = self.video_embed(data)
        return emb_data


    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def proj_feat_2d(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], hidden_size)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x



    # --- 2D AP ---
    def encode_2d_ap(self, x_in):
        x_emb = self._tokenize(x_in)
        x_encoded, hidden_states = self.fused_encoder_2d_ap(x_emb, None)
        enc1 = self.encoder1_2d_ap(x_in)
        return x_encoded, hidden_states, enc1

    def decode_2d_ap(self, x_encoded, hidden_states, enc1):
        dec4 = self.proj_feat_2d(x_encoded, self.hidden_size, self.feat_size_2d)
        
        enc2 = self.encoder2_2d_ap(self.proj_feat_2d(hidden_states[3], self.hidden_size, self.feat_size_2d))
        enc3 = self.encoder3_2d_ap(self.proj_feat_2d(hidden_states[6], self.hidden_size, self.feat_size_2d))
        enc4 = self.encoder4_2d_ap(self.proj_feat_2d(hidden_states[9], self.hidden_size, self.feat_size_2d))
        
        dec3 = self.decoder5_2d_ap(dec4, enc4)
        dec2 = self.decoder4_2d_ap(dec3, enc3)
        dec1 = self.decoder3_2d_ap(dec2, enc2)
        out = self.decoder2_2d_ap(dec1, enc1)
        return self.out_2d_ap(out)

    def encode_2d_lat(self, x_in):
        x_emb = self._tokenize(x_in)
        x_encoded, hidden_states = self.fused_encoder_2d_lat(x_emb, None)
        enc1 = self.encoder1_2d_lat(x_in)
        return x_encoded, hidden_states, enc1

    def decode_2d_lat(self, x_encoded, hidden_states, enc1):
        dec4 = self.proj_feat_2d(x_encoded, self.hidden_size, self.feat_size_2d)
        
        enc2 = self.encoder2_2d_lat(self.proj_feat_2d(hidden_states[3], self.hidden_size, self.feat_size_2d))
        enc3 = self.encoder3_2d_lat(self.proj_feat_2d(hidden_states[6], self.hidden_size, self.feat_size_2d))
        enc4 = self.encoder4_2d_lat(self.proj_feat_2d(hidden_states[9], self.hidden_size, self.feat_size_2d))
        
        dec3 = self.decoder5_2d_lat(dec4, enc4)
        dec2 = self.decoder4_2d_lat(dec3, enc3)
        dec1 = self.decoder3_2d_lat(dec2, enc2)
        out = self.decoder2_2d_lat(dec1, enc1)
        return self.out_2d_lat(out)

    def encode_3d(self, x_in):
        x_emb = self._tokenize(x_in)
        x_encoded, hidden_states = self.fused_encoder(x_emb, None)
        enc1 = self.encoder1(x_in)
        return x_encoded, hidden_states, enc1

    def decode_3d(self, x_encoded, hidden_states, enc1):
        dec4 = self.proj_feat(x_encoded, self.hidden_size, self.feat_size)
        
        enc2 = self.encoder2(self.proj_feat(hidden_states[3], self.hidden_size, self.feat_size))
        enc3 = self.encoder3(self.proj_feat(hidden_states[6], self.hidden_size, self.feat_size))
        enc4 = self.encoder4(self.proj_feat(hidden_states[9], self.hidden_size, self.feat_size))
        
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        return self.out(out)

    def forward(self, data):
        pred_2d_ap, pred_2d_lat, pred_3d = None, None, None
        valid_shapes = data.get("valid_shapes", None)

        feat_ap, hs_ap, enc1_ap = None, None, None
        if self.mode in ['only_2d_ap', 'joint', 'fusion']:
            feat_ap, hs_ap, enc1_ap = self.encode_2d_ap(data["data_2d_ap"])

        feat_lat, hs_lat, enc1_lat = None, None, None
        if self.mode in ['only_2d_lat', 'joint', 'fusion']:
            feat_lat, hs_lat, enc1_lat = self.encode_2d_lat(data["data_2d_lat"])

        feat_3d, hs_3d, enc1_3d = None, None, None
        if self.mode in ['only_3d', 'joint', 'fusion']:
            feat_3d, hs_3d, enc1_3d = self.encode_3d(data["data_3d"])

        fused_ap_final = feat_ap
        fused_lat_final = feat_lat
        fused_3d_final = feat_3d

        if self.mode == 'fusion' and hs_ap is not None and hs_lat is not None and hs_3d is not None:          
            for i, layer_idx in enumerate(self.fusion_indices):
                fusion_module = self.fusion_layers[i]

                if layer_idx == 11:
                    curr_ap = feat_ap
                    curr_lat = feat_lat
                    curr_3d = feat_3d
                else:
                    curr_ap = hs_ap[layer_idx]
                    curr_lat = hs_lat[layer_idx]
                    curr_3d = hs_3d[layer_idx]

                x_3d_s = self.proj_feat(curr_3d, self.hidden_size, self.feat_size)
                x_ap_s = self.proj_feat_2d(curr_ap, self.hidden_size, self.feat_size_2d)
                x_lat_s = self.proj_feat_2d(curr_lat, self.hidden_size, self.feat_size_2d)

                out_ap_s, out_lat_s, out_3d_s = fusion_module(
                    x_ap_s, x_lat_s, x_3d_s, valid_shapes
                )
                out_ap_flat = out_ap_s.flatten(2).transpose(1, 2)
                out_lat_flat = out_lat_s.flatten(2).transpose(1, 2)
                out_3d_flat = out_3d_s.flatten(2).transpose(1, 2)

                if layer_idx == 11:
                    fused_ap_final = out_ap_flat
                    fused_lat_final = out_lat_flat
                    fused_3d_final = out_3d_flat
                else:
                    hs_ap[layer_idx] = out_ap_flat
                    hs_lat[layer_idx] = out_lat_flat
                    hs_3d[layer_idx] = out_3d_flat

        if self.mode in ['only_2d_ap', 'joint', 'fusion']:
            pred_2d_ap = self.decode_2d_ap(fused_ap_final, hs_ap, enc1_ap)

        if self.mode in ['only_2d_lat', 'joint', 'fusion']:
            pred_2d_lat = self.decode_2d_lat(fused_lat_final, hs_lat, enc1_lat)

        if self.mode in ['only_3d', 'joint', 'fusion']:
            pred_3d = self.decode_3d(fused_3d_final, hs_3d, enc1_3d)

        
        if self.mode in ['fusion','joint']:
            coords_3d, coords_ap, coords_lat = None, None, None
            if pred_3d is not None:
                coords_3d = self.soft_argmax_layer(pred_3d)

            if pred_2d_ap is not None:
                coords_ap = self.soft_argmax_layer(pred_2d_ap)

            if pred_2d_lat is not None:
                coords_lat = self.soft_argmax_layer(pred_2d_lat)
            

            return {
                "pred_3d": pred_3d,
                "pred_2d_ap": pred_2d_ap,
                "pred_2d_lat": pred_2d_lat,
                "coords_3d": coords_3d, 
                "coords_ap": coords_ap, 
                "coords_lat": coords_lat 
            }
        else:
            return pred_2d_ap, pred_2d_lat, pred_3d
        
    def refine_keypoints_rl(self, image, init_coords, is_3d=True, max_steps=20,
                        reward_threshold=3.0, args=None, heatmap=None):
        refined_coords = []
        self.eval()

        for index, init_pos in enumerate(init_coords):
            kp_heatmap = heatmap[index] if heatmap is not None else None
            if kp_heatmap is not None:
                kp_heatmap = kp_heatmap.float()
                if kp_heatmap.min() < 0 or kp_heatmap.max() > 1.5:
                    kp_heatmap = torch.sigmoid(kp_heatmap)

            env = KeypointEnv(image, None, is_3d=is_3d, max_steps=max_steps,
                            heatmap=kp_heatmap)
            state = env.reset(init_pos=init_pos)

            done = False
            steps_taken = 0

            while not done and steps_taken < max_steps:
                q_net = self.q_net if is_3d else self.q_net_2d
                with torch.no_grad():
                    q_values = q_net(state.unsqueeze(0).cuda())
                action = torch.argmax(q_values).item()

                next_state, _, done, _ = env.step(action) 
                steps_taken += 1
                state = next_state
      
            final_pos = env.current_pos.cpu().numpy().tolist()
            refined_coords.append(final_pos)

        return refined_coords

    def update_target_net(self, net_type='3d'):
        for target_param, param in zip(self.target_q_net_2d.parameters(), self.q_net_2d.parameters()):
            target_param.data.copy_(self.tau_2d * param.data + (1 - self.tau_2d) * target_param.data)

    
