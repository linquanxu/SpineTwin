import os
from scipy.fftpack import shift
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDNN_DETERMINISTIC"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import os.path as osp
import time
import numpy as np
import cv2
cv2.setNumThreads(0) 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import MOTS_Dataset_train_2d3d, MOTS_Dataset_test_2d3d, my_collate, my_collate_test
from model.Unimodel import Unified_Model
from tensorboardX import SummaryWriter
from utils.ParaFlop import print_model_parm_nums
from engine_seed import Engine
from tools.m_utils import *
from model.project import DiffDRR_Projector
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
def get_scheduler(optimizer, num_epochs, warmup_epochs, base_lr, min_lr):
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=min_lr / base_lr if base_lr > 0 else 0, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs - warmup_epochs, 
        eta_min=min_lr
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    return scheduler

def main():
    parser = get_arguments()
    parser.add_argument("--find_lr", action="store_true", help="Run LR Finder instead of training")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--warmup_epochs", type=int, default=10, help="Number of warmup epochs")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    
    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        
        base_seed = int(getattr(args, "random_seed", 42))
        rank_seed = base_seed + int(getattr(args, "local_rank", 0))
        set_seeds(rank_seed)
        enforce_determinism()
        set_monai_determinism(seed=rank_seed)
        
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)
        
        writer = SummaryWriter(args.snapshot_dir)
        
        d, h, w = map(int, args.input_size.split(','))
        input_size = (d, h, w)
        h_2d, w_2d = map(int, args.input_size_2d.split(','))
        input_size_2d = (h_2d, w_2d)
        device = torch.device('cuda:{}'.format(args.local_rank))

        if args.mode == "fusion":
            drr_projector = DiffDRR_Projector(sdd=1000, height=256, delx=0.8, device=device)


        if args.arch == "unified_vit":
            model = Unified_Model(now_3D_input_size=input_size, num_classes=args.num_classes, 
                                  pre_trained=args.reload_from_pretrained,
                                  pre_trained_weight=args.pretrained_path, 
                                  now_2D_input_size=input_size_2d, mode=args.mode)
        else:
            print("Architecture not supported")
            exit()
            
        model.train()
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
        trainloader, train_sampler = engine.get_train_loader(
            MOTS_Dataset_train_2d3d(args.train_dir_3d, args.train_list_3d, 
                            crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                            root_2d=args.train_dir_2d,list_path_2d=args.train_list_2d,
                            crop_size_2d=input_size_2d,scale_2d=args.random_scale_2d,mirror_2d=args.random_mirror_2d,
                            split_2d="train",ratio_labels_2d=args.ratio_labels_2d, num_classes = args.num_classes,mode=args.mode),
                            drop_last=True, collate_fn=my_collate)
        
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
        scheduler = get_scheduler(optimizer, args.num_epochs, args.warmup_epochs, args.learning_rate, args.min_lr)
        restart_path = '/results/test/fusion/checkpoint.pth'

        to_restore = {"epoch": 0}
        restart_from_checkpoint( 
            restart_path,
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler
        )
        start_epoch = to_restore["epoch"]

        valloader, val_sampler = engine.get_test_loader(
            MOTS_Dataset_test_2d3d(args.test_dir_3d, args.test_list_3d,root_2d=args.test_dir_2d,list_path_2d=args.test_list_2d,
            crop_size=input_size,crop_size_2d=input_size_2d, num_classes = args.num_classes, mode=args.mode), 1,collate_fn=my_collate_test)

        best_mre_total = 1000
        if args.local_rank == 0:
            print(f"Start training with Fixed Weights (MSE + 0.01 * KL)")
            print(f"Current Mode: {args.mode}")
        
        for epoch in range(start_epoch, args.num_epochs):          
            model.eval()
            validate(args, input_size, [model], valloader, args.num_classes, engine, input_size_2d=input_size_2d, writer=writer, epoch=epoch)   
            return
    
                                                        
if __name__ == '__main__':
    main()