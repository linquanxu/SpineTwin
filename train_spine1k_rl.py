import os
from scipy.fftpack import shift
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["CUDNN_DETERMINISTIC"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import os.path as osp
import random
import time
import math
import numpy as np
import shutil
import cv2
cv2.setNumThreads(0) 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import MOTS_Dataset_train_2d3d, MOTS_Dataset_test_2d3d, my_collate, my_collate_test
from model.Unimodel_rl import Unified_Model
from tensorboardX import SummaryWriter
from utils.ParaFlop import print_model_parm_nums
from engine_seed import Engine
from tools.m_utils_rl import *
from model.project import DiffDRR_Projector
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
import psutil  


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

        if args.mode == 'cce':
            print("Mode: CCE (Cross-Consistency Evaluation)")
            print("Initializing models...")
            model_ap = Unified_Model(now_3D_input_size=input_size, num_classes=args.num_classes, 
                                     now_2D_input_size=input_size_2d, mode='only_2d_ap')
            model_lat = Unified_Model(now_3D_input_size=input_size, num_classes=args.num_classes, 
                                      now_2D_input_size=input_size_2d, mode='only_2d_lat')
            model_ct = Unified_Model(now_3D_input_size=input_size, num_classes=args.num_classes, 
                                     now_2D_input_size=input_size_2d, mode='only_3d')
            def load_checkpoint(model, path, name):
                if os.path.isfile(path):
                    print(f"Loading {name} from {path}...")
                    checkpoint = torch.load(path, map_location='cpu')
                    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
                    msg = model.load_state_dict(state_dict, strict=False)
                    print(f"Loaded {name}. Msg: {msg}")
                else:
                    print(f"Error: {name} checkpoint not found at {path}")
                    exit()

            load_checkpoint(model_ap, '/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_2d_ap/fold0/37_2.7562checkpoint_total.pth', "AP Model")
            load_checkpoint(model_lat, '/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_2d_lat/fold0/33_4.1637checkpoint_total.pth', "Lat Model")
            load_checkpoint(model_ct, '/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/fold0/66_3.1529checkpoint_total.pth', "CT Model")

            models_dict = {
                'ap': model_ap,
                'lat': model_lat,
                'ct': model_ct
            }
            valloader, val_sampler = engine.get_test_loader(
                MOTS_Dataset_test_2d3d(args.test_dir_3d, args.test_list_3d,
                                       root_2d=args.test_dir_2d, list_path_2d=args.test_list_2d,
                                       crop_size=input_size, crop_size_2d=input_size_2d, 
                                       num_classes=args.num_classes, 
                                       mode='fusion'),
                batch_size=1, 
                collate_fn=my_collate_test
            )
            val_cce(args, models_dict, valloader, device, input_size_3d=input_size)
            return 

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
        
        if args.use_rl and args.mode == 'fusion':
            print("=> Stage 2: Loading pre-trained Fusion model and freezing it for RL training...")
            
            for param in model.parameters():
                param.requires_grad = False
                
            rl_params = []
            if hasattr(model, 'q_net_2d'):
                for param in model.q_net_2d.parameters():
                    param.requires_grad = True
                rl_params += list(model.q_net_2d.parameters())
                for param in model.target_q_net_2d.parameters():
                    param.requires_grad = False 
                    
            
            optimizer_rl_2d = torch.optim.Adam(model.q_net_2d.parameters(), lr=1e-4)
            scheduler = None
            
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
            scheduler = get_scheduler(optimizer, args.num_epochs, args.warmup_epochs, args.learning_rate, args.min_lr)
        
        trainloader, train_sampler = engine.get_train_loader(
            MOTS_Dataset_train_2d3d(args.train_dir_3d, args.train_list_3d, 
                            crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                            root_2d=args.train_dir_2d, list_path_2d=args.train_list_2d,
                            crop_size_2d=input_size_2d, scale_2d=args.random_scale_2d, mirror_2d=args.random_mirror_2d,
                            split_2d="train", ratio_labels_2d=args.ratio_labels_2d, num_classes=args.num_classes, mode=args.mode),
                            drop_last=True, collate_fn=my_collate)
        
        kl_criterion = nn.KLDivLoss(reduction='batchmean')
       
        if args.find_lr:
            run_lr_finder(args, model, trainloader, optimizer, device, None, None, kl_criterion, start_lr=1e-7, end_lr=50, num_iter=1000)
            return

        restart_path = 'results/fusion/fold0/checkpoint.pth'

        to_restore = {"epoch": 0}
        restart_from_checkpoint( 
            restart_path,
            run_variables=to_restore,
            model=model,
            optimizer=None,
            scheduler=scheduler if scheduler is not None else None
        )
        start_epoch = 0

        valloader, val_sampler = engine.get_test_loader(
            MOTS_Dataset_test_2d3d(args.test_dir_3d, args.test_list_3d, root_2d=args.test_dir_2d, list_path_2d=args.test_list_2d,
            crop_size=input_size, crop_size_2d=input_size_2d, num_classes=args.num_classes, mode=args.mode), 1, collate_fn=my_collate_test)

        best_mre_total = 1000

        if args.local_rank == 0:
            print(f"Start training with Fixed Weights (MSE + 0.01 * KL)")
            print(f"Current Mode: {args.mode}")

        rl_patience = 20
        rl_patience_counter = 0
        best_rl_mre = 1000.0
        
        if args.mode == "fusion" and args.use_rl:
            global_step = 0
            total_steps = args.num_epochs * len(trainloader)
            replay_buffer_2d = PrioritizedReplayBuffer(args.rl_buffer_size, alpha=0.7, beta=0.5, beta_anneal_steps=5000)

            
        for epoch in range(start_epoch, args.num_epochs):
            if args.local_rank == 0:
                mem = psutil.virtual_memory()
                print(f"\n>>> Starting Epoch {epoch}/{args.num_epochs}, "
                      f"RAM: {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f} GB ({mem.percent}%)")
            
            model.train()
            
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            mse_ap_loss, mse_lat_loss, mse_ct_loss = [], [], []
            kl_ap_loss, kl_lat_loss, kl_ct_loss = [], [], []

            epoch_rl_loss_2d_list = []
            epoch_rl_loss_3d_list = []
            
            time_t1 = time.time()
            
            epoch_rewards_2d = []
            epoch_qvalues_2d = []
            
            for iter, batch in enumerate(trainloader):
                # if iter == 0:
                #     break
                rec_mse_2d_ap, rec_mse_2d_lat, rec_mse_3d = 0, 0, 0
                rec_kl_2d_ap, rec_kl_2d_lat, rec_kl_3d = 0, 0, 0
                
                r_2d_ap = batch.get('r_2d_ap')
                r_2d_lat = batch.get('r_2d_lat')
                r_3d = batch.get('r_3d')
                ct_oir_size = batch['ct_oir_size']

                images_2d_ap, labels_2d_ap = None, None
                images_2d_lat, labels_2d_lat = None, None
                images, labels = None, None
                valid_shapes = None
                
                if args.mode in ['fusion']:
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
                    "data_2d_ap": images_2d_ap, 
                    "labels_2d_ap": labels_2d_ap, 
                    "data_2d_lat": images_2d_lat, 
                    "labels_2d_lat": labels_2d_lat,
                    "data_3d": images, 
                    "labels_3d": labels, 
                    'ct_oir_size': ct_oir_size,
                    'valid_shapes': valid_shapes
                }

                if args.mode == 'fusion' and args.use_rl:
                    model.eval() 
                    with torch.no_grad():
                        outputs = model(data) 
                        pred_2d_ap = outputs['pred_2d_ap']
                        pred_2d_lat = outputs['pred_2d_lat']
                    
                    if args.rl_epsilon_decay:
                        current_epsilon = max(args.rl_epsilon_min, args.rl_epsilon * (1 - global_step / total_steps))
                    else:
                        current_epsilon = args.rl_epsilon
                            
                    sig_pred_2d_ap = torch.sigmoid(pred_2d_ap)    
                    sig_pred_2d_lat = torch.sigmoid(pred_2d_lat) 

                    
                    rl_loss_2d = 0.0
                    coords_2d_ap_pred = extract_coordinates_2d_unbiased(sig_pred_2d_ap[0].detach().cpu().numpy())
                    coords_2d_ap_gt = extract_coordinates_2d_unbiased(labels_2d_ap[0].detach().cpu().numpy())
                    heatmap_rl_ap = sig_pred_2d_ap[0].cpu()
                    
                    envs_2d_ap = []
                    states_2d_ap = []
                    for kp_idx in range(len(coords_2d_ap_pred)):
                        heatmap_k = heatmap_rl_ap[kp_idx]
                        for ep in range(args.rl_episodes_per_batch // 2):
                            env = KeypointEnv(images_2d_ap[0].cpu(), coords_2d_ap_gt[kp_idx], is_3d=False, max_steps=args.rl_max_steps_2d, rl_reward_scale=args.rl_reward_scale, heatmap=heatmap_k)
                            
                            state = env.reset(init_pos=coords_2d_ap_pred[kp_idx], train_mode=True)
                            envs_2d_ap.append(env)
                            states_2d_ap.append(state)
                            
                    run_batched_rl_episodes(envs_2d_ap, states_2d_ap, model.q_net_2d, current_epsilon, args.rl_max_steps_2d, replay_buffer_2d, epoch_rewards_2d, epoch_qvalues_2d)

       
                    coords_2d_lat_pred = extract_coordinates_2d_unbiased(sig_pred_2d_lat[0].detach().cpu().numpy())
                    coords_2d_lat_gt = extract_coordinates_2d_unbiased(labels_2d_lat[0].detach().cpu().numpy())
                    heatmap_rl_lat = sig_pred_2d_lat[0].cpu()
                    
                    envs_2d_lat = []
                    states_2d_lat = []
                    for kp_idx in range(len(coords_2d_lat_pred)):
                        heatmap_k = heatmap_rl_lat[kp_idx]
                        for ep in range(args.rl_episodes_per_batch // 2):
                            env = KeypointEnv(images_2d_lat[0].cpu(), coords_2d_lat_gt[kp_idx], is_3d=False, max_steps=args.rl_max_steps_2d, rl_reward_scale=args.rl_reward_scale, heatmap=heatmap_k)
                            state = env.reset(init_pos=coords_2d_lat_pred[kp_idx], train_mode=True)
                            envs_2d_lat.append(env)
                            states_2d_lat.append(state)
                            
                    run_batched_rl_episodes(envs_2d_lat, states_2d_lat, model.q_net_2d, current_epsilon, args.rl_max_steps_2d, replay_buffer_2d, epoch_rewards_2d, epoch_qvalues_2d)
            
                    model.q_net_2d.train()
                    if len(replay_buffer_2d.buffer) >= args.rl_batch_size:
                        samples, indices, weights = replay_buffer_2d.sample(args.rl_batch_size)
                        states, actions, rewards, next_states, dones = zip(*samples)
                        states = torch.stack(states).cuda()
                        next_states = torch.stack(next_states).cuda()
                        actions = torch.tensor(actions, dtype=torch.long).cuda()
                        rewards = torch.tensor(rewards).cuda()
                        dones = torch.tensor(dones, dtype=torch.float32).cuda()
                        weights = weights.cuda()
                        
                        with torch.no_grad():
                            next_actions_online = model.q_net_2d(next_states).max(1)[1]
                            next_q_values_target = model.target_q_net_2d(next_states)
                            
                            target_q = rewards + args.rl_gamma * next_q_values_target.gather(1, next_actions_online.unsqueeze(1)).squeeze(1) * (1 - dones)

                        q_values = model.q_net_2d(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    
                        td_errors = torch.abs(q_values - target_q).detach().cpu().numpy()
                        rl_loss_2d = (F.mse_loss(q_values, target_q, reduction='none') * weights).mean()
                        replay_buffer_2d.update_priorities(indices, td_errors)

                        optimizer_rl_2d.zero_grad()
                        rl_loss_2d.backward()
                        torch.nn.utils.clip_grad_norm_(model.q_net_2d.parameters(), 1.0)
                        optimizer_rl_2d.step()
                        
                        if isinstance(rl_loss_2d, torch.Tensor):
                            epoch_rl_loss_2d_list.append(rl_loss_2d.item())
                    
                    global_step += 1
                    if global_step % 1000 == 0:
                        model.update_target_net(net_type='2d')

                mse_ap_loss.append(rec_mse_2d_ap)
                mse_lat_loss.append(rec_mse_2d_lat)
                mse_ct_loss.append(rec_mse_3d)
                kl_ap_loss.append(rec_kl_2d_ap)
                kl_lat_loss.append(rec_kl_2d_lat)
                kl_ct_loss.append(rec_kl_3d)
            
            if scheduler is not None:
                scheduler.step()
            
      
            time_t2 = time.time()
            
            if args.local_rank == 0:
                epoch_time = time_t2 - time_t1
                avg_rl_loss_2d = sum(epoch_rl_loss_2d_list) / len(epoch_rl_loss_2d_list) if epoch_rl_loss_2d_list else 0.0
                avg_rl_loss_3d = sum(epoch_rl_loss_3d_list) / len(epoch_rl_loss_3d_list) if epoch_rl_loss_3d_list else 0.0
                
                print(f"Epoch {epoch} finished in {epoch_time:.2f}s.")
                if args.use_rl and args.mode == 'fusion':
                    print(f"    Avg RL Loss 2D: {avg_rl_loss_2d:.6f} | Avg RL Loss 3D: {avg_rl_loss_3d:.6f}")
            
            save_dict = {
                'model': model.state_dict(),
                'optimizer_2d': optimizer_rl_2d.state_dict(),
                'epoch': epoch + 1,
            }
            if scheduler is not None:
                save_dict['scheduler'] = scheduler.state_dict()
                
            if args.local_rank == 0:
                torch.save(save_dict, osp.join(args.snapshot_dir, 'checkpoint.pth'))
            
       
            model.eval()
            val_metrics = validate(args, input_size, [model], valloader, args.num_classes, engine, input_size_2d=input_size_2d, writer=writer, epoch=epoch)     
            current_mre_total = val_metrics['total']
            # return
            
            if args.use_rl and args.mode == 'fusion':
                if current_mre_total < best_rl_mre:
                    best_rl_mre = current_mre_total
                    rl_patience_counter = 0
                    if args.local_rank == 0:
                        torch.save(save_dict, osp.join(args.snapshot_dir, f'{epoch}_{best_rl_mre:.4f}checkpoint_total.pth'))
                        print(f"New best RL model saved with MRE: {best_rl_mre:.4f}")
                else:
                    rl_patience_counter += 1
                    if args.local_rank == 0:
                        print(f"    [RL] No improvement. Patience: {rl_patience_counter}/{rl_patience}")
                    if rl_patience_counter >= rl_patience:
                        if args.local_rank == 0:
                            print(f"RL Early Stopping triggered at epoch {epoch}. Best MRE: {best_rl_mre:.4f}")
                        break
            else:
                if current_mre_total < best_mre_total:
                    best_mre_total = current_mre_total
                    if args.local_rank == 0:
                        torch.save(save_dict, osp.join(args.snapshot_dir, f'{epoch}_{best_mre_total:.4f}checkpoint_total.pth'))
                        print(f"New best model saved with MRE sum: {best_mre_total:.4f}")
        
        if args.local_rank == 0:
            print(f"\n>>> Training finished normally after {args.num_epochs} epochs.")

if __name__ == '__main__':
    main()