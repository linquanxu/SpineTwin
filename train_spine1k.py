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

            load_checkpoint(model_ap, '/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_2d_ap/fold4/53_2.3201checkpoint_total.pth', "AP Model")
            load_checkpoint(model_lat, '/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_2d_lat/fold4/12_3.5821checkpoint_total.pth', "Lat Model")
            load_checkpoint(model_ct, '/data/xulinquan/medcoss/checkpoints/spine_1015_n/only_3d/fold4/72_2.8093checkpoint_total.pth', "CT Model")

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
            val_metrics_baseline_mean(args, models_dict, valloader, device)
            val_cce(args, models_dict, valloader, device, input_size_3d=input_size)
            
            return 


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
        if args.find_lr:
            run_lr_finder(args, model, trainloader, optimizer, device, None, None, kl_criterion, start_lr=1e-7, end_lr=50, num_iter=1000)
            return

        restart_path = os.path.join(args.snapshot_dir, "checkpoint.pth")
        # restart_path = '/data/xulinquan/medcoss/checkpoints/spine_1015_n/joint/fold2/59_3.0247checkpoint_total.pth'

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
            model.train()
            
            if engine.distributed:
                train_sampler.set_epoch(epoch)

            mse_ap_loss, mse_lat_loss, mse_ct_loss = [], [], []
            kl_ap_loss, kl_lat_loss, kl_ct_loss = [], [], []
            all_loss_record = []
            
            time_t1 = time.time()
            optimizer.zero_grad()
            for iter, batch in enumerate(trainloader):
                # if iter == 0:
                #     break
                r_2d_ap = batch.get('r_2d_ap')
                r_2d_lat = batch.get('r_2d_lat')
                r_3d = batch.get('r_3d')
                ct_oir_size = batch['ct_oir_size']
                # print(batch.get('name'))

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
                    "data_2d_ap": images_2d_ap, 
                    "labels_2d_ap": labels_2d_ap, 
                    "data_2d_lat": images_2d_lat, 
                    "labels_2d_lat": labels_2d_lat,
                    "data_3d": images, 
                    "labels_3d": labels, 
                    'ct_oir_size': ct_oir_size,
                    'valid_shapes': valid_shapes
                }

                if args.mode in ['fusion']:
                    outputs = model(data) 
                    pred_2d_ap, pred_2d_lat, pred_3d = outputs['pred_2d_ap'], outputs['pred_2d_lat'], outputs['pred_3d']
                    coords_3d, coords_ap, coords_lat = outputs['coords_3d'], outputs['coords_ap'], outputs['coords_lat']
                elif args.mode in ['joint']: 
                    outputs = model(data)
                    pred_2d_ap, pred_2d_lat, pred_3d = outputs['pred_2d_ap'], outputs['pred_2d_lat'], outputs['pred_3d']
                else:      
                    pred_2d_ap, pred_2d_lat, pred_3d = model(data) 
                
                total_loss = 0.0
                rec_mse_2d_ap, rec_mse_2d_lat, rec_mse_3d = 0, 0, 0
                rec_kl_2d_ap, rec_kl_2d_lat, rec_kl_3d = 0, 0, 0

                if args.mode in ['only_2d_ap', 'joint', 'fusion']:
                    sig_pred_2d_ap = torch.sigmoid(pred_2d_ap)
                    mse_2d_ap = F.mse_loss(sig_pred_2d_ap, labels_2d_ap) 
                    
                    log_prob_2d_ap = spatial_log_softmax(pred_2d_ap)
                    with torch.no_grad():
                        prob_gt_2d_ap = linear_normalize_gt(labels_2d_ap)
                    kl_2d_ap = kl_criterion(log_prob_2d_ap, prob_gt_2d_ap)
                    
                    total_loss += mse_2d_ap + 0.01 * kl_2d_ap
                    
                    rec_mse_2d_ap = mse_2d_ap.item()
                    rec_kl_2d_ap = kl_2d_ap.item()


                if args.mode in ['only_2d_lat', 'joint', 'fusion']:
                    sig_pred_2d_lat = torch.sigmoid(pred_2d_lat)
                    mse_2d_lat = F.mse_loss(sig_pred_2d_lat, labels_2d_lat)
                    
                    log_prob_2d_lat = spatial_log_softmax(pred_2d_lat)
                    with torch.no_grad():
                        prob_gt_2d_lat = linear_normalize_gt(labels_2d_lat)
                    kl_2d_lat = kl_criterion(log_prob_2d_lat, prob_gt_2d_lat)
                    
                    total_loss += mse_2d_lat + 0.01 * kl_2d_lat

                    rec_mse_2d_lat = mse_2d_lat.item()
                    rec_kl_2d_lat = kl_2d_lat.item()
            
            
                if args.mode in ['only_3d', 'joint', 'fusion']:
                    sig_pred_3d = torch.sigmoid(pred_3d)
                    mse_3d = F.mse_loss(sig_pred_3d, labels)
                    
                    log_prob_3d = spatial_log_softmax(pred_3d)
                    with torch.no_grad():
                        prob_gt_3d = linear_normalize_gt(labels)
                    kl_3d = kl_criterion(log_prob_3d, prob_gt_3d)
                    
                    total_loss += mse_3d + 0.01 * kl_3d
                    rec_mse_3d = mse_3d.item()
                    rec_kl_3d = kl_3d.item()

                    
                if args.mode == 'fusion' and pred_3d is not None:
                    batch_props = batch.get('prop', [])
                    batch_shifts = []
                    for p in batch_props:
                        batch_shifts.append(p.get('shift', np.array([0.0, 0.0, 0.0])))
                    batch_shifts = torch.tensor(np.stack(batch_shifts), device=device, dtype=torch.float32)

                    
                    gt_3d_x = extract_coordinates_unbiased(labels[0].detach())
                    
                    gt_3d_voxels = torch.tensor(gt_3d_x, device=device, dtype=torch.float32)
                    if gt_3d_voxels.ndim == 2:
                        gt_3d_voxels = gt_3d_voxels.unsqueeze(0) 
                    codap,codlat = drr_projector(gt_3d_voxels,batch_props,shifts=batch_shifts)
                    codap[..., 1] = 255.0 - codap[..., 1]
                    codlat[..., 1] = 255.0 - codlat[..., 1]
                    # print(codap)
                    labels_2d_ap_coord = extract_coordinates_2d_unbiased(labels_2d_ap[0].detach().cpu())
                    # print(labels_2d_ap_coord)
                    labels_2d_lat_coord = extract_coordinates_2d_unbiased(labels_2d_lat[0].detach().cpu())
                    # print(labels_2d_lat_coord)

                    
                    if True and (len(batch_props) > 0 and 
                        coords_3d is not None and 
                        coords_ap is not None and 
                        coords_lat is not None):
                        
                        pad_d, pad_h, pad_w = 64, 160, 160 
                        padded_size = torch.tensor([pad_d, pad_h, pad_w], device=device, dtype=torch.float32)
            
                        coords_3d_01 = (coords_3d + 1) / 2
                        coords_3d_vox = coords_3d_01 * padded_size.view(1, 1, 3) # (B, N, 3) (Z, Y, X)
                        
                        proj_ap_pix, proj_lat_pix = drr_projector(coords_3d_vox, batch_props,shifts=batch_shifts)
                        
                        proj_ap_pix_flipped = proj_ap_pix.clone()
                        proj_ap_pix_flipped[..., 1] = 255.0 - proj_ap_pix_flipped[..., 1] 
                        
                        proj_lat_pix_flipped = proj_lat_pix.clone()
                        proj_lat_pix_flipped[..., 1] = 255.0 - proj_lat_pix_flipped[..., 1] 

                       
                        drr_height = 256.0 
                        coords_ap_01_yx = (coords_ap + 1) / 2
                        coords_lat_01_yx = (coords_lat + 1) / 2

                        coords_ap_01 = torch.flip(coords_ap_01_yx, dims=[-1])
                        coords_lat_01 = torch.flip(coords_lat_01_yx, dims=[-1])
                        
                        pred_ap_pix = coords_ap_01 * drr_height
                        pred_lat_pix = coords_lat_01 * drr_height
                        
                        
                        valid_mask_ap = ((proj_ap_pix_flipped >= 0) & (proj_ap_pix_flipped <= drr_height)).all(dim=-1)
                        valid_mask_lat = ((proj_lat_pix_flipped >= 0) & (proj_lat_pix_flipped <= drr_height)).all(dim=-1)
                        
                        loss_proj = torch.tensor(0.0, device=device)
                    
                        if valid_mask_ap.any():
                            loss_proj += F.l1_loss(
                                proj_ap_pix_flipped[valid_mask_ap], 
                                pred_ap_pix[valid_mask_ap]
                            )
                            
                        if valid_mask_lat.any():
                            loss_proj += F.l1_loss(
                                proj_lat_pix_flipped[valid_mask_lat], 
                                pred_lat_pix[valid_mask_lat]
                            )
                                                
                        total_loss += 0.005 * loss_proj 
                        
                        # if iter == 0:
                        #     # print(f"\n[{'='*15} Epoch {epoch} | Iter {iter} : Consistency Loss Info {'='*15}]")
                        #     # print(f"-> loss_proj (original): {loss_proj.item():.4f}")
                        #     # print(f"-> loss_proj (weighted): {0.005 * loss_proj.item():.4f}")
                            
                        #     if valid_mask_ap.any():
                        #         valid_pred_ap = pred_ap_pix[valid_mask_ap]
                        #         valid_proj_ap = proj_ap_pix_flipped[valid_mask_ap]
                                
                        #         num_to_print = min(2, valid_pred_ap.shape[0]) 
                        #         for i in range(num_to_print):
                        #             pred_pt = [round(x, 2) for x in valid_pred_ap[i].tolist()]
                        #             proj_pt = [round(x, 2) for x in valid_proj_ap[i].tolist()]
                        #             print(f"   point {i+1}: pred {pred_pt}  |  proj {proj_pt}")
                            
                        #     if valid_mask_lat.any():
                        #         valid_pred_lat = pred_lat_pix[valid_mask_lat]
                        #         valid_proj_lat = proj_lat_pix_flipped[valid_mask_lat]
                                
                        #         num_to_print = min(2, valid_pred_lat.shape[0])
                        #         for i in range(num_to_print):
                        #             pred_pt = [round(x, 2) for x in valid_pred_lat[i].tolist()]
                        #             proj_pt = [round(x, 2) for x in valid_proj_lat[i].tolist()]
                        #             print(f"  point {i+1}: pred {pred_pt}  |  proj {proj_pt}")
                                    
                        #     print("="*65 + "\n")
        
                        
                total_loss = total_loss / args.grad_accum_steps
                total_loss.backward()
                
                if (iter + 1) % args.grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
                    optimizer.step()
                    optimizer.zero_grad()

                all_loss_record.append(total_loss.item() * args.grad_accum_steps)
                mse_ap_loss.append(rec_mse_2d_ap)
                mse_lat_loss.append(rec_mse_2d_lat)
                mse_ct_loss.append(rec_mse_3d)
                kl_ap_loss.append(rec_kl_2d_ap)
                kl_lat_loss.append(rec_kl_2d_lat)
                kl_ct_loss.append(rec_kl_3d)
            scheduler.step()
            
            avg_all_loss = np.mean(all_loss_record)
            time_t2 = time.time()
            
            if (args.local_rank == 0):
                current_lr = optimizer.param_groups[0]['lr']
                print('Epoch {}: lr = {:.6f}, Loss = {:.4f}, Time = {}s'.format(
                    epoch, current_lr, avg_all_loss, int(time_t2 - time_t1)))
                
                writer.add_scalar('learning_rate', current_lr, epoch)
                writer.add_scalar('Train_loss', avg_all_loss, epoch)

            save_dict = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch + 1,
            }
            if args.local_rank == 0:
                torch.save(save_dict, osp.join(args.snapshot_dir, 'checkpoint.pth'))           
            model.eval()
            val_metrics = validate(args, input_size, [model], valloader, args.num_classes, engine, input_size_2d=input_size_2d, writer=writer, epoch=epoch)   
            # return      
            current_mre_total = val_metrics['total']
            if current_mre_total < best_mre_total:
                best_mre_total = current_mre_total
                if args.local_rank == 0:
                    torch.save(save_dict, osp.join(args.snapshot_dir, f'{epoch}_{best_mre_total:.4f}checkpoint_total.pth'))
                    print(f"New best model saved with MRE sum: {best_mre_total:.4f}")
                                                        
if __name__ == '__main__':
    main()