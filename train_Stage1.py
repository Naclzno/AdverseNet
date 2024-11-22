"""
ID: Naclzno
Name: Xinyuan Yan
Email: yan1075783878@gmail.com
"""

import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.metric_util import MeanIoU
from utils.load_save_util import revise_ckpt, revise_ckpt_2
from dataloader.dataset import get_label_name
from builder import loss_builder

from mmengine import Config
from mmengine.optim.optimizer.builder import build_optim_wrapper
from mmengine.logging.logger import MMLogger
from mmengine.utils import symlink
from timm.scheduler import CosineLRScheduler

import warnings
warnings.filterwarnings("ignore")
import random

# def setup_seed(seed):
#     torch.manual_seed(seed) 
#     torch.cuda.manual_seed_all(seed) 
#     np.random.seed(seed) 
#     random.seed(seed) 
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False 
# setup_seed(13) 

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config 
    cfg = Config.fromfile(args.py_config) 
    cfg.work_dir = args.work_dir 

    dataset_config = cfg.dataset_params 
    ignore_label = dataset_config['ignore_label'] 
    train_dataloader_config = cfg.train_data_loader 
    val_dataloader_config = cfg.val_data_loader 

    max_num_epochs = cfg.max_epochs  
    grid_size = cfg.grid_size 

    # init DDP
    if args.launcher == 'none': 
        distributed = False
        rank = 0 
        cfg.gpu_ids = [0]         
    else: 
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20506")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank
        )
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size) 
        torch.cuda.set_device(local_rank) 
        
        if dist.get_rank() != 0:
            import builtins
            builtins.print = pass_print

    # configure logger
    if local_rank == 0 and rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
    

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger(name='train_log', log_file=log_file, log_level='INFO')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    from builder import model_builder
    my_model = model_builder.build(cfg.model_Stage1)
    total_params = sum(p.numel() for p in my_model.parameters())
    trainable_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)


    logger.info(f'Total number of parameters: {total_params}')
    logger.info(f'Number of trainable parameters: {trainable_params}')
    logger.info(f'Model:\n{my_model}')

    my_model = my_model.cuda()  

    print('done model')

    label_name = get_label_name(dataset_config["label_mapping"])  

    unique_label = np.asarray(cfg.unique_label) 

    unique_label_str = [label_name[x] for x in unique_label]

    from builder import data_builder
    train_dataset_loader, val_dataset_loader = \
        data_builder.build_seg(
            dataset_config, 
            train_dataloader_config, 
            val_dataloader_config, 
            grid_size=grid_size, 
            dist=distributed, 
        )

    # get optimizer, loss, scheduler
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper_stage1)

    # in the lidar segmentation，we employ the classic cross-entropy loss and lovasz-softmax loss
    loss_func, lovasz_softmax = \
        loss_builder.build(ignore_label=ignore_label) 
    
    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=len(train_dataset_loader)*max_num_epochs, 
        lr_min=1e-6, 
        warmup_t=500, 
        warmup_lr_init=1e-5, 
        t_in_epochs=False 
    )
   
    CalMeanIou_pts = MeanIoU(unique_label, ignore_label, unique_label_str, 'pts')
    
    # resume and load
    epoch = 0
    best_val_miou_pts = 0

    global_iter = 0

    cfg.resume_from = ''

    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    print('resume from: ', cfg.resume_from) 
    print('work dir: ', args.work_dir) 
    
    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'

        ckpt = torch.load(cfg.resume_from, map_location=map_location)

        print(my_model.load_state_dict(revise_ckpt(ckpt['state_dict']), strict=False))
        
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

        epoch = ckpt['epoch']

        if 'best_val_miou_pts' in ckpt:
            best_val_miou_pts = ckpt['best_val_miou_pts']

        global_iter = ckpt['global_iter']
        print(f'successfully resumed from epoch {epoch}')
    
    elif cfg.load_from: 
        
        ckpt = torch.load(cfg.load_from, map_location='cpu')

        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt 
        state_dict = revise_ckpt(state_dict)

        try:
            print(my_model.load_state_dict(state_dict, strict=False))
        except:
            state_dict = revise_ckpt_2(state_dict)
            print(my_model.load_state_dict(state_dict, strict=False))
        

    # training
    print_freq = cfg.print_freq

    while epoch < max_num_epochs: 

        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)
        
        loss_list = []
        time.sleep(10) 

        data_time_s = time.time() 
        time_s = time.time() 
        
        for i_iter, data_total in enumerate(train_dataset_loader):

            data_rain, data_snow, data_fog = data_total

            points = []
            train_grid = []
            train_pt_labs = []
            train_grid_vox = []

            for data in [data_rain, data_snow, data_fog]:
                (points_list, train_grid_list, train_pt_labs_list, train_grid_vox_list) = data
                points.extend([torch.from_numpy(feat).to(torch.float32).contiguous().cuda() for feat in points_list])
                train_grid.extend([torch.from_numpy(grid_ind).to(torch.float32).contiguous().cuda() for grid_ind in train_grid_list])
                train_pt_labs.extend([torch.from_numpy(pt_lab).to(torch.long).contiguous().cuda() for pt_lab in train_pt_labs_list])
                train_grid_vox.extend([torch.from_numpy(grid_ind_vox).to(torch.float32).contiguous().cuda() for grid_ind_vox in train_grid_vox_list])

            # forward + backward + optimize
            data_time_e = time.time()

            # with torch.cuda.amp.autocast():

            outputs_pts = my_model(points=points, grid_ind=train_grid, grid_ind_vox=train_grid_vox)
                
            total_loss = 0.0

            for idx, output_pts in enumerate(outputs_pts):
                lovasz_input = output_pts  
                lovasz_label = train_pt_labs[idx].unsqueeze(0) 

                ce_input = output_pts.squeeze(-1).squeeze(-1)  
                ce_label = lovasz_label.squeeze(-1)  

                loss = lovasz_softmax(torch.nn.functional.softmax(lovasz_input, dim=1), lovasz_label, ignore=ignore_label) + \
                        loss_func(ce_input, ce_label)
                total_loss += loss

                
            """ loss.backward() """
            total_loss.backward() 

            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm) 
                
            """ optimizer.step() """
            optimizer.step() 
            optimizer.zero_grad()
            loss_list.append(total_loss.item())

            scheduler.step_update(global_iter)

            time_e = time.time()

            global_iter += 1

            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: Loss: %.3f (%.3f), lr: %.7f, time: %.3f (%.3f)'%(
                    epoch+1, i_iter, len(train_dataset_loader), 
                    loss_list[-1], np.mean(loss_list), lr,
                    time_e - time_s, data_time_e - data_time_s
                ))
                loss_list = [] 
            data_time_s = time.time() 
            time_s = time.time() 
        
        # save checkpoint
        if dist.get_rank() == 0: 

            dict_to_save = {
                'state_dict': my_model.state_dict(), 
                'optimizer': optimizer.state_dict(), 
                'scheduler': scheduler.state_dict(), 
                'epoch': epoch + 1, 
                'global_iter': global_iter, 
                'best_val_miou_pts': best_val_miou_pts
            }
            save_file_name = os.path.join(os.path.abspath(args.work_dir), f'epoch_{epoch+1}.pth')
            torch.save(dict_to_save, save_file_name)
            dst_file = osp.join(args.work_dir, 'latest.pth')
            symlink(save_file_name, dst_file)

        epoch += 1
        
        # eval
        my_model.eval()
        val_loss_list = []
        CalMeanIou_pts.reset()

        with torch.no_grad():

            for i_iter_val, data in enumerate(val_dataset_loader):
                
                (points_list, val_grid_list, val_pt_labs_list, val_grid_vox_list, _) = data

                points = [torch.from_numpy(feat).to(torch.float32).contiguous().cuda() for feat in points_list]
                val_grid_float = [torch.from_numpy(grid_ind).to(torch.float32).contiguous().cuda() for grid_ind in val_grid_list]
                val_pt_labs = [torch.from_numpy(pt_lab).to(torch.long).contiguous().cuda() for pt_lab in val_pt_labs_list]
                val_grid_vox = [torch.from_numpy(grid_ind_vox).to(torch.float32).contiguous().cuda() for grid_ind_vox in val_grid_vox_list]

                predict_labels_pts = my_model(points=points, grid_ind=val_grid_float, grid_ind_vox=val_grid_vox)
 
                lovasz_input = predict_labels_pts[0] 
                lovasz_label = val_pt_labs[0].unsqueeze(0) 
                    
                ce_input = lovasz_input.squeeze(-1).squeeze(-1) 
                ce_label = lovasz_label.squeeze(-1) 
                
                loss = lovasz_softmax(
                    torch.nn.functional.softmax(lovasz_input, dim=1).detach(), 
                    lovasz_label, ignore=ignore_label
                ) + loss_func(ce_input.detach(), ce_label)

                predict_labels_pts = predict_labels_pts[0].squeeze(-1).squeeze(-1)

                predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) 

                predict_labels_pts = predict_labels_pts.detach().cpu()

                val_pt_labs = ce_label.squeeze(-1).cpu()
                
                for count in range(len(predict_labels_pts)):
                    CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count]) 

                
                val_loss_list.append(loss.detach().cpu().numpy()) 

                if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f)'%(
                        epoch, i_iter_val, loss.item(), np.mean(val_loss_list)))

        val_miou_pts = CalMeanIou_pts._after_epoch()

        if best_val_miou_pts < val_miou_pts:
            best_val_miou_pts = val_miou_pts

        logger.info('Current val miou pts is %.3f while the best val miou pts is %.3f' %
                (val_miou_pts, best_val_miou_pts))
        logger.info('Current val loss is %.3f' %
                (np.mean(val_loss_list)))

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='') # /home/yxy/AdverseNet/config/AdverseNet_config.py
    parser.add_argument('--work-dir', type=str, default='/home/yxy/work/fifth') 
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch') 
    parser.add_argument('--resume-from', type=str, default='')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count() 
    args.gpus = ngpus
    print(args)
    
    if args.launcher == 'none':
        main(0, args)
    else:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus) 