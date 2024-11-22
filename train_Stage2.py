
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

def setup_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    # random.seed(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
setup_seed(13) 

def pass_print(*args, **kwargs):
    pass

def classify_condition(subcondition_list):
    subconditions = {
        'rain': ['rain15', 'rain33', 'rain55'],
        'snow': ['light', 'medium', 'heavy'],
        'fog': ['foga', 'fogb', 'fogc']
    }
    
    condition_class = None
    for condition, sub_list in subconditions.items():
        if subcondition_list[0] in sub_list:
            condition_class = condition
            break
    
    if not all(sub in subconditions[condition_class] for sub in subcondition_list):
        raise ValueError("Subconditions in the batch belong to different conditions.")
    
    return condition_class

def main(local_rank, args):
    # # global settings
    # torch.backends.cudnn.benchmark = True

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
        cfg.gpu_ids = [0]         # debug
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
    if args.flag == 'K1':
        my_model = model_builder.build(cfg.model_Stage2_K1)
    elif args.flag == 'K3':
        my_model = model_builder.build(cfg.model_Stage2_K3)
    
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
    optimizer = build_optim_wrapper(my_model, cfg.optimizer_wrapper_stage2)
    
    # scaler = torch.cuda.amp.GradScaler() 
    
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
    
    print('work dir: ', args.work_dir) 
    
    if cfg.load_from:  
        try:
            pretrained_model = torch.load(cfg.load_from, map_location='cpu')
            my_model.load_state_dict(pretrained_model['state_dict'], strict=False)
            print(f"Pretrained model loaded successfully from: {cfg.load_from}")
        except Exception as e:
            print(f"Failed to load pretrained model from: {cfg.load_from}")
            print(f"Error: {e}")
    else:
        print("No pretrained model specified in the configuration.")


    for name, param in my_model.named_parameters():
        if "B1" not in name and "B2" not in name and "B3" not in name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in my_model.parameters())
    trainable_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f'Total number of parameters: {total_params}')
    logger.info(f'Number of trainable parameters: {trainable_params}')
    logger.info(f'Number of frozen parameters: {frozen_params}')

    # training
    print_freq = cfg.print_freq

    while epoch < max_num_epochs: 

        my_model.train()
        if hasattr(train_dataset_loader.sampler, 'set_epoch'):
            train_dataset_loader.sampler.set_epoch(epoch)

        loss_total_list = []

        loss_rain_list = []
        loss_snow_list = []
        loss_fog_list = []
        
        # rain
        loss1_list = []
        loss2_list = []
        # snow
        loss3_list = []
        loss4_list = []
        # fog
        loss5_list = []
        loss6_list = []

        data_time_s = time.time() 
        time_s = time.time() 
        
        for i_iter, data_total in enumerate(train_dataset_loader):

            data_rain, data_snow, data_fog = data_total

            points_rain, points_snow, points_fog = [], [], []
            train_grid_rain, train_grid_snow, train_grid_fog = [], [], []
            train_pt_labs_rain, train_pt_labs_snow, train_pt_labs_fog = [], [], []
            train_grid_vox_rain, train_grid_vox_snow, train_grid_vox_fog = [], [], []

            for data, points_list, grid_list, pt_labs_list, grid_vox_list in [
                (data_rain, points_rain, train_grid_rain, train_pt_labs_rain, train_grid_vox_rain),
                (data_snow, points_snow, train_grid_snow, train_pt_labs_snow, train_grid_vox_snow),
                (data_fog, points_fog, train_grid_fog, train_pt_labs_fog, train_grid_vox_fog)]:

                (points_data, grid_data, pt_labs_data, grid_vox_data) = data

                points_list.extend([torch.from_numpy(feat).to(torch.float32).contiguous().cuda() for feat in points_data])
                grid_list.extend([torch.from_numpy(grid_ind).to(torch.float32).contiguous().cuda() for grid_ind in grid_data])
                pt_labs_list.extend([torch.from_numpy(pt_lab).to(torch.long).contiguous().cuda() for pt_lab in pt_labs_data])
                grid_vox_list.extend([torch.from_numpy(grid_ind_vox).to(torch.float32).contiguous().cuda() for grid_ind_vox in grid_vox_data])

            # forward + backward + optimize
            data_time_e = time.time()
            # ============================== data_rain  ============================== #
            outputs_pts_rain = my_model(points=points_rain, grid_ind=train_grid_rain, grid_ind_vox=train_grid_vox_rain, flag = [1,0,0])
                
            lovasz_input_rain = outputs_pts_rain[0] # (1, 5, n, 1, 1)
            lovasz_label_rain = train_pt_labs_rain[0].unsqueeze(0) 
                
            ce_input_rain = lovasz_input_rain.squeeze(-1).squeeze(-1) # (1, 5, n)
            ce_label_rain = lovasz_label_rain.squeeze(-1) # [1, n]
                
            loss1 = lovasz_softmax(
                    torch.nn.functional.softmax(lovasz_input_rain, dim=1), 
                    lovasz_label_rain, ignore=ignore_label
                ) + loss_func(ce_input_rain, ce_label_rain)
                
            loss2 = args.lam * sum([abs(i) for i in my_model.getIndicators_B1()]) /1000
                
            loss_rain = loss1 + loss2
            
            """ loss.backward() """
            loss_rain.backward() 
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                
            """ optimizer.step() """
            optimizer.step() 
            optimizer.zero_grad()
            
            loss1_list.append(loss1.item())
            loss2_list.append(loss2.item())
            loss_rain_list.append(loss_rain.item())
            # ============================== data_snow  ============================== #
            outputs_pts_snow = my_model(points=points_snow, grid_ind=train_grid_snow, grid_ind_vox=train_grid_vox_snow, flag = [0,1,0])
                
            lovasz_input_snow = outputs_pts_snow[0]
            lovasz_label_snow = train_pt_labs_snow[0].unsqueeze(0) 
                
            ce_input_snow = lovasz_input_snow.squeeze(-1).squeeze(-1) # (1, 5, n)
            ce_label_snow = lovasz_label_snow.squeeze(-1) # [1, n]
                
            loss3 = lovasz_softmax(
                    torch.nn.functional.softmax(lovasz_input_snow, dim=1), 
                    lovasz_label_snow, ignore=ignore_label
                ) + loss_func(ce_input_snow, ce_label_snow)
                
            loss4 = args.lam * sum([abs(i) for i in my_model.getIndicators_B2()]) / 1000
                
            loss_snow = loss3 + loss4
            
            """ loss.backward() """
            loss_snow.backward() 
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                
            """ optimizer.step() """
            optimizer.step()  
            optimizer.zero_grad()

            loss3_list.append(loss3.item())
            loss4_list.append(loss4.item())
            loss_snow_list.append(loss_snow.item())
            # ============================== data_fog  ============================== #
            outputs_pts_fog = my_model(points=points_fog, grid_ind=train_grid_fog, grid_ind_vox=train_grid_vox_fog, flag = [0,0,1])
                
            lovasz_input_fog = outputs_pts_fog[0]
            lovasz_label_fog = train_pt_labs_fog[0].unsqueeze(0) 
                
            ce_input_fog = lovasz_input_fog.squeeze(-1).squeeze(-1) 
            ce_label_fog = lovasz_label_fog.squeeze(-1) 
                
            loss5 = lovasz_softmax(
                    torch.nn.functional.softmax(lovasz_input_fog, dim=1), 
                    lovasz_label_fog, ignore=ignore_label
                ) + loss_func(ce_input_fog, ce_label_fog)
                
            loss6 = args.lam * sum([abs(i) for i in my_model.getIndicators_B3()]) /1000
                
            loss_fog = loss5 + loss6
        
            """ loss.backward() """
            loss_fog.backward() 
            grad_norm = torch.nn.utils.clip_grad_norm_(my_model.parameters(), cfg.grad_max_norm)
                
            """ optimizer.step() """
            optimizer.step()  
            optimizer.zero_grad()

            loss5_list.append(loss5.item())
            loss6_list.append(loss6.item())
            loss_fog_list.append(loss_fog.item())

            loss_total = loss_rain + loss_snow + loss_fog
            loss_total_list.append(loss_total.item())

            scheduler.step_update(global_iter)

            time_e = time.time()

            global_iter += 1

            Percent_B1 = torch.mean((torch.tensor(my_model.getIndicators_B1()) >= .1).float())
            Percent_B2 = torch.mean((torch.tensor(my_model.getIndicators_B2()) >= .1).float())
            Percent_B3 = torch.mean((torch.tensor(my_model.getIndicators_B3()) >= .1).float())
            
            Percent_B1_1 = torch.mean((torch.tensor(my_model.getIndicators_B1()) >= .2).float())
            Percent_B2_1 = torch.mean((torch.tensor(my_model.getIndicators_B2()) >= .2).float())
            Percent_B3_1 = torch.mean((torch.tensor(my_model.getIndicators_B3()) >= .2).float())
            
            Percent_B1_2 = torch.mean((torch.tensor(my_model.getIndicators_B1()) >= .4).float())
            Percent_B2_2 = torch.mean((torch.tensor(my_model.getIndicators_B2()) >= .4).float())
            Percent_B3_2 = torch.mean((torch.tensor(my_model.getIndicators_B3()) >= .4).float())


            if i_iter % print_freq == 0 and dist.get_rank() == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info('[TRAIN] Epoch %d Iter %5d/%d: '
                             'Loss_total: %.3f (%.3f), '
                             'Loss_rain: %.3f (%.3f), Loss1: %.3f (%.3f), Loss2: %.3f (%.3f), '
                             'Loss_snow: %.3f (%.3f), Loss3: %.3f (%.3f), Loss4: %.3f (%.3f), '
                             'Loss_fog: %.3f (%.3f), Loss5: %.3f (%.3f), Loss6: %.3f (%.3f), '
                             'lr: %.7f, time: %.3f (%.3f)' % (
                                 epoch + 1, i_iter, len(train_dataset_loader),
                                 loss_total_list[-1], np.mean(loss_total_list),
                                 loss_rain_list[-1], np.mean(loss_rain_list), loss1_list[-1], np.mean(loss1_list), loss2_list[-1], np.mean(loss2_list),
                                 loss_snow_list[-1], np.mean(loss_snow_list), loss3_list[-1], np.mean(loss3_list), loss4_list[-1], np.mean(loss4_list),
                                 loss_fog_list[-1], np.mean(loss_fog_list), loss5_list[-1], np.mean(loss5_list), loss6_list[-1], np.mean(loss6_list),
                                 lr, time_e - time_s, data_time_e - data_time_s
                             ))
                logger.info("[Threshold [0.1],  PercentB1: %.3f, PercentB2: %.3f, PercentB3: %.3f]" % (Percent_B1.item(), Percent_B2.item(), Percent_B3.item()))
                logger.info("[Threshold [0.2],  PercentB1: %.3f, PercentB2: %.3f, PercentB3: %.3f]" % (Percent_B1_1.item(), Percent_B2_1.item(), Percent_B3_1.item()))
                logger.info("[Threshold [0.4],  PercentB1: %.3f, PercentB2: %.3f, PercentB3: %.3f]" % (Percent_B1_2.item(), Percent_B2_2.item(), Percent_B3_2.item()))

                loss_total_list = [] 
                loss_rain_list = []
                loss_snow_list = []
                loss_fog_list = []
                # rain
                loss1_list = []
                loss2_list = []
                # snow
                loss3_list = []
                loss4_list = []
                # fog
                loss5_list = []
                loss6_list = []

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
        val_loss_1_list = []
        val_loss_2_list = []
        CalMeanIou_pts.reset()

        with torch.no_grad():
            for i_iter_val, data in enumerate(val_dataset_loader):

                (points_list, val_grid_list, val_pt_labs_list, val_grid_vox_list, subcondition_list) = data

                points = [torch.from_numpy(feat).to(torch.float32).contiguous().cuda() for feat in points_list]
                val_grid_float = [torch.from_numpy(grid_ind).to(torch.float32).contiguous().cuda() for grid_ind in val_grid_list]
                val_pt_labs = [torch.from_numpy(pt_lab).to(torch.long).contiguous().cuda() for pt_lab in val_pt_labs_list]
                val_grid_vox = [torch.from_numpy(grid_ind_vox).to(torch.float32).contiguous().cuda() for grid_ind_vox in val_grid_vox_list]

                condition_class = classify_condition(subcondition_list)

                flag_mapping = {
                    'rain': [1, 0, 0],
                    'snow': [0, 1, 0],
                    'fog': [0, 0, 1]
                }
                flag = flag_mapping.get(condition_class)
                if flag is None:
                    raise ValueError("Unknown condition class.")

                predict_labels_pts = my_model(points=points, grid_ind=val_grid_float, grid_ind_vox=val_grid_vox, flag=flag) 
 
                lovasz_input = predict_labels_pts[0]
                lovasz_label = val_pt_labs[0].unsqueeze(0) 
                    
                ce_input = lovasz_input.squeeze(-1).squeeze(-1) 
                ce_label = lovasz_label.squeeze(-1) 

                loss_1 = lovasz_softmax(
                    torch.nn.functional.softmax(lovasz_input, dim=1).detach(), 
                    lovasz_label, ignore=ignore_label
                ) + loss_func(ce_input.detach(), ce_label)

                indicators_method = {
                    'rain': my_model.getIndicators_B1,
                    'snow': my_model.getIndicators_B2,
                    'fog': my_model.getIndicators_B3
                }.get(condition_class)
                if indicators_method is None:
                    raise ValueError("Unknown condition class.")
                loss_2 = args.lam * sum([abs(i) for i in indicators_method()]) / 1000            
                
                loss = loss_1 + loss_2

                # (1, 5, n, 1, 1) -> (1, 5, n)
                predict_labels_pts = predict_labels_pts[0].squeeze(-1).squeeze(-1)

                predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n

                predict_labels_pts = predict_labels_pts.detach().cpu()
                val_pt_labs = ce_label.cpu()
                
                for count in range(len(predict_labels_pts)):
                    CalMeanIou_pts._after_step(predict_labels_pts[count], val_pt_labs[count]) 
 
                val_loss_list.append(loss.detach().cpu().numpy()) 
                val_loss_1_list.append(loss_1.detach().cpu().numpy())
                val_loss_2_list.append(loss_2.detach().cpu().numpy())

              
                if i_iter_val % print_freq == 0 and dist.get_rank() == 0:
                    logger.info('[EVAL] Epoch %d Iter %5d: Loss: %.3f (%.3f), Loss_1: %.3f (%.3f), Loss_2: %.3f (%.3f)'%(
                        epoch, i_iter_val, 
                        loss.item(), np.mean(val_loss_list),
                        loss_1.item(), np.mean(val_loss_1_list),  
                        loss_2.item(), np.mean(val_loss_2_list) 
                    ))

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
    parser.add_argument('--work-dir', type=str, default='') 
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch') 
    parser.add_argument('--flag', type=str, default= '')
    parser.add_argument('--lam', type=float, default= '')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count() 
    args.gpus = ngpus
    print(args)
    
    if args.launcher == 'none':
        main(0, args)
    else:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus) 