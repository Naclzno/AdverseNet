
import os, time, argparse, os.path as osp, numpy as np
import torch
import torch.distributed as dist
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'

from utils.metric_util import MeanIoU_test
from utils.load_save_util import revise_ckpt
from dataloader.dataset import get_label_name
from builder import loss_builder

from mmengine import Config
from mmengine.logging.logger import MMLogger

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def classify_condition(subcondition_list):
    # 定义每个条件对应的子条件
    subconditions = {
        'rain': ['rain15', 'rain33', 'rain55'],
        'snow': ['light', 'medium', 'heavy'],
        'fog': ['foga', 'fogb', 'fogc']
    }
    
    # 确保subcondition_list中的所有子条件都属于同一大类
    condition_class = None
    for condition, sub_list in subconditions.items():
        if subcondition_list[0] in sub_list:
            condition_class = condition
            break
    
    # 确认所有子条件都属于识别的大类
    if not all(sub in subconditions[condition_class] for sub in subcondition_list):
        raise ValueError("Subconditions in the batch belong to different conditions.")
    
    return condition_class

def main(local_rank, args):
    # global settings
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)

    dataset_config = cfg.dataset_params
    ignore_label = dataset_config['ignore_label']
    
    test_dataloader_config = cfg.test_data_loader

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


    logger = MMLogger(name='test_log', log_file=args.log_file, log_level='INFO')

    # build model
    from builder import model_builder

    if args.flag == 'K1':
        my_model = model_builder.build(cfg.model_Stage2_K1)
    elif args.flag == 'K3':
        my_model = model_builder.build(cfg.model_Stage2_K3)
    elif args.flag == 'S1':
        my_model = model_builder.build(cfg.model_Stage1)

    # n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    # logger.info(f'Number of params: {n_parameters}')

    # 计算总参数数
    total_params = sum(p.numel() for p in my_model.parameters())
    # 计算可训练的参数数
    trainable_params = sum(p.numel() for p in my_model.parameters() if p.requires_grad)

    # 打印参数信息
    logger.info(f'Total number of parameters: {total_params}')
    logger.info(f'Number of trainable parameters: {trainable_params}')

    logger.info(f'Model:\n{my_model}')
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', True)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        my_model = my_model.cuda()
    print('done ddp model')

    # generate datasets
    label_name = get_label_name(dataset_config["label_mapping"])
    unique_label = np.asarray(cfg.unique_label)
    # unique_label_str = [label_name[x] for x in unique_label]
    unique_label_str = ['clear', 'rain15', 'rain33', 'rain55', 'light', 'medium', 'heavy', 'foga', 'fogb', 'fogc']

    from builder import data_builder_test
    test_dataset_loader = \
        data_builder_test.build_seg(
            dataset_config,
            test_dataloader_config,
            grid_size=grid_size,
            dist=distributed,
        )

    CalMeanIou_pts = MeanIoU_test(unique_label, ignore_label, unique_label_str, 'pts')
    
    # resume and load
    assert osp.isfile(args.ckpt_path)
    print('ckpt path:', args.ckpt_path)
    
    map_location = 'cpu'
    ckpt = torch.load(args.ckpt_path, map_location=map_location)
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    print(my_model.load_state_dict(revise_ckpt(ckpt), strict=False))
    print(f'successfully loaded ckpt')
    
    print_freq = cfg.print_freq
    
    # eval
    my_model.eval()
    CalMeanIou_pts.reset()

    with torch.no_grad():
        for i_iter_test, data in enumerate(test_dataset_loader):
            # test_pt_labs的形状是[bs, n, 1]
            (points_list, test_grid_list, test_pt_labs_list, test_grid_vox_list, subcondition_list) = data

            # 对列表中的每个元素逐个转换为Tensor并移动到GPU上
            points = [torch.from_numpy(feat).to(torch.float32).contiguous().cuda() for feat in points_list]
            test_grid_float = [torch.from_numpy(grid_ind).to(torch.float32).contiguous().cuda() for grid_ind in test_grid_list]
            test_pt_labs = [torch.from_numpy(pt_lab).to(torch.long).contiguous().cuda() for pt_lab in test_pt_labs_list]
            test_grid_vox = [torch.from_numpy(grid_ind_vox).to(torch.float32).contiguous().cuda() for grid_ind_vox in test_grid_vox_list]

            # 使用辅助函数来分类subcondition_list，并设置flag
            condition_class = classify_condition(subcondition_list)

            flag_mapping = {
                'rain': [1, 0, 0],
                'snow': [0, 1, 0],
                'fog': [0, 0, 1]
            }
            flag = flag_mapping.get(condition_class)
            if flag is None:
                raise ValueError("Unknown condition class.")


            if args.flag == 'S1':
                predict_labels_pts = my_model(points=points, grid_ind=test_grid_float, grid_ind_vox=test_grid_vox)
            else:
                predict_labels_pts = my_model(points=points, grid_ind=test_grid_float, grid_ind_vox=test_grid_vox, flag=flag)
            # logits_pts = logits.reshape(bs, self.classes, n, 1, 1)
        
            predict_labels_pts = predict_labels_pts[0].squeeze(-1).squeeze(-1)
            predict_labels_pts = torch.argmax(predict_labels_pts, dim=1) # bs, n
            predict_labels_pts = predict_labels_pts.detach().cpu()
            test_pt_labs = test_pt_labs[0].unsqueeze(0) # (1, n, 1)
            test_pt_labs = test_pt_labs.squeeze(-1).cpu() # (bs, n)
            
            
            for count in range(len(predict_labels_pts)):
                CalMeanIou_pts._after_step(predict_labels_pts[count], test_pt_labs[count], subcondition_list[count])

            
            if i_iter_test % print_freq == 0 and dist.get_rank() == 0:
                logger.info('[EVAL] Iter %5d: Loss: None'%(i_iter_test))
        
        test_miou_pts = CalMeanIou_pts._after_epoch()
        logger.info('test miou pts is %.3f' % (test_miou_pts))

        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch')
    parser.add_argument('--ckpt-path', type=str, default=None)
    parser.add_argument('--log-file', type=str, default=None)
    parser.add_argument('--save-result', type=bool, default= False)
    parser.add_argument('--flag', type=str, default= '')

    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)
    
    if args.launcher == 'none':
        main(0, args)
    else:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
