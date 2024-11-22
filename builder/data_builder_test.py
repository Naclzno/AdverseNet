import torch
from dataloader.dataset import AdverseNet_Dataset_eval
from dataloader.dataset_wrapper import seg_custom_collate_fn, DatasetWrapper_AdverseNet_eval


def build_seg(dataset_config,
          test_dataloader_config,
          grid_size=[200, 200, 16],          
          dist=False,
    ):
    #---------------------------------------------------------------------------------------------------------------#
    
    data_path_test = test_dataloader_config["data_path"]
    fix_sample_test = test_dataloader_config["fix_sample"]
     
    # dataset_config 就是 配置文件 中的 dataset_params
    label_mapping = dataset_config["label_mapping"]  # AdverseNet.yaml的路径

    """ data_tuple[0]代表点云数据 n行4列 
        data_tuple[1]代表标签数据 n行1列
        lidar数据存储格式 : (x,y,z,intensity)"""
    # data_tuple = (points, mapped_labels, subcondition)
    test_dataset = AdverseNet_Dataset_eval(data_path_test, fix_sample_test, label_mapping=label_mapping)

    
    #---------------------------------------------------------------------------------------------------------------#
    
    test_dataset = DatasetWrapper_AdverseNet_eval(
        test_dataset,
        grid_size=grid_size,
        grid_size_vox=dataset_config['grid_size_vox'],
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        fill_label=dataset_config["fill_label"],
    )
        
    collate_fn = seg_custom_collate_fn


    # 控制Distributed Data Parallelism
    if dist:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    else:
        test_sampler = None
        
    test_dataset_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                     batch_size=test_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn,
                                                     shuffle=False if dist else test_dataloader_config["shuffle"],
                                                     sampler=test_sampler,
                                                     num_workers=test_dataloader_config["num_workers"])

    return test_dataset_loader