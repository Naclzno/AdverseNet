import torch
from dataloader.dataset import AdverseNet_Dataset_train, AdverseNet_Dataset_eval
from dataloader.dataset_wrapper import seg_custom_collate_fn, seg_custom_collate_fn_train, DatasetWrapper_AdverseNet_train, DatasetWrapper_AdverseNet_eval


def build_seg(dataset_config,
          train_dataloader_config,
          val_dataloader_config,
          grid_size=[200, 200, 16],          
          dist=False,
    ):
    #---------------------------------------------------------------------------------------------------------------#
    
    data_path_train = train_dataloader_config["data_path"]
    fix_sample_train = train_dataloader_config["fix_sample"]
     
    data_path_val = val_dataloader_config["data_path"]
    fix_sample_val = val_dataloader_config["fix_sample"]
    
    label_mapping = dataset_config["label_mapping"]  

    # data_tuple_rain, data_tuple_snow, data_tuple_fog
    train_dataset = AdverseNet_Dataset_train(data_path_train, fix_sample_train, label_mapping=label_mapping)
    # data_tuple = (points, mapped_labels, subcondition)
    val_dataset = AdverseNet_Dataset_eval(data_path_val, fix_sample_val, label_mapping=label_mapping)
    
    train_dataset = DatasetWrapper_AdverseNet_train(
        train_dataset, # 
        grid_size=grid_size, # [480, 360, 32]
        grid_size_vox=dataset_config['grid_size_vox'], # [480, 360, 32]
        fixed_volume_space=dataset_config['fixed_volume_space'], # True
        max_volume_space=dataset_config['max_volume_space'], # [50, 3.1415926, 3]
        min_volume_space=dataset_config['min_volume_space'], # [0, -3.1415926, -5]
        fill_label=dataset_config["fill_label"], # 0
        rotate_aug=dataset_config['rotate_aug'], # True
        flip_aug=dataset_config['flip_aug'], # True
        scale_aug=dataset_config['scale_aug'], # True
        transform_aug=dataset_config['transform_aug'], # True
        trans_std=dataset_config['trans_std'], # [0.1, 0.1, 0.1]
    )

    val_dataset = DatasetWrapper_AdverseNet_eval(
        val_dataset,
        grid_size=grid_size,
        grid_size_vox=dataset_config['grid_size_vox'],
        fixed_volume_space=dataset_config['fixed_volume_space'],
        max_volume_space=dataset_config['max_volume_space'],
        min_volume_space=dataset_config['min_volume_space'],
        fill_label=dataset_config["fill_label"],
    )
        
    collate_fn_train = seg_custom_collate_fn_train
    collate_fn = seg_custom_collate_fn

    if dist:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, drop_last=False)
    else:
        sampler = None
        val_sampler = None
    

    train_dataset_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=1,
                                                       collate_fn=collate_fn_train, 
                                                       shuffle=False if dist else train_dataloader_config["shuffle"],
                                                       sampler=sampler,
                                                       num_workers=train_dataloader_config["num_workers"])
    
    val_dataset_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=val_dataloader_config["batch_size"],
                                                     collate_fn=collate_fn,
                                                     shuffle=False if dist else val_dataloader_config["shuffle"],
                                                     sampler=val_sampler,
                                                     num_workers=val_dataloader_config["num_workers"])

    return train_dataset_loader, val_dataset_loader