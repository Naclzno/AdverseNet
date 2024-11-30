
print_freq = 50
max_epochs = 40
# load_from = '/home/yxy/work/fifth/stage1/epoch_9.pth'
# load_from = '/home/yxy/work/fourth/stage1-40/epoch_16.pth'
# load_from = '/home/yxy/work/first/stage1/epoch_11.pth'
load_from = None

grad_max_norm = 35 

optimizer_wrapper_stage1 = dict(
    optimizer = dict(
        type='AdamW', 
        lr=2e-4, 
        weight_decay=0.01, 
    ),
)

optimizer_wrapper_stage2 = dict(
    optimizer = dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=0.01,
    ),
)


find_unused_parameters = False
unique_label = [1, 2, 3, 4]
track_running_stats = False

_dim_ = 24 

tpv_w_ = 240
tpv_h_ = 180
tpv_z_ = 16
scale_w = 2
scale_h = 2
scale_z = 2

grid_size = [480, 360, 32] 

nbr_class = 5

dataset_params = dict(
    grid_size_vox = [tpv_w_*scale_w, tpv_h_*scale_h, tpv_z_*scale_z], 
    fill_label = 0,
    ignore_label = 0,
    fixed_volume_space = True, 
    # fixed_volume_space = False,
    label_mapping = "./config/label_mapping/AdverseNet.yaml",
    max_volume_space = [50, 3.1415926, 2], 
    min_volume_space = [0, -3.1415926, -4], 
    rotate_aug = True, 
    flip_aug = True,
    scale_aug = True,
    transform_aug = True,
    trans_std=[0.1, 0.1, 0.1], 
)

train_data_loader = dict(
    data_path = "/home/yxy/datasets/AdverseNet/train",
    fix_sample = 3000,
    shuffle = True,
    num_workers = 3,
)

val_data_loader = dict(
    data_path = "/home/yxy/datasets/AdverseNet/val",
    fix_sample = 1000,
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)

test_data_loader = dict(
    data_path = "/home/yxy/datasets/AdverseNet/test(backup)",
    fix_sample = 10000,
    batch_size = 1,
    shuffle = False,
    num_workers = 1,
)


model_Stage1 = dict(

    type='AdverseNet_Stage1',
    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_, # 180
        tpv_w=tpv_w_, # 240
        tpv_z=tpv_z_, # 16
        nbr_classes=nbr_class, # 5
        in_dims=_dim_, # 192
        hidden_dims=2*_dim_, 
        out_dims=_dim_, 
        scale_h=scale_h, # 2
        scale_w=scale_w, # 2
        scale_z=scale_z # 2
    ),

    lidar_tokenizer=dict(
        type='CylinderEncoder_Seg', 
        grid_size=grid_size, 
        in_channels=9, 
        out_channels=256,
        fea_compre=32, 
        base_channels=32, 
        split=[16,16,16], 
        track_running_stats=track_running_stats, 
    ),


    tpv_encoder_decoder=dict(
        type='UNet_Stage1', 
        base_channel=32, 
    ),

)

model_Stage2_K1 = dict(

    type='AdverseNet_Stage2_K1',

    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_, # 180
        tpv_w=tpv_w_, # 240
        tpv_z=tpv_z_, # 16
        nbr_classes=nbr_class, # 5
        in_dims=_dim_, # 192
        hidden_dims=2*_dim_, 
        out_dims=_dim_, 
        scale_h=scale_h, # 2
        scale_w=scale_w, # 2
        scale_z=scale_z # 2
    ),

    lidar_tokenizer=dict(
        type='CylinderEncoder_Seg', 
        grid_size=grid_size, # [480, 360, 32]
        in_channels=9,
        out_channels=256,
        fea_compre=32, 
        base_channels=32,
        split=[16,16,16], 
        track_running_stats=track_running_stats, 
    ),

    tpv_encoder_decoder=dict(
        type='UNet_Stage2_K1', 
        base_channel=32, 
    ),

)

model_Stage2_K3 = dict(

    type='AdverseNet_Stage2_K3',

    tpv_aggregator=dict(
        type='TPVAggregator_Seg',
        tpv_h=tpv_h_, # 180
        tpv_w=tpv_w_, # 240
        tpv_z=tpv_z_, # 16
        nbr_classes=nbr_class, # 5
        in_dims=_dim_, # 192
        hidden_dims=2*_dim_, 
        out_dims=_dim_, 
        scale_h=scale_h, # 2
        scale_w=scale_w, # 2
        scale_z=scale_z # 2
    ),

    lidar_tokenizer=dict(
        type='CylinderEncoder_Seg', 
        grid_size=grid_size, # [480, 360, 32]
        in_channels=9, 
        out_channels=256,
        fea_compre=32, 
        base_channels=32,
        split=[16,16,16], 
        track_running_stats=track_running_stats, 
    ),


    tpv_encoder_decoder=dict(
        type='UNet_Stage2_K3', 
        base_channel=32, 
    ),

)