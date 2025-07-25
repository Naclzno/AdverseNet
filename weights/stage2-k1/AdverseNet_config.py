_dim_ = 24
dataset_params = dict(
    fill_label=0,
    fixed_volume_space=True,
    flip_aug=True,
    grid_size_vox=[
        480,
        360,
        32,
    ],
    ignore_label=0,
    label_mapping='./config/label_mapping/AdverseNet.yaml',
    max_volume_space=[
        50,
        3.1415926,
        2,
    ],
    min_volume_space=[
        0,
        -3.1415926,
        -4,
    ],
    rotate_aug=True,
    scale_aug=True,
    trans_std=[
        0.1,
        0.1,
        0.1,
    ],
    transform_aug=True)
find_unused_parameters = False
gpu_ids = range(0, 1)
grad_max_norm = 35
grid_size = [
    480,
    360,
    32,
]
load_from = '/home/yxy/work/fifth/stage1/epoch_9.pth'
max_epochs = 40
model_Stage1 = dict(
    lidar_tokenizer=dict(
        base_channels=32,
        fea_compre=32,
        grid_size=[
            480,
            360,
            32,
        ],
        in_channels=9,
        out_channels=256,
        split=[
            16,
            16,
            16,
        ],
        track_running_stats=False,
        type='CylinderEncoder_Seg'),
    tpv_aggregator=dict(
        hidden_dims=48,
        in_dims=24,
        nbr_classes=5,
        out_dims=24,
        scale_h=2,
        scale_w=2,
        scale_z=2,
        tpv_h=180,
        tpv_w=240,
        tpv_z=16,
        type='TPVAggregator_Seg'),
    tpv_encoder_decoder=dict(base_channel=32, type='UNet_Stage1'),
    type='AdverseNet_Stage1')
model_Stage2_K1 = dict(
    lidar_tokenizer=dict(
        base_channels=32,
        fea_compre=32,
        grid_size=[
            480,
            360,
            32,
        ],
        in_channels=9,
        out_channels=256,
        split=[
            16,
            16,
            16,
        ],
        track_running_stats=False,
        type='CylinderEncoder_Seg'),
    tpv_aggregator=dict(
        hidden_dims=48,
        in_dims=24,
        nbr_classes=5,
        out_dims=24,
        scale_h=2,
        scale_w=2,
        scale_z=2,
        tpv_h=180,
        tpv_w=240,
        tpv_z=16,
        type='TPVAggregator_Seg'),
    tpv_encoder_decoder=dict(base_channel=32, type='UNet_Stage2_K1'),
    type='AdverseNet_Stage2_K1')
model_Stage2_K3 = dict(
    lidar_tokenizer=dict(
        base_channels=32,
        fea_compre=32,
        grid_size=[
            480,
            360,
            32,
        ],
        in_channels=9,
        out_channels=256,
        split=[
            16,
            16,
            16,
        ],
        track_running_stats=False,
        type='CylinderEncoder_Seg'),
    tpv_aggregator=dict(
        hidden_dims=48,
        in_dims=24,
        nbr_classes=5,
        out_dims=24,
        scale_h=2,
        scale_w=2,
        scale_z=2,
        tpv_h=180,
        tpv_w=240,
        tpv_z=16,
        type='TPVAggregator_Seg'),
    tpv_encoder_decoder=dict(base_channel=32, type='UNet_Stage2_K3'),
    type='AdverseNet_Stage2_K3')
nbr_class = 5
optimizer_wrapper_stage1 = dict(
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.01))
optimizer_wrapper_stage2 = dict(
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.01))
print_freq = 50
scale_h = 2
scale_w = 2
scale_z = 2
test_data_loader = dict(
    batch_size=1,
    data_path='/home/yxy/datasets/AdverseNet/test(backup)',
    fix_sample=10000,
    num_workers=1,
    shuffle=False)
tpv_h_ = 180
tpv_w_ = 240
tpv_z_ = 16
track_running_stats = False
train_data_loader = dict(
    data_path='/home/yxy/datasets/AdverseNet/train',
    fix_sample=3000,
    num_workers=3,
    shuffle=True)
unique_label = [
    1,
    2,
    3,
    4,
]
val_data_loader = dict(
    batch_size=1,
    data_path='/home/yxy/datasets/AdverseNet/val',
    fix_sample=1000,
    num_workers=1,
    shuffle=False)
work_dir = '/home/yxy/work/ablation/expansion/0.03'
