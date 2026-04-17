# BEV Velocity Estimation Configuration
model = dict(
    type='BEVVelocity',
    backbone=dict(
        type='BEVBackbone',
        num_cameras=6,
        bev_h=32,
        bev_w=32,
        bev_c=256,
    ),
    velocity_head=dict(
        type='VelocityHead',
        bev_channels=256,
        output_channels=2,  # vx, vy
    ),
)

# Dataset (with temporal data)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='NuScenesDataset',
        data_root='./data',
        ann_file='./data/nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
        ),
        test_mode=False,
        with_velocity=True,  # Enable velocity annotations
    ),
)

# Training
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
lr_config = dict(
    policy='step',
    step=[16, 22],
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
)
runner = dict(type='EpochBasedRunner', max_epochs=24)

# Loss
loss_velocity = dict(type='L1Loss', loss_weight=1.0)