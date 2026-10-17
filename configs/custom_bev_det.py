# BEV Detection Configurations
model = dict(
    type='BEVDetector',
    backbone=dict(
        type='BEVBackbone',
        num_cameras=6,
        bev_h=32,
        bev_w=32,
        bev_c=256,
    ),
    neck=dict(
        type='DetectionHead',
        bev_channels=256,
    ),
    bbox_head=dict(
        type='BEVDetHead',
        num_classes=10,  # NuScenes classes
        in_channels=256,
        feat_channels=256,
        use_direction_classifier=True,
        bbox_coder=dict(
            type='BEVDetBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            out_size_factor=8,
            voxel_size=[0.1, 0.1, 0.2],
            code_size=9,
        ),
    ),
)

# Dataset
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
    ),
    val=dict(
        type='NuScenesDataset',
        data_root='./data',
        ann_file='./data/nuscenes_infos_val.pkl',
        pipeline=val_pipeline,
        classes=class_names,
        modality=dict(
            use_camera=True,
            use_lidar=False,
            use_radar=False,
        ),
        test_mode=True,
    ),
)

# Training
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3,
)
runner = dict(type='EpochBasedRunner', max_epochs=24)

# Evaluation
evaluation = dict(interval=1, pipeline=val_pipeline)
