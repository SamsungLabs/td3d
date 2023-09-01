voxel_size = 0.1
padding = .08
n_points = 100000

class_names = ("building", "low vegetation", "med. vegetation", "high vegetation", "vehicle", "truck", "aircraft", 
                "militaryVehicle", "bike", "motorcycle", "light pole", "street sign", "clutter", "fence")
model = dict(
    type='TD3DInstanceSegmentor',
    voxel_size=voxel_size,
    backbone=dict(type='MinkResNet', in_channels=3, depth=34, norm='batch', return_stem=True, stride=1),
    neck=dict(
        type='NgfcTinySegmentationNeck',
        in_channels=(64, 128, 256, 512),
        out_channels=128),
    head=dict(
        type='TD3DInstanceHead',
        in_channels=128,
        n_reg_outs=6,
        n_classes=len(class_names),
        n_levels=4,
        padding=padding,
        voxel_size=voxel_size,
        unet=dict(
            type='MinkUNet14B', 
            in_channels=32, 
            out_channels=len(class_names) + 1,
            D=3),
        first_assigner=dict(
            type='S3DISAssigner',
            top_pts_threshold=6,
            label2level=[3, 1, 2, 3, 2, 2, 3, 2, 1, 1, 2, 1, 2, 3]),
        second_assigner=dict(
            type='MaxIoU3DAssigner',
            threshold=.25),
        roi_extractor=dict(
            type='Mink3DRoIExtractor',
            voxel_size=voxel_size,
            padding=padding,
            min_pts_threshold=10)),
    train_cfg=dict(num_rois=1),
    test_cfg=dict(
        nms_pre=100,
        iou_thr=.4,
        score_thr=.15,
        binary_score_thr=0.2))

optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=10, norm_type=2))
lr_config = dict(policy='step', warmup=None, step=[28, 32])
runner = dict(type='EpochBasedRunner', max_epochs=33)
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]

checkpoint_config = dict(interval=1, max_keep_ckpts=40)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]

dataset_type = 'STPLS3DInstanceSegDataset'
data_root = './data/stpls3d/'

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='LoadAnnotations3D',
        with_mask_3d=True,
        with_seg_3d=True),
    dict(
        type='PointSample', num_points=n_points),
    dict(
        type='PointSegClassMappingV2',
        valid_cat_ids=tuple(range(1, len(class_names) + 1)),
        max_cat_id=15),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0, 0],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[.1, .1, .1],
        shift_height=False),
    dict(
        type='BboxRecalculation'),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d',
                                 'pts_semantic_mask', 'pts_instance_mask'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='NormalizePointsColor', color_mean=None),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=5,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'stpls3d_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'stpls3d_infos_val.pkl',
        pipeline=test_pipeline,
        filter_empty_gt=False,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'stpls3d_infos_val.pkl',
        pipeline=test_pipeline,
        filter_empty_gt=False,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth')
)
