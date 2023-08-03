voxel_size = .02
padding = .08
n_points = 100000

class_names = ('cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
               'bookshelf', 'picture', 'counter', 'desk', 'curtain',
               'refrigerator', 'showercurtrain', 'toilet', 'sink', 'bathtub',
               'garbagebin')
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
            type='NgfcV2Assigner',
            min_pts_threshold=18,
            top_pts_threshold=8,
            padding=padding),
        second_assigner=dict(
            type='MaxIoU3DAssigner',
            threshold=.37),
        roi_extractor=dict(
            type='Mink3DRoIExtractor',
            voxel_size=voxel_size,
            padding=padding,
            min_pts_threshold=10)),
    train_cfg=dict(num_rois=2),
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

dataset_type = 'ScanNetInstanceSegV2Dataset'
data_root = './data/scannet/'

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
        type='GlobalAlignment', rotation_axis=2),
    dict(
        type='PointSample', num_points=n_points),
    dict(
        type='PointSegClassMappingV2',
        valid_cat_ids=(3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28,
                       33, 34, 36, 39),
        max_cat_id=40),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='Elastic'),
    dict(
        type='MiniMosaic',
        remaining_points_thr=0.3,
        n_src_points=n_points),
    dict(
        type='GlobalRotScaleTransV2',
        rot_range_z=[-3.14, 3.14],
        rot_range_x_y=[-0.1308, 0.1308],
        scale_ratio_range=[.8, 1.2],
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
    samples_per_gpu=6,
    workers_per_gpu=10,
    train=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'scannet_td3d_infos_train.pkl',
            pipeline=train_pipeline,
            filter_empty_gt=True,
            classes=class_names,
            box_type_3d='Depth')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_td3d_infos_val.pkl',
        pipeline=test_pipeline,
        filter_empty_gt=False,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'scannet_td3d_infos_val.pkl',
        pipeline=test_pipeline,
        filter_empty_gt=False,
        classes=class_names,
        test_mode=True,
        box_type_3d='Depth')
)
