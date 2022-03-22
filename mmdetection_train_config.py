# model settings
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[1],
            ratios=[0.5, 1.0, 2.0],
            base_sizes=[16, 32, 48, 68, 168],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=1
            ,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0))),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=20)
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ))

# Dataset settings
dataset_type = 'VOCDataset'
classes = ('1',)  # 与标注中的类别对应，可以加载数据, '1' means DBD

img_norm_cfg = dict(
    mean=[110.460, 110.460, 110.460], std=[71.994, 71.994, 71.994], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(842, 550), keep_ratio=True),
    dict(type='RandomShift'),
    dict(type='RandomCrop', crop_type='relative_range', crop_size=(0.8, 0.8)),
    dict(type='Rotate', img_fill_val=0),
    dict(type='RandomFlip', flip_ratio=[0.2, 0.2, 0.2], direction=['horizontal', 'vertical', 'diagonal']),
    dict(type='PhotoMetricDistortion'),
    # dict(type='Expand', ratio_range=(1, 2)),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(842, 550),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(
        dataset=dict(
          type=dataset_type,
          img_prefix='/data/ljm/LRCN/VOC2007/',
          classes=classes,
          ann_file='/data/ljm/LRCN/VOC2007/ImageSets/Main/train.txt',
          pipeline=train_pipeline)),
    val=dict(
        type=dataset_type,
        img_prefix='/data/ljm/LRCN/VOC2007/',
        classes=classes,
        ann_file='/data/ljm/LRCN/VOC2007/ImageSets/Main/val.txt',
        pipeline=test_pipeline),
    test=dict(
        img_prefix='/data/ljm/LRCN/VOC2007/',
        classes=classes,
        ann_file='/data/ljm/LRCN/VOC2007/ImageSets/Main/val.txt',
        pipeline=test_pipeline))

# optimizer
evaluation = dict(interval=1, metric='mAP')
optimizer = dict(type='SGD', lr=0.02/8, momentum=0.9, weight_decay=0.0001)


checkpoint_config = dict(interval=1)
load_from = '/data/ljm/mmdetection/checkpoints/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'
work_dir = '/data/ljm/LRCN/LITS2017肝脏肿瘤分割挑战数据集/DBD/Faster_R-CNN_50_2rd'
gpu_ids = range(0, 1)
