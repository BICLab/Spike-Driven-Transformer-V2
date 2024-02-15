_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# pretrained = "/lxh/spike-driven-transformer/mmdet3/work_dirs/mask-rcnn_sdt-t_fpn_1x_coco/epoch_12.pth"
# pretrained = "/lxh/checkpoint-15M.pth"
# pretrained = None
# augmentation strategy originates from DETR / Sparse RCNN
train_pipeline = [

    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    # dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='Spiking_vit_MetaFormer',
        # embed_dim=[128, 256, 512, 640],
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        qkv_bias=False,
        depths=8,
        sr_ratios=1,
        T=4,
        # init_cfg=dict(type='Pretrained', checkpoint=pretrained)
    ),
    neck=dict(
        type='SpikeFPN',
        # in_channels=[128, 256, 512, 640],
        in_channels=[64, 128, 256, 360],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='SpikeRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
        #                loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0),
        # loss_bbox=dict(type='GIoULoss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='SpikeStandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='SharedSpike2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=80,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0),
            # loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
            #                loss_weight=2.0),
            # loss_bbox=dict(type='GIoULoss', loss_weight=1.0), #有bug，loss变负数
        ),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='SpikeFCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=80,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),)

max_epochs = 8
max_iter = max_epochs * 19545
train_cfg = dict(max_epochs=max_epochs)

# learning rate
# param_scheduler = [
#     dict(
#         type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
#         end=1000),
#     dict(
#         type='MultiStepLR',
#         begin=0,
#         end=max_epochs,
#         by_epoch=True,
#         milestones=[27, 33],
#         gamma=0.1)
# ]

param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=4000),
    dict(
        type='LinearLR', start_factor=1,end_factor=0.01, by_epoch=False, begin=1,
        end=max_iter),
    # dict(
    #     type='MultiStepLR',
    #     begin=0,
    #     end=max_epochs,
    #     by_epoch=True,
    #     milestones=[27, 33],
    #     gamma=0.1)

]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            # 'absolute_pos_embed': dict(decay_mult=0.),
            # 'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0002,
        betas=(0.9, 0.999),
        weight_decay=0.05),
        # optimizer=dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)

)

# fp16 = dict(loss_scale=512.)
