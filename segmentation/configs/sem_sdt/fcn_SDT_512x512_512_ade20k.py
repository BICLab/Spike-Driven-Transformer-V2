_base_ = [
    '../_base_/models/fcn_snn.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
# model settings
# checkpoint_file = '/raid/ligq/lzx/spikeformerv2/seg/checkpoint/checkpoint-199.pth'
# checkpoint_file = '/raid/ligq/lzx/output/seg_basic/ADE20k/checkpoint-best.pth'
checkpoint_file = "/raid/ligq/lzx/ckpt/sdtv2/T4/checkpoint-15M.pth"
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    # pretrained='/raid/ligq/lzx/output/seg_basic/ADE20k/checkpoint-best.pth',
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        type='Spiking_vit_MetaFormer',
        img_size_h=512,
        img_size_w=512,
        patch_size=16,
        embed_dim=[64, 128, 256, 360],
        num_heads=8,
        mlp_ratios=4,
        in_channels=3,
        num_classes=150,
        qkv_bias=False,
        depths=8,
        sr_ratios=1,
        T=4,
        decode_mode='snn',
    ),
    decode_head=dict(
        _delete_=True,
        type='FCNHead_SNN',
        in_channels=360,
        channels=360,
        num_convs=0,
        dropout_ratio=0.0,
        concat_input=False,
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    dtype='float16',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999),  weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys={
            # 'neck': dict(lr_mult=2.0),
            'head': dict(lr_mult=2.0)}
        ),
    # clip_grad=dict(max_norm=0.01, norm_type=2)
    )
#
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]
# policy='poly', power=0.9, min_lr=0.0, by_epoch=False
optimizer_config = dict()
# learning policy
lr_config = dict(warmup_iters=1500)
# runtime settings

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=160000, val_interval=5000)
train_dataloader = dict(batch_size=4)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

vis_backends = [dict(type='LocalVisBackend'),
                dict(type='TensorboardVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
