_base_ = [
    '../_base_/models/fpn_snn_r50.py',
    # '../_base_/models/fpn_r50.py',
    '../_base_/datasets/pascal_voc12_aug.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint_file = '/raid/ligq/lzx/spikeformerv2/seg/checkpoint/checkpoint-190.pth'
# checkpoint_file = '/raid/ligq/lzx/spikeformerv2/seg/mmsegmentation/tools/work_dirs/fpn_SDT_512x512_512_ade20k/iter_80000.pth'
# checkpoint_file = '/raid/ligq/lzx/output/seg_basic/ADE20k/checkpoint-best.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    type='EncoderDecoder',
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
        T=1,  # T=1 & T=4
        decode_mode='snn',
        ),
    neck=dict(
        in_channels=[32, 64, 128, 360],
        out_channels=128,
        act_cfg=None),
    decode_head=dict(
        in_channels=[128, 128, 128, 128],
        channels=128,
        num_classes=150,
        act_cfg=None)
        ,
    )

# load_from = checkpoint_file
gpu_multiples = 1  # we use 8 gpu instead of 4 in mmsegmentation, so lr*2 and max_iters/2
# optimizer

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.001, betas=(0.9, 0.999),  weight_decay=0.005),
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=10.)}
        ))
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
    type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
train_dataloader = dict(batch_size=10)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader