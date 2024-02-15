# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='/raid/ligq/lzx/spikeformerv2/seg/checkpoint/checkpoint-199.pth',
    backbone=dict(
        # init_cfg=dict(type='Pretrained', checkpoint="/raid/ligq/lzx/spikeformerv2/seg/checkpoint/checkpoint-199.pth"),
        type='Sdtv2',
        img_size_h=512,
        img_size_w=512,
        embed_dim=[128, 256, 512, 640],
        num_classes=150,
        T=1,
        qkv_bias=False,
        decode_mode='snn',
        ),
    neck=dict(
        type='FPN_SNN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead_SNN',
        in_channels=[256, 256, 256, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
