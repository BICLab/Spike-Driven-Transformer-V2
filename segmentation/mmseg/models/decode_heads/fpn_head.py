# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import Upsample, resize
from .decode_head import BaseDecodeHead
from spikingjelly.clock_driven.neuron import (
    MultiStepParametricLIFNode,
    MultiStepLIFNode,
)
from spikingjelly.clock_driven import layer
from spikingjelly.clock_driven.functional import reset_net
@MODELS.register_module()
class FPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))


    def forward(self, inputs):
        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output

@MODELS.register_module()
class FPNHead_SNN(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super(FPNHead_SNN, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        # self.size = size
        self.scale_heads = nn.ModuleList()
        self.decode_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy")
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                if feature_strides[i] != feature_strides[0]:
                    scale_head.append(
                        nn.Sequential(
                            MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy"),
                            layer.SeqToANNContainer(
                                ConvModule(
                                    self.in_channels[i] if k == 0 else self.channels,
                                    self.channels,
                                    3,
                                    padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=None),
                                Upsample(
                                    scale_factor=2,
                                    mode='bilinear',
                                    align_corners=self.align_corners),

                            ),
                        )
                    )
                else:
                    scale_head.append(
                        nn.Sequential(
                            MultiStepLIFNode(tau=2.0, detach_reset=True, backend="cupy"),
                            layer.SeqToANNContainer(
                                ConvModule(
                                    self.in_channels[i] if k == 0 else self.channels,
                                    self.channels,
                                    3,
                                    padding=1,
                                    conv_cfg=self.conv_cfg,
                                    norm_cfg=self.norm_cfg,
                                    act_cfg=None)),
                        )
                    )

            self.scale_heads.append(nn.Sequential(*scale_head))
            # self.Upsample = nn.Upsample()

    def forward(self, inputs):

        T, B = inputs[0].shape[0], inputs[0].shape[1]
        # print(T)
        input_map = [input_value.flatten(0, 1) for input_value in inputs]
        x = self._transform_inputs(input_map)
        tmp = []
        for feature in x:
            _, C, H, W = feature.shape
            tmp.append(feature.reshape(T, B, C, H, W))

        x = tmp
        output = self.scale_heads[0](x[0])
        T, B, C, H, W = output.shape
        # import pdb; pdb.set_trace()
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]).flatten(0, 1),
                size=output.shape[3:],
                mode='bilinear',
                align_corners=self.align_corners).reshape(T, B, C, H, W)


        T, B, _, H, W = output.shape
        output = self.cls_seg(self.decode_lif(output).flatten(0, 1))
        num_class = output.shape[1]
        return output.reshape(T, B, num_class, H, W).mean(0)
