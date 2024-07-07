# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from mmseg.ops import resize
from typing import List

from ..builder import HEADS
from ..utils import SelfAttentionBlock as _SelfAttentionBlock
from .cascade_decode_head import BaseCascadeDecodeHead


class ContextModule(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 branches:List[int]=None,
                 branches_groups=None,
                 in_project=False,
                 in_project_kernel_size=5,
                 in_project_groups=1,
                 out_project=True,
                 out_project_kernel_size=1,
                 out_project_groups=1,
                 channels=None,
                 **kwargs,
                 ):
        super().__init__()
        if channels is None:
            channels=in_channels
        if branches is None:
            branches=[]
        if branches_groups is None:
            branches_groups=channels
        #暂时不用
        self.conv0 = ConvModule(
            in_channels,
            channels,
            in_project_kernel_size,
            padding=in_project_kernel_size//2,
            groups=in_project_groups,
            **kwargs,
        ) if in_project else nn.Identity()
        self.convs = nn.ModuleList([
            nn.Sequential(
                ConvModule(channels, channels, (1, b), padding=(0, b//2), groups=branches_groups,**kwargs),
                ConvModule(channels, channels, (b, 1), padding=(b//2, 0), groups=branches_groups,**kwargs)
            )
            for b in branches
        ])
        self.conv2 = ConvModule(channels, out_channels, out_project_kernel_size,padding=out_project_kernel_size//2,
                                groups=out_project_groups,**kwargs) if out_project else nn.Identity()

    def forward(self, x):
        out = self.conv0(x)

        for i in self.convs:
            out=out+i(x)

        out = self.conv2(out)

        return out


class ConfidenceWeightedNonParametricProto(nn.Module):

    def __init__(self, scale):
        super(ConfidenceWeightedNonParametricProto, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        proto = torch.matmul(probs, feats)
        proto = proto.permute(0, 2, 1).contiguous().unsqueeze(3)
        return proto


class ObjectAttentionBlock(_SelfAttentionBlock):

    def __init__(self, in_channels, channels, scale, conv_cfg, norm_cfg,
                 act_cfg):
        if scale > 1:
            query_downsample = nn.MaxPool2d(kernel_size=scale)
        else:
            query_downsample = None
        super(ObjectAttentionBlock, self).__init__(
            key_in_channels=in_channels,
            query_in_channels=in_channels,
            channels=channels,
            out_channels=in_channels,
            share_key_query=False,
            query_downsample=query_downsample,
            key_downsample=None,
            key_query_num_convs=2,
            key_query_norm=True,
            value_out_num_convs=1,
            value_out_norm=True,
            matmul_norm=True,
            with_out=True,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.bottleneck = ConvModule(
            in_channels * 2,
            in_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, query_feats, key_feats):
        """Forward function."""
        context = super(ObjectAttentionBlock,
                        self).forward(query_feats, key_feats)
        output = self.bottleneck(torch.cat([context, query_feats], dim=1))
        if self.query_downsample is not None:
            output = resize(query_feats)

        return output


@HEADS.register_module()
class PMDAV2(BaseCascadeDecodeHead):


    def __init__(self, feat_channels,bottleneck_kernel_size=3,branches_groups=1, scale=1, **kwargs):
        super(PMDAV2, self).__init__(**kwargs)
        self.feat_channels = feat_channels
        self.bottleneck_kernel_size = bottleneck_kernel_size
        self.branches_groups = branches_groups
        self.scale = scale
        self.pm = ObjectAttentionBlock(
            self.channels,
            self.feat_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.npp = ConfidenceWeightedNonParametricProto(self.scale)

        if type(self.bottleneck_kernel_size) is int:
            self.ms = ConvModule(
                self.in_channels,
                self.channels,
                self.bottleneck_kernel_size,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        elif type(self.bottleneck_kernel_size) is list:
            self.ms = ContextModule(
                self.in_channels,
                self.channels,
                branches=self.bottleneck_kernel_size,
                branches_groups=branches_groups,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg
            )

    def forward(self, inputs, prev_output):
        """Forward function."""
        x = self._transform_inputs(inputs)
        feats = self.ms(x)
        protos = self.npp(feats, prev_output)
        d = self.pm(feats, protos)
        output = self.cls_seg(d)

        return output
