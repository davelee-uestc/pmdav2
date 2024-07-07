# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import ModuleList
from torch import nn

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder


@SEGMENTORS.register_module()
class CascadeEncoderDecoder(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 backbone,
                 decode_head,
                 inference_use_first_head=False,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        self.num_stages = num_stages
        self.inference_use_first_head = inference_use_first_head
        super(CascadeEncoderDecoder, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = ModuleList()
        for i in range(self.num_stages):
            self.decode_head.append(builder.build_head(decode_head[i]))
        self.align_corners = self.decode_head[-1].align_corners
        self.num_classes = self.decode_head[-1].num_classes
        self.out_channels = self.decode_head[-1].out_channels

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        if not self.first_head_cas:
            out = self.decode_head[0].forward_test(x, img_metas, self.test_cfg)
        else:
            out = self.decode_head[0].forward_test(x,None, img_metas, self.test_cfg)

        if not self.inference_use_first_head:
            for i in range(1, self.num_stages):
                out = self.decode_head[i].forward_test(x, out, img_metas,
                                                       self.test_cfg)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    @property
    def first_head_cas(self):
        from mmseg.models.decode_heads.cascade_decode_head import BaseCascadeDecodeHead
        return issubclass(type(self.decode_head[0]), BaseCascadeDecodeHead)

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg,seg_weight=None,
                      detach_feats=False):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        prev_outputs=None

        if detach_feats:
            assert False

        if not self.first_head_cas:
            loss_decode = self.decode_head[0].forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg,seg_weight=seg_weight)
        else:
            loss_decode = self.decode_head[0].forward_train(
                x,None, img_metas, gt_semantic_seg, self.train_cfg, seg_weight=seg_weight)

        losses.update(add_prefix(loss_decode, 'decode_0'))

        for i in range(1, self.num_stages):
            # forward test again, maybe unnecessary for most methods.
            if i == 1:
                if prev_outputs is None and not self.first_head_cas:
                    prev_outputs = self.decode_head[0].forward_test(
                    x, img_metas, self.test_cfg)
                elif prev_outputs is None and self.first_head_cas:
                    prev_outputs = self.decode_head[0].forward_test(
                    x,None, img_metas, self.test_cfg)
            else:
                prev_outputs = self.decode_head[i - 1].forward_test(
                    x, prev_outputs, img_metas, self.test_cfg)
            loss_decode = self.decode_head[i].forward_train(
                x, prev_outputs, img_metas, gt_semantic_seg, self.train_cfg,seg_weight=seg_weight)
            losses.update(add_prefix(loss_decode, f'decode_{i}'))

        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        if not self.first_head_cas:
            prev_outputs = self.decode_head[0].forward_test(
                x, img_metas, self.test_cfg)
        else:
            prev_outputs = self.decode_head[0].forward_test(
                x, None, img_metas, self.test_cfg)
        if not self.inference_use_first_head:
            for i in range(1, self.num_stages):
                    prev_outputs = self.decode_head[i].forward_test(
                        x, prev_outputs, img_metas, self.test_cfg)
        return prev_outputs