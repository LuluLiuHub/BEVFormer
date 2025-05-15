# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Lulu Liu
# ---------------------------------------------
#  Further modified to incorporate TensorCP fallback

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule
from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext('_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

class TensorCPField(nn.Module):
    def __init__(self, x_size, y_size, t_size, rank=32, feature_dim=256):
        super().__init__()
        self.rank = rank
        self.A = nn.Parameter(torch.randn(rank, x_size))
        self.B = nn.Parameter(torch.randn(rank, y_size))
        self.C = nn.Parameter(torch.randn(rank, t_size))
        self.proj = nn.Linear(rank, feature_dim)

    def query(self, x_idx, y_idx, t_idx):
        x_idx = (x_idx * (self.A.shape[1] - 1)).long().clamp(0, self.A.shape[1] - 1)
        y_idx = (y_idx * self.B.shape[1] - 1).long().clamp(0, self.B.shape[1] - 1)
        t_idx = (t_idx * self.C.shape[1] - 1).long().clamp(0, self.C.shape[1] - 1)
        Ax = self.A[:, x_idx]
        By = self.B[:, y_idx]
        Ct = self.C[:, t_idx]
        feat = Ax * By * Ct
        feat = feat.sum(dim=0).T
        return self.proj(feat)

@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=4,
                 num_bev_queue=2, im2col_step=64, dropout=0.1, batch_first=True,
                 norm_cfg=None, init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError('embed_dims must be divisible by num_heads')
        dim_per_head = embed_dims // num_heads
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.im2col_step = im2col_step

        self.sampling_offsets = nn.Linear(embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points)
        self.confidence = nn.Linear(embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.tensorcp = TensorCPField(x_size=128, y_size=128, t_size=2)  # Add actual dimensions
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self, query, key=None, value=None, identity=None, query_pos=None,
                key_padding_mask=None, reference_points=None, spatial_shapes=None,
                level_start_index=None, flag='decoder', **kwargs):

        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs * 2, len_bev, c)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        value = value.reshape(bs * self.num_bev_queue, -1, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points).softmax(-1)
        attention_weights = attention_weights.view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)

        conf_level = self.confidence(query).view(bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points).softmax(dim=3)
        conf_avg = conf_level.mean(dim=(2, 4, 5))
        conf_avg = conf_avg.softmax(dim=2)

        curr_weight = conf_avg[:, :, 1].permute(1, 0).unsqueeze(1)
        hist_weight = conf_avg[:, :, 0].permute(1, 0).unsqueeze(1)

        low_conf_mask = (hist_weight < 0.3).squeeze(1)
        if low_conf_mask.any():
            x, y = reference_points[low_conf_mask, :, 0, 0].flatten(), reference_points[low_conf_mask, :, 0, 1].flatten()
            t = torch.zeros_like(x)
            fallback_feat = self.tensorcp.query(x, y, t)
            output_tensorcp = torch.zeros_like(query[:, :, :self.embed_dims])
            output_tensorcp[low_conf_mask.transpose(0, 1)] = fallback_feat.view_as(output_tensorcp[low_conf_mask.transpose(0, 1)])
        else:
            output_tensorcp = 0

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]

        output = MultiScaleDeformableAttnFunction_fp32.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights.permute(0, 3, 1, 2, 4, 5).reshape(bs * self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points),
            self.im2col_step)

        output = output.permute(1, 2, 0).view(num_query, self.embed_dims, bs, self.num_bev_queue)
        output = (curr_weight * output[:, :, :, 1] + hist_weight * output[:, :, :, 0])
        output = output.permute(2, 0, 1) + output_tensorcp
        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
