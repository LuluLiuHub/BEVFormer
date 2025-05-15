# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
#  Modified by Lulu Liu
# ---------------------------------------------

from projects.mmdet3d_plugin.models.utils.bricks import run_time
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION
import math
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.utils import (ConfigDict, build_from_cfg, deprecated_api_warning,
                        to_2tuple)

from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])

# Add TensorCP for auxilary architecture when the conf is low
# it takes the x_size, y_size, t_size, scale
# and get feature embeddings 256
# for t=0, use sampling locations
# for t=1, use reference points
class TensorCPField(nn.Module):
    class TensorCPField(nn.Module):
    def __init__(self, x_size, y_size, t_size, s_size, rank=32, feature_dim=256):
        super().__init__()
        self.rank = rank
        self.x_size = x_size
        self.y_size = y_size
        self.t_size = t_size
        self.s_size = s_size

        self.A = nn.Parameter(torch.randn(rank, x_size))  # x
        self.B = nn.Parameter(torch.randn(rank, y_size))  # y
        self.C = nn.Parameter(torch.randn(rank, t_size))  # time
        self.D = nn.Parameter(torch.randn(rank, s_size))  # scale

        self.proj = nn.Linear(rank, feature_dim)

    def interpolate_1d(self, tensor, idx_norm, axis_size):
        idx = idx_norm * (axis_size - 1)
        idx0 = torch.floor(idx).long().clamp(0, axis_size - 2)
        idx1 = idx0 + 1
        weight = (idx - idx0.float()).unsqueeze(0)
        v0 = tensor[:, idx0]
        v1 = tensor[:, idx1]
        return (1 - weight) * v0 + weight * v1

    def forward(self, x_norm, y_norm, t_norm, s_norm):
        Ax = self.interpolate_1d(self.A, x_norm, self.x_size)  # [rank, N]
        By = self.interpolate_1d(self.B, y_norm, self.y_size)
        Ct = self.interpolate_1d(self.C, t_norm, self.t_size)
        Ds = self.interpolate_1d(self.D, s_norm, self.s_size)

        feat = Ax * By * Ct * Ds  # [rank, N]
        feat = feat.permute(1, 0).contiguous()  # [N, rank]
        return self.proj(feat)  # [N, feature_dim]

@ATTENTION.register_module()
class TemporalSelfAttention(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=2,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.sampling_offsets = nn.Linear(
            embed_dims*self.num_bev_queue, num_bev_queue*num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.confidence = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        # Add fall back to tensorCP confidence
        self.grid_conf = nn.Linear(embed_dims, 1)
        self.init_weights()
        self.tensorcp = None

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                flag='decoder',

                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            assert self.batch_first
            bs, len_bev, c = query.shape
            value = torch.stack([query, query], 1).reshape(bs*2, len_bev, c)

            # value = torch.cat([query, query], 0)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
        bs,  num_query, embed_dims = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        assert self.num_bev_queue == 2

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)

        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.reshape(bs*self.num_bev_queue,
                              num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bs*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]

        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        
        if torch.cuda.is_available() and value.is_cuda:
            # run both architecture in parallel to save time
            stream_deform = torch.cuda.Stream()
            stream_tensorcp = torch.cuda.Stream()
            # using fp16 deformable attention is unstable because it performs many sum operations
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            with torch.cuda.stream(stream_deform):
                output = MultiScaleDeformableAttnFunction.apply(
                    value, spatial_shapes, level_start_index, sampling_locations,
                    attention_weights, self.im2col_step)
        else:
        # Will not be used
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        # output shape (bs*num_bev_queue, num_query, embed_dims)
        # (bs*num_bev_queue, num_query, embed_dims)-> (num_query, embed_dims, bs*num_bev_queue)
        output = output.permute(1, 2, 0)

        # fuse history value and current value
        # (num_query, embed_dims, bs*num_bev_queue)-> (num_query, embed_dims, bs, num_bev_queue)
        output = output.view(num_query, embed_dims, bs, self.num_bev_queue)
        
        #################### Need modification ############################
        if self.tensorcp is None:
            max_h = spatial_shapes[:, 0].max().item()
            max_w = spatial_shapes[:, 1].max().item()
            self.tensorcp = TensorCPField(
                x_size=int(max_w),
                y_size=int(max_h),
                t_size=2,
                s_size=self.num_levels,
                rank=32,
                feature_dim=self.embed_dims
            ).to(query.device)


        with torch.cuda.stream(stream_tensorcp):
            # Call tensorCP to get another set of result:
            num_heads = self.num_heads
            num_levels = self.num_levels
            num_points = self.num_points

            coords_hist = sampling_locations[:bs]  # use history half
            coords_hist = coords_hist.permute(0, 2, 1, 3, 4, 5).contiguous()  # [bs, heads, queries, levels, points, 2]
            coords_hist = coords_hist.view(bs * num_heads * num_query * num_levels * num_points, 2)
            x, y = coords_hist[:, 0], coords[:, 1]

            # Coordinates for t=1: reference_points (current)
            # reference_points: [bs, num_query, num_levels, 2]
            coords_curr = reference_points  # [bs, num_query, num_levels, 2]
            coords_curr = coords_curr.unsqueeze(2).expand(-1, -1, num_heads, -1, -1)  # [bs, queries, heads, levels, 2]
            coords_curr = coords_curr.unsqueeze(4).expand(-1, -1, -1, -1, num_points, -1)  # [bs, queries, heads, levels, points, 2]
            coords_curr = coords_curr.contiguous().view(bs * num_query * num_heads * num_levels * num_points, 2)
                    
            x_curr, y_curr = coords_curr[:, 0], coords_curr[:, 1]
            t_hist = torch.zeros_like(x)       
            t_curr = torch.ones_like(x)       

            # Form scale to pass
            scale_ids = torch.arange(num_levels, device=query.device).float()
            scale_ids = scale_ids / (num_levels - 1)  # normalize to [0,1]
            scale_tensor = scale_ids[None, None,None, :, None].expand(bs, num_query, num_heads, num_levels, num_points)
            scale_tensor = scale_tensor.contiguous().view(-1)


            feat_hist = self.tensorcp(x, y, t_hist,scale_tensor)  # [M, embed_dim]
            feat_curr = self.tensorcp(x_curr, y_curr, t_curr,scale_tensor)

            feat_hist = feat_hist.view(bs, num_query, num_heads, num_levels, num_points, self.embed_dims)
            feat_curr = feat_curr.view(bs, num_query, num_heads, num_levels, num_points, self.embed_dims)

            # Use current attention weights:
            attn_weights = attention_weights.view(bs, self.num_bev_queue, num_query, num_heads, num_levels, num_points)
            attn_hist = attn_weights[:, 0]  # [bs, num_query, num_heads, num_levels, num_points]
            attn_curr = attn_weights[:, 1]

            attn_hist = attn_hist.unsqueeze(-1)  # [bs, num_query, num_heads, num_levels, num_points, 1]
            attn_curr = attn_curr.unsqueeze(-1)
            
            fused_hist = (feat_hist * attn_hist).sum(dim=(2, 3, 4))  # [bs, num_query, embed_dim]
            fused_curr = (feat_curr * attn_curr).sum(dim=(2, 3, 4))

            fused_hist = fused_hist.permute(1, 2, 0)  # [num_query, embed_dim, bs]
            fused_curr = fused_curr.permute(1, 2, 0)

            output_tensorcp = torch.stack([fused_hist, fused_curr], dim=-1)  # [num_query, embed_dim, bs, 2]
       
        
        fall_back_score = torch.sigmoid(self.grid_conf(query))  # [bs, num_query, 1]
        fall_back_score = fall_back_score.permute(1, 2, 0)  # [num_query, 1, bs]
        fall_back_score = fall_back_score.expand(-1, self.embed_dims, -1)  # [num_query, embed_dim, bs]
        fall_back_score = fall_back_score.unsqueeze(-1)  # [num_query, embed_dim, bs, 1]

        torch.cuda.synchronize()

        output = fall_back_score * output + (1 - fall_back_score) * output_tensorcp

        
        #### Add confidence to the attention weights
        conf_level = self.confidence(query).view(
            bs, num_query,  self.num_heads, self.num_bev_queue, self.num_levels, self.num_points)
        conf_level = conf_level.softmax(dim=3)
      
        # Normalize across BEV queues (current vs. history) for each point
        # total_weight = conf_level.sum(dim=3, keepdim=True) + 1e-6  # [bs, num_query, num_heads, 1, num_levels, num_points]
        # conf_level_normalized = conf_level / total_weight 

        # Average over heads, levels, points -> shape: (bs, num_query, num_bev_queue)
        conf_level_avg = conf_level.mean(dim=(2, 4, 5))  # -> (bs, num_query, num_bev_queue)
        # Softmax over time (num_bev_queue)
        conf_level_avg = conf_level_avg.softmax(dim=2)
        # use only 2 dimensions
       
        curr_bev_weight =  conf_level_avg[:, :, 1].permute(1, 0).unsqueeze(1)  # (bs, num_query)
        hist_bev_weight =  conf_level_avg[:, :, 0].permute(1, 0).unsqueeze(1)  # (bs, num_query)

        # Dynamically fuse
        output = (curr_bev_weight * output[:, :, :, 1] + 
        hist_bev_weight * output[:, :, :, 0])
 
        # output = output.mean(-1)

        # (num_query, embed_dims, bs)-> (bs, num_query, embed_dims)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
