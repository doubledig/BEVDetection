import warnings
from functools import partial
from typing import Dict, Optional, Tuple, Union

import mmengine
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch
from mmcv.utils import IS_CUDA_AVAILABLE, IS_MLU_AVAILABLE, IS_NPU_AVAILABLE
from mmengine import deprecated_api_warning
from mmengine.model import constant_init, kaiming_init, BaseModule, xavier_init
from mmengine.registry import MODELS
from mmengine.utils.dl_utils.parrots_wrapper import _BatchNorm

from project_plugin import GeometricKernelAttentionFunc


def efficient_conv_bn_eval_forward(bn: _BatchNorm,
                                   conv: nn.modules.conv._ConvNd,
                                   x: torch.Tensor):
    # These lines of code are designed to deal with various cases
    # like bn without affine transform, and conv without bias
    weight_on_the_fly = conv.weight
    if conv.bias is not None:
        bias_on_the_fly = conv.bias
    else:
        bias_on_the_fly = torch.zeros_like(bn.running_var)

    if bn.weight is not None:
        bn_weight = bn.weight
    else:
        bn_weight = torch.ones_like(bn.running_var)

    if bn.bias is not None:
        bn_bias = bn.bias
    else:
        bn_bias = torch.zeros_like(bn.running_var)

    # shape of [C_out, 1, 1, 1] in Conv2d
    weight_coeff = torch.rsqrt(bn.running_var +
                               bn.eps).reshape([-1] + [1] *
                                               (len(conv.weight.shape) - 1))
    # shape of [C_out, 1, 1, 1] in Conv2d
    coefff_on_the_fly = bn_weight.view_as(weight_coeff) * weight_coeff

    # shape of [C_out, C_in, k, k] in Conv2d
    weight_on_the_fly = weight_on_the_fly * coefff_on_the_fly
    # shape of [C_out] in Conv2d
    bias_on_the_fly = bn_bias + coefff_on_the_fly.flatten() * \
                      (bias_on_the_fly - bn.running_mean)

    return conv._conv_forward(x, weight_on_the_fly, bias_on_the_fly)


class ConvModule(nn.Module):
    _abbr_ = 'conv_block'

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1,
                 groups: int = 1,
                 bias: Union[bool, str] = 'auto',
                 conv_cfg: Optional[Dict] = None,
                 norm_cfg: Optional[Dict] = None,
                 act_cfg: Optional[Dict] = dict(type='ReLU'),
                 inplace: bool = True,
                 with_spectral_norm: bool = False,
                 padding_mode: str = 'zeros',
                 order: tuple = ('conv', 'norm', 'act'),
                 efficient_conv_bn_eval: bool = False):
        super().__init__()
        assert conv_cfg is None or isinstance(conv_cfg, dict)
        assert norm_cfg is None or isinstance(norm_cfg, dict)
        assert act_cfg is None or isinstance(act_cfg, dict)
        official_padding_mode = ['zeros', 'circular']
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.inplace = inplace
        self.with_spectral_norm = with_spectral_norm
        self.with_explicit_padding = padding_mode not in official_padding_mode
        self.order = order
        assert isinstance(self.order, tuple) and len(self.order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        self.with_norm = norm_cfg is not None
        self.with_activation = act_cfg is not None
        # if the conv layer is before a norm layer, bias is unnecessary.
        if bias == 'auto':
            bias = not self.with_norm
        self.with_bias = bias

        # reset padding to 0 for conv module
        conv_padding = 0 if self.with_explicit_padding else padding
        # build convolution layer
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=conv_padding,
            dilation=dilation,
            groups=groups,
            bias=bias)
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        if self.with_spectral_norm:
            self.conv = nn.utils.spectral_norm(self.conv)

        # build normalization layers
        # if self.with_norm:
        #     # norm layer is after conv layer
        #     if order.index('norm') > order.index('conv'):
        #         norm_channels = out_channels
        #     else:
        #         norm_channels = in_channels
        #     self.norm_name, norm = build_norm_layer(
        #         norm_cfg, norm_channels)  # type: ignore
        #     self.add_module(self.norm_name, norm)
        #     if self.with_bias:
        #         if isinstance(norm, (_BatchNorm, _InstanceNorm)):
        #             warnings.warn(
        #                 'Unnecessary conv bias before batch/instance norm')
        # else:
        #     self.norm_name = None  # type: ignore

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        # build activation layer
        if self.with_activation:
            act_cfg_ = act_cfg.copy()  # type: ignore
            # nn.Tanh has no 'inplace' argument
            if act_cfg_['type'] not in [
                'Tanh', 'PReLU', 'Sigmoid', 'HSigmoid', 'Swish', 'GELU'
            ]:
                act_cfg_.setdefault('inplace', inplace)
            self.activate = MODELS.build(act_cfg_)

        # Use msra init by default
        self.init_weights()

    @property
    def norm(self):
        if self.norm_name:
            return getattr(self, self.norm_name)
        else:
            return None

    def init_weights(self):
        # 1. It is mainly for customized conv layers with their own
        #    initialization manners by calling their own ``init_weights()``,
        #    and we do not want ConvModule to override the initialization.
        # 2. For customized conv layers without their own initialization
        #    manners (that is, they don't have their own ``init_weights()``)
        #    and PyTorch's conv layers, they will be initialized by
        #    this method with default ``kaiming_init``.
        # Note: For PyTorch's conv layers, they will be overwritten by our
        #    initialization implementation using default ``kaiming_init``.
        if not hasattr(self.conv, 'init_weights'):
            if self.with_activation and self.act_cfg['type'] == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
                a = self.act_cfg.get('negative_slope', 0.01)
            else:
                nonlinearity = 'relu'
                a = 0
            kaiming_init(self.conv, a=a, nonlinearity=nonlinearity)
        if self.with_norm:
            constant_init(self.norm, 1, bias=0)

    def forward(self,
                x: torch.Tensor,
                activate: bool = True,
                norm: bool = True) -> torch.Tensor:
        layer_index = 0
        while layer_index < len(self.order):
            layer = self.order[layer_index]
            if layer == 'conv':
                if self.with_explicit_padding:
                    x = self.padding_layer(x)
                # if the next operation is norm and we have a norm layer in
                # eval mode and we have enabled `efficient_conv_bn_eval` for
                # the conv operator, then activate the optimized forward and
                # skip the next norm operator since it has been fused
                if layer_index + 1 < len(self.order) and \
                        self.order[layer_index + 1] == 'norm' and norm and \
                        self.with_norm and not self.norm.training and \
                        self.efficient_conv_bn_eval_forward is not None:
                    self.conv.forward = partial(
                        self.efficient_conv_bn_eval_forward, self.norm,
                        self.conv)
                    layer_index += 1
                    x = self.conv(x)
                    del self.conv.forward
                else:
                    x = self.conv(x)
            elif layer == 'norm' and norm and self.with_norm:
                x = self.norm(x)
            elif layer == 'act' and activate and self.with_activation:
                x = self.activate(x)
            layer_index += 1
        return x

    def turn_on_efficient_conv_bn_eval(self, efficient_conv_bn_eval=True):
        # efficient_conv_bn_eval works for conv + bn
        # with `track_running_stats` option
        if efficient_conv_bn_eval and self.norm \
                and isinstance(self.norm, _BatchNorm) \
                and self.norm.track_running_stats:
            self.efficient_conv_bn_eval_forward = efficient_conv_bn_eval_forward  # noqa: E501
        else:
            self.efficient_conv_bn_eval_forward = None  # type: ignore

    @staticmethod
    def create_from_conv_bn(conv: torch.nn.modules.conv._ConvNd,
                            bn: torch.nn.modules.batchnorm._BatchNorm,
                            efficient_conv_bn_eval=True) -> 'ConvModule':
        """Create a ConvModule from a conv and a bn module."""
        self = ConvModule.__new__(ConvModule)
        super(ConvModule, self).__init__()

        self.conv_cfg = None
        self.norm_cfg = None
        self.act_cfg = None
        self.inplace = False
        self.with_spectral_norm = False
        self.with_explicit_padding = False
        self.order = ('conv', 'norm', 'act')

        self.with_norm = True
        self.with_activation = False
        self.with_bias = conv.bias is not None

        # build convolution layer
        self.conv = conv
        # export the attributes of self.conv to a higher level for convenience
        self.in_channels = self.conv.in_channels
        self.out_channels = self.conv.out_channels
        self.kernel_size = self.conv.kernel_size
        self.stride = self.conv.stride
        self.padding = self.conv.padding
        self.dilation = self.conv.dilation
        self.transposed = self.conv.transposed
        self.output_padding = self.conv.output_padding
        self.groups = self.conv.groups

        # build normalization layers
        self.norm_name, norm = 'bn', bn
        self.add_module(self.norm_name, norm)

        self.turn_on_efficient_conv_bn_eval(efficient_conv_bn_eval)

        return self


class GridMask(nn.Module):
    def __init__(self,
                 use_h=True,
                 use_w=True,
                 rotate=1,
                 offset=False,
                 ratio=0.5,
                 mode=1,
                 p=0.7):
        super().__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.prob = p

    def forward(self, x):
        if self.training and np.random.rand() > self.prob:
            n, c, h, w = x.size()
            x = x.view(-1, h, w)
            hh = int(1.5 * h)
            ww = int(1.5 * w)
            d = np.random.randint(2, h)
            l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
            mask = np.ones((hh, ww), np.float32)
            st_h = np.random.randint(d)
            st_w = np.random.randint(d)
            if self.use_h:
                for i in range(hh // d):
                    s = d * i + st_h
                    t = min(s + l, hh)
                    mask[s:t, :] *= 0
            if self.use_w:
                for i in range(ww // d):
                    s = d * i + st_w
                    t = min(s + l, ww)
                    mask[:, s:t] *= 0
            r = np.random.randint(self.rotate)
            mask = Image.fromarray(np.uint8(mask))
            mask = mask.rotate(r)
            mask = np.asarray(mask)
            mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]

            mask = torch.from_numpy(mask).to(x.device, dtype=x.dtype)
            if self.mode == 1:
                mask = 1 - mask
            mask = mask.expand_as(x)
            if self.offset:
                offset = torch.from_numpy(2 * (np.random.rand(h, w) - 0.5)).to(x.device, dtype=x.dtype)
                x = x * mask + offset * (1 - mask)
            else:
                x = x * mask
            return x.view(n, c, h, w)
        return x


class LearnedPositionalEncoding(BaseModule):
    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50,
                 init_cfg=dict(type='Uniform', layer='Embedding')):
        super().__init__(init_cfg)
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(
                1, w, 1)),
            dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


class SinePositionalEncoding3D(BaseModule):
    def __init__(self,
                 num_feats,
                 temperature=10000,
                 normalize=False,
                 scale=2 * torch.pi,
                 eps=1e-6,
                 offset=0,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        not_mask = ~mask
        b, n, h, w = mask.shape
        device = mask.device
        n_embed = not_mask.cumsum(1, dtype=torch.float32)
        y_embed = not_mask.cumsum(2, dtype=torch.float32)
        x_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            n_embed = (n_embed + self.offset) / (n_embed[:, -1:, :, :] + self.eps) * self.scale
            y_embed = (y_embed + self.offset) / (y_embed[:, :, -1:, :] + self.eps) * self.scale
            x_embed = (x_embed + self.offset) / (x_embed[:, :, :, -1:] + self.eps) * self.scale
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        n_embed = n_embed[:, :, :, :, None] / dim_t
        x_embed = x_embed[:, :, :, :, None] / dim_t
        y_embed = y_embed[:, :, :, :, None] / dim_t
        n_embed = torch.stack(
            (n_embed[:, :, :, :, 0::2].sin(), n_embed[:, :, :, :, 1::2].cos()),
            dim=4).view(b, n, h, w, -1)
        x_embed = torch.stack(
            (x_embed[:, :, :, :, 0::2].sin(), x_embed[:, :, :, :, 1::2].cos()),
            dim=4).view(b, n, h, w, -1)
        y_embed = torch.stack(
            (y_embed[:, :, :, :, 0::2].sin(), y_embed[:, :, :, :, 1::2].cos()),
            dim=4).view(b, n, h, w, -1)
        pos = torch.cat((n_embed, y_embed, x_embed), dim=4).permute(0, 1, 4, 2, 3)
        return pos


class TemporalAttention(BaseModule):
    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 num_bev_queue: int = 2,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[mmengine.ConfigDict] = None,
                 value_proj_ratio: float = 1.0):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

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
            embed_dims * num_bev_queue, num_bev_queue * num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims * num_bev_queue,
                                           num_bev_queue * num_heads * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * torch.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels * self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.)
        xavier_init(self.value_proj, distribution='uniform')
        xavier_init(self.output_proj, distribution='uniform')
        self._is_init = True

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if value is None:
            value = torch.cat([query, query], dim=0 if self.batch_first else 1)

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs2, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs2, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5) \
            .reshape(bs2, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6) \
            .reshape(bs2, num_query, self.num_heads, self.num_levels, self.num_points, 2)

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
        if ((IS_CUDA_AVAILABLE and value.is_cuda)
                or (IS_MLU_AVAILABLE and value.is_mlu)
                or (IS_NPU_AVAILABLE and value.device.type == 'npu')):
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = output.permute(1, 2, 0)
        output = output.view(num_query, -1, bs, self.num_bev_queue).mean(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


class GeometrySpatialCrossAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 dropout=0.1,
                 init_cfg=None,
                 attention=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.attention = build_attention(attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weight()

    def init_weight(self):
        xavier_init(self.output_proj, distribution='uniform')

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                bev_mask: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if key is None:
            key = query
        if value is None:
            value = key

        identity = query
        slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos

        bs, num_query, embed_dims = query.size()  # b, 20000, 256

        d = reference_points.shape[3]  # 4
        indexes = []
        max_len = 0
        for mask_per_img in bev_mask:
            indexe = []
            for mask in mask_per_img:
                index = mask.sum(-1).nonzero().squeeze(-1)
                indexe.append(index)
                max_len = max(max_len, index.shape[0])
            indexes.append(indexe)

        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points.new_zeros(
            [bs, self.num_cams, max_len, d, 2])
        for i in range(bs):
            for j in range(self.num_cams):
                index = indexes[i][j]
                queries_rebatch[i, j, :index.shape[0]] = query[i, index]
                reference_points_rebatch[i, j, :index.shape[0]] = reference_points[i, j, index]

        bs, num_cam, l, embed_dims = key.shape

        key = key.reshape(bs * self.num_cams, l, embed_dims)
        value = value.reshape(bs * self.num_cams, l, embed_dims)

        queries = self.attention(query=queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
                                 key=key,
                                 value=value,
                                 reference_points=reference_points_rebatch.view(bs * self.num_cams, max_len, d, 2),
                                 spatial_shapes=spatial_shapes,
                                 level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for i in range(bs):
            for j in range(self.num_cams):
                index = indexes[i][j]
                slots[i, index] = slots[i, index] + queries[i, j, :index.shape[0]]

        count = bev_mask.sum(-1) > 0
        count = count.sum(1)
        count = count.clamp(min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + identity


class GeometryKernelAttention(BaseModule):
    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 kernel_size=(3, 3),
                 dilation=1,
                 im2col_step=64,
                 batch_first=False,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None

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
        self.kernel_size = kernel_size
        self.num_points = kernel_size[0] * kernel_size[1]
        self.attention_weights = nn.Linear(
            embed_dims, num_levels * self.num_points * self.num_heads)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        grid_h, grid_w = kernel_size
        y = (torch.arange(grid_h) - grid_h // 2) * dilation
        x = (torch.arange(grid_w) - grid_w // 2) * dilation
        offsets = torch.stack(
            torch.meshgrid(x, y)).permute(1, 2, 0).reshape(grid_h * grid_w, 2)
        self.register_buffer("grid_offsets", offsets, persistent=False)
        self.init_weights()

    def init_weights(self):
        constant_init(self.attention_weights, val=0.)
        xavier_init(self.value_proj, distribution='uniform')
        xavier_init(self.output_proj, distribution='uniform')
        self._is_init = True

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        if value is None:
            value = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)

        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            with torch.no_grad():
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                offsets = self.grid_offsets[None, None, None, None]
                reference_points = reference_points[:, :, :, None, :] * offset_normalizer
                sampling_locations = (reference_points[:, :, :, :, None, :] + offsets).round().long()
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2, but get {reference_points.shape[-1]} instead.')

        output = GeometricKernelAttentionFunc.apply(
            value, spatial_shapes, level_start_index, sampling_locations.contiguous(), attention_weights,
            self.im2col_step
        )
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        return output
