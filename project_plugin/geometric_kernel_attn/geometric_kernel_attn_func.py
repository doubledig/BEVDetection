from typing import Any

import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from . import GeometricKernelAttention


class GeometricKernelAttentionFunc(Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, value_spatial_shapes: torch.Tensor,
                value_level_start_index: torch.Tensor,
                sampling_locations: torch.Tensor,
                attention_weights: torch.Tensor,
                im2col_step: torch.Tensor) -> torch.Tensor:
        ctx.im2col_step = im2col_step
        output = GeometricKernelAttention.geometric_kernel_attn_cuda_forward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations, attention_weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output) -> Any:
        value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights = ctx.saved_tensors
        grad_value, grad_attn_weight = GeometricKernelAttention.geometric_kernel_attn_cuda_backward(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, grad_output,
            ctx.im2col_step
        )
        return grad_value, None, None, None, grad_attn_weight, None