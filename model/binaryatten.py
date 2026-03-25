import torch
import torch.nn as nn
from typing import Any, NewType
from torch.autograd import Function
from einops import rearrange

# === 基础量化算子 (保持不变) ===
BinaryTensor = NewType('BinaryTensor', torch.Tensor)


def binary_sign(x: torch.Tensor) -> BinaryTensor:
    return x.sign() + (x == 0).type(torch.float)


class STESign(Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor) -> BinaryTensor:
        ctx.save_for_backward(x)
        return binary_sign(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
        x, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[x.gt(1)] = 0
        grad_input[x.lt(-1)] = 0
        return grad_input


binarize = STESign.apply


class SymQuantizer(Function):
    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise=False):
        ctx.save_for_backward(input, clip_val)
        max_input = torch.max(torch.abs(input)).expand_as(input) if layerwise else \
            (torch.max(torch.abs(input), dim=-2, keepdim=True)[0].expand_as(input).detach())
        s = (2 ** (num_bits - 1) - 1) / (max_input + 1e-6)
        return torch.round(input * s).div(s + 1e-6)

    @staticmethod
    def backward(ctx, grad_output):
        input, clip_val = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None


symquantize = SymQuantizer.apply


def round_ste(z):
    zhat = z.round()
    return z + (zhat - z).detach()


# ========================================================
# 🌟 核心：专门针对时空 (T, H, W) 设计的 3D Binary Attention
# ========================================================
class SpatiotemporalBinaryAttention(nn.Module):
    def __init__(self, dim, num_heads=8, window_size=(16, 12, 12), qkv_bias=False,
                 attn_drop=0., proj_drop=0., attn_quant=True, pv_quant=True, attn_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.dim = dim

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attn_quant = attn_quant
        self.pv_quant = pv_quant
        self.attn_bias = attn_bias
        self.window_size = window_size
        self.num_tokens = window_size[0] * window_size[1] * window_size[2]

        if self.attn_bias:
            T, H, W = self.window_size
            self.num_relative_distance = (2 * T - 1) * (2 * H - 1) * (2 * W - 1)
            # 定义可学习的偏置表: [距离种类数, 多头数]
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(self.num_relative_distance, num_heads)
            )

            # 1. 构建 (T, H, W) 三维网格坐标
            coords_t = torch.arange(T)
            coords_h = torch.arange(H)
            coords_w = torch.arange(W)
            coords = torch.stack(torch.meshgrid([coords_t, coords_h, coords_w], indexing='ij'))  # 3, T, H, W
            coords_flatten = torch.flatten(coords, 1)  # 3, T*H*W

            # 2. 计算两两点之间的相对距离向量
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 3, T*H*W, T*H*W
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # T*H*W, T*H*W, 3

            # 3. 相对距离偏移，保证其 >= 0
            relative_coords[:, :, 0] += T - 1
            relative_coords[:, :, 1] += H - 1
            relative_coords[:, :, 2] += W - 1

            # 4. 将 3D 坐标打平映射到 1D 的查表索引 (类似多进制转换)
            relative_coords[:, :, 0] *= (2 * H - 1) * (2 * W - 1)
            relative_coords[:, :, 1] *= (2 * W - 1)
            relative_position_index = relative_coords.sum(-1)  # T*H*W, T*H*W

            # 注册为 buffer，不会随梯度更新，随模型保存
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    @staticmethod
    def _quantize(x):
        s = x.abs().mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)
        return s * binarize(x)

    @staticmethod
    def _quantize_p(x):
        qmax = 255;
        s = 1.0 / qmax
        return s * round_ste(x / s).clamp(0, qmax)

    @staticmethod
    def _quantize_v(x, bits=8):
        return symquantize(x, torch.tensor([-2.0, 2.0]), bits, False)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.attn_quant:
            q = self._quantize(q)
            k = self._quantize(k)

            attn = (q @ k.transpose(-2, -1)) * self.scale

            # 🌟 核心：在 Softmax 前加上相对位置偏置
            if self.attn_bias:
                relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.num_tokens, self.num_tokens, -1)  # N, N, num_heads
                relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # num_heads, N, N
                attn = attn + relative_position_bias.unsqueeze(0)  # [B, num_heads, N, N]

            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            if self.pv_quant:
                attn = self._quantize_p(attn)
                v = self._quantize_v(v, 8)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            # 全精度也依然推荐加上位置偏置
            if self.attn_bias:
                bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                    self.num_tokens, self.num_tokens, -1).permute(2, 0, 1).contiguous()
                attn = attn + bias.unsqueeze(0)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x