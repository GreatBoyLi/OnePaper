import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange


class SpatioTemporalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        # 1. 空间 Mamba
        self.spatial_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # 2. 时间 Mamba
        self.temporal_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # 🌟 移除了 LayerNorm 和 Pooling，保持纯粹的特征提取，方便堆叠

    def forward(self, x):
        # x shape: (B, T, S, C)
        B, T, S, C = x.shape

        # 空间扫描
        x_spatial = x.reshape(B * T, S, C)
        x_spatial = self.spatial_mamba(x_spatial)
        x_spatial = x_spatial.reshape(B, T, S, C)

        # 时间扫描
        x_temporal = x_spatial.transpose(1, 2).contiguous()  # (B, S, T, C)
        x_temporal = x_temporal.reshape(B * S, T, C)
        x_temporal = self.temporal_mamba(x_temporal)

        # 还原回 (B, T, S, C)
        out = x_temporal.reshape(B, S, T, C).transpose(1, 2).contiguous()

        # 加上残差连接 (Residual Connection) 帮助深层网络收敛
        return out + x