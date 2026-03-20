import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange


class SpatioTemporalMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()

        # 🌟 恢复 LayerNorm (Pre-Norm 策略)，这对深层网络收敛至关重要
        self.norm_spatial = nn.LayerNorm(d_model)
        self.norm_temporal = nn.LayerNorm(d_model)

        # 1. 空间 Mamba
        self.spatial_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        # 2. 时间 Mamba
        self.temporal_mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, x):
        # x shape: (B, T, S, C)
        B, T, S, C = x.shape

        # --- 空间扫描 ---
        # 1. Norm
        x_norm = self.norm_spatial(x)
        # 2. Reshape for Mamba (B*T, S, C)
        x_spatial = x_norm.reshape(B * T, S, C)
        # 3. Mamba
        x_spatial = self.spatial_mamba(x_spatial)
        # 4. Reshape back
        x_spatial = x_spatial.reshape(B, T, S, C)

        # 5. 残差连接
        x = x + x_spatial

        # --- 时间扫描 ---
        # 注意：这里是对“已经更新过空间信息”的 x 进行操作
        # 1. Norm (对更新后的 x 做归一化)
        x_norm = self.norm_temporal(x)

        # 2. Transpose & Reshape for Temporal Mamba (B, S, T, C) -> (B*S, T, C)
        x_temporal = x_norm.transpose(1, 2).contiguous()
        x_temporal = x_temporal.reshape(B * S, T, C)

        # 3. Mamba
        x_temporal = self.temporal_mamba(x_temporal)

        # 4. Reshape & Transpose back to (B, T, S, C)
        x_temporal = x_temporal.reshape(B, S, T, C).transpose(1, 2).contiguous()

        # 5. 残差连接
        out = x + x_temporal

        return out
