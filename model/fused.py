import torch.nn as nn
import torch


# ================= 新增：门控融合模块 =================
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 生成一个 0~1 的门控向量，决定听信哪个模态
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, t_feat, v_feat):
        # 拼接后计算门控权重
        g = self.gate(torch.cat([t_feat, v_feat], dim=-1))
        # 动态加权：g 控制时序，(1-g) 控制视觉
        fused = g * t_feat + (1 - g) * v_feat
        return self.out_proj(fused)
