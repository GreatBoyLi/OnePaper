import torch.nn as nn
import torch
import torch.nn.functional as F


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


class TemporalAttentionPooling(nn.Module):
    """
    时序注意力池化层 (Temporal Attention Pooling)
    为时间序列中的每一个时间步动态分配权重，聚焦突变/关键时刻。
    """

    def __init__(self, dim):
        super(TemporalAttentionPooling, self).__init__()
        # 用一个小型的多层感知机来评估每个时间步的“重要性得分”
        self.attention_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.Tanh(),
            nn.Linear(dim // 2, 1)
        )

    def forward(self, x):
        # x 的输入形状: (Batch, Seq_len=16, Dim=128)

        # 1. 计算每个时间步的原始得分 -> 形状: (Batch, Seq_len, 1)
        scores = self.attention_net(x)

        # 2. 在时间维度 (dim=1) 上进行 Softmax 归一化，得到概率权重
        attn_weights = F.softmax(scores, dim=1)

        # 3. 广播相乘并求和：用权重对各个时刻的特征进行加权浓缩
        # (Batch, 16, 128) * (Batch, 16, 1) -> (Batch, 16, 128) -> sum -> (Batch, 128)
        pooled_feat = torch.sum(x * attn_weights, dim=1)

        # 将池化后的浓缩特征和注意力权重一起返回
        return pooled_feat, attn_weights
