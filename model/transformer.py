import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==========================================
# 1. 新核心：O(N) 复杂度的线性注意力机制
# ==========================================
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 🌟 核心创新点：用 ELU+1 替代 Softmax，将 $O(N^2)$ 降维打击到 $O(N)$
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # 改变矩阵乘法顺序
        kv = torch.einsum('b h n d, b h n m -> b h d m', k, v)
        z = 1 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum('b h n d, b h d m, b h n -> b h n m', q, kv, z)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    标准的 Pre-Norm Transformer 编码器块
    """

    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, heads=heads, dim_head=dim_head)

        # 🌟 新增：用于残差连接前的 Dropout
        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        # 带有残差连接的 Pre-Norm 结构
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class LinearSpatiotemporalTransformer(nn.Module):
    def __init__(self, in_channels=1, patch_size=8, embed_dim=128, img_size=96, depth=3, out_channels=16, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv3d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size)
        )

        self.time_pos_embed = nn.Parameter(torch.randn(1, 16, 1, embed_dim))  # 这个16是时间序列的长度
        self.space_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, embed_dim))

        # 🌟 新增：位置编码后的 Dropout
        self.pos_drop = nn.Dropout(p=dropout)

        # 🌟 修复点 1：使用刚写好的 TransformerBlock
        self.layers = nn.ModuleList([
            TransformerBlock(dim=embed_dim, heads=6, dim_head=64, dropout=dropout)
            for _ in range(depth)
        ])

        # self.to_hidden_map = nn.Sequential(
        #     nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=1),
        #     nn.ReLU(), )

    def forward(self, S_t):
        x = S_t.transpose(1, 2)
        x = self.patch_embed(x)
        B, C, T, H_p, W_p = x.shape

        x_v = rearrange(x, 'b c t h w -> b t (h w) c')
        x_v = x_v + self.time_pos_embed + self.space_pos_embed
        v_tokens = rearrange(x_v, 'b t p c -> b (t p) c')

        # 🌟 在进入 Transformer Block 之前应用 Dropout
        x = self.pos_drop(v_tokens)

        # 🌟 修复点 2：极其清爽的层级调用
        for block in self.layers:
            x = block(x)

        x = rearrange(x, 'b (t h w) c -> b t h w c', t=T, h=H_p, w=W_p)
        H_t = x[:, -1, :, :, :]
        H_t = H_t.permute(0, 3, 1, 2)

        # H_t = F.interpolate(H_t, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        # H_t = self.to_hidden_map(H_t)

        return H_t


# ==========================================
# 新增：O(N) 复杂度的线性交叉注意力机制
# ==========================================
class LinearCrossAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads

        # Q 来自时间序列，KV 来自卫星云图
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x_q, x_kv):
        # x_q: (B, N_q, D) 时间序列 Tokens
        # x_kv: (B, N_kv, D) 视觉空间 Tokens
        q = self.to_q(x_q)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # 线性化核函数
        q = F.elu(q) + 1.0
        k = F.elu(k) + 1.0

        # 矩阵乘法结合律改变
        kv = torch.einsum('b h n d, b h n m -> b h d m', k, v)
        z = 1 / (torch.einsum('b h n d, b h d -> b h n', q, k.sum(dim=2)) + 1e-6)
        out = torch.einsum('b h n d, b h d m, b h n -> b h n m', q, kv, z)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossTransformerBlock(nn.Module):
    """
    专门处理模态融合的交叉注意力块
    """

    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = LinearCrossAttention(dim, heads=heads, dim_head=dim_head)

        # 🌟 新增 Dropout
        self.drop = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x_q, x_kv):
        # 时序 (Q) 主动去查询 视觉 (KV)
        attn_out = self.cross_attn(self.norm_q(x_q), self.norm_kv(x_kv))
        x_q = x_q + self.drop(attn_out)
        x_q = x_q + self.drop(self.ff(self.norm_ff(x_q)))
        return x_q
