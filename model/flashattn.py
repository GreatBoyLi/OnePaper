import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==========================================
# 1. 新核心：基于 PyTorch 2.0 SDPA (FlashAttention) 的自注意力
# ==========================================
class SDPAAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, attn_drop=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.attn_drop = attn_drop

        # 注意：SDPA 内部会自动进行 scale (1 / sqrt(dim_head))，所以不需要手动定义 scale

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        # x: (B, N, D)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # 转换为 SDPA 要求的形状: (Batch, Heads, Seq_Len, Head_Dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 🌟 核心创新点：调用 PyTorch 2.0 内置的 SDPA 接口
        # 在显卡支持时，它会自动底层调用 FlashAttention 或 Memory-Efficient Attention
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )

        # 还原形状
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class TransformerBlock(nn.Module):
    """
    标准的 Pre-Norm Transformer 编码器块 (使用 SDPA)
    """

    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)

        # 🌟 替换为 SDPAAttention
        self.attn = SDPAAttention(dim, heads=heads, dim_head=dim_head, attn_drop=0.)

        self.drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class SpatiotemporalTransformer(nn.Module):
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

        self.time_pos_embed = nn.Parameter(torch.randn(1, 16, 1, embed_dim))
        self.space_pos_embed = nn.Parameter(torch.randn(1, 1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(dim=embed_dim, heads=6, dim_head=64, dropout=dropout)
            for _ in range(depth)
        ])

    def forward(self, S_t):
        x = S_t.transpose(1, 2)
        x = self.patch_embed(x)
        B, C, T, H_p, W_p = x.shape

        x_v = rearrange(x, 'b c t h w -> b t (h w) c')
        x_v = x_v + self.time_pos_embed + self.space_pos_embed
        v_tokens = rearrange(x_v, 'b t p c -> b (t p) c')

        x = self.pos_drop(v_tokens)

        for block in self.layers:
            x = block(x)

        x = rearrange(x, 'b (t h w) c -> b t h w c', t=T, h=H_p, w=W_p)
        H_t = x[:, -1, :, :, :]
        H_t = H_t.permute(0, 3, 1, 2)

        return H_t


# ==========================================
# 2. 新增：基于 SDPA 的交叉注意力机制 (模态融合)
# ==========================================
class SDPACrossAttention(nn.Module):
    def __init__(self, dim, heads=6, dim_head=64, attn_drop=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.attn_drop = attn_drop

        # Q 来自时间序列，KV 来自卫星云图等视觉特征
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x_q, x_kv):
        # x_q: (B, N_q, D)
        # x_kv: (B, N_kv, D)
        q = self.to_q(x_q)
        k, v = self.to_kv(x_kv).chunk(2, dim=-1)

        # 转换为 SDPA 要求的形状
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        # 🌟 调用 SDPA 进行高效交叉注意力计算
        out = F.scaled_dot_product_attention(
            query=q,
            key=k,
            value=v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False
        )

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossTransformerBlock(nn.Module):
    """
    专门处理模态融合的交叉注意力块 (使用 SDPA)
    """

    def __init__(self, dim, heads, dim_head, dropout=0.1):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        # 🌟 替换为 SDPACrossAttention
        self.cross_attn = SDPACrossAttention(dim, heads=heads, dim_head=dim_head, attn_drop=0.)

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