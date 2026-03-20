import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.transformer import TransformerBlock, CrossTransformerBlock
from model.mamba import SpatioTemporalMambaBlock


class MultiModalPVNet(nn.Module):
    def __init__(self, visual_input_cha=3, times_input_cha=10, patch_size=8, img_size=96, transformer_dim=384,
                 self_depth=3, cross_depth=3,
                 ricnn_in_channels=384, roi_size=16, final_dim=256, output_seq_len=4, heads=6, dim_head=64,
                 dropout=0.1):
        super(MultiModalPVNet, self).__init__()
        self.img_size = img_size

        # ================= 1. Token 提取器 =================
        self.v_patch_embed = nn.Conv3d(visual_input_cha, transformer_dim, kernel_size=(1, patch_size, patch_size),
                                       stride=(1, patch_size, patch_size))
        num_patches = (img_size // patch_size) ** 2

        # 🌟 这里的视觉位置编码不需要改，但我们在 forward 中会改变它的使用方式
        self.v_pos_embed = nn.Parameter(torch.randn(1, 16 * num_patches, transformer_dim))

        self.t_embed = nn.Linear(times_input_cha, transformer_dim)
        self.t_pos_embed = nn.Parameter(torch.randn(1, 16, transformer_dim))

        # ================= 2. Stage 1: 多层独立特征提取 =================
        # 🌟 视觉支路：从 Transformer 升级为连续堆叠的 Mamba Block！
        self.visual_mamba_layers = nn.ModuleList([
            SpatioTemporalMambaBlock(d_model=transformer_dim)
            for _ in range(self_depth)
        ])

        # 时序支路：依然保持 Transformer
        self.ts_sa_layers = nn.ModuleList([
            TransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(self_depth)
        ])

        # ================= 3. Stage 2: 多层交叉融合 =================
        self.cross_attn_layers = nn.ModuleList([
            CrossTransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(cross_depth)
        ])

        # ================= 4. 最终单一预测头 =================
        self.predictor = nn.Sequential(
            nn.LayerNorm(transformer_dim),
            nn.Linear(transformer_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_seq_len),
        )

    def forward(self, x_images, x_numeric):
        # --- 1. Tokenization ---
        # 视觉分支
        x_v = x_images.transpose(1, 2)
        x_v = self.v_patch_embed(x_v)
        B, C, T, H_p, W_p = x_v.shape

        # 🌟 Mamba 需要的形状是 (B, T, S, C)，其中 S 是空间块数量
        v_tokens = rearrange(x_v, 'b c t h w -> b t (h w) c')

        # 处理位置编码的形状以适配 (B, T, S, C)
        pos_embed = rearrange(self.v_pos_embed[:, :T * H_p * W_p, :], '1 (t s) c -> 1 t s c', t=T)
        v_tokens = v_tokens + pos_embed

        # 时序分支
        t_tokens = self.t_embed(x_numeric)
        t_tokens = t_tokens + self.t_pos_embed

        # --- 2. Stage 1: 独立特征提取 ---
        # 🌟 视觉分支：经过 3 层 Mamba 捕捉时空物理演变
        for mamba_block in self.visual_mamba_layers:
            v_tokens = mamba_block(v_tokens)  # 进出都是 (B, T, S, C)

        # 🌟 关键聚合：对时间维度进行池化（将过去 16 帧的动态浓缩到每一个空间区块中）
        # 形状变化: (B, T, S, C) -> (B, S, C)
        v_tokens = v_tokens.mean(dim=1)

        # 时序分支：提炼历史发电趋势
        for sa_block in self.ts_sa_layers:
            t_tokens = sa_block(t_tokens)  # (B, T_seq, C)

        # --- 3. Stage 2: 交叉融合 ---
        fused_tokens = t_tokens
        for cross_block in self.cross_attn_layers:
            # 时序 (Q: B, 16, C) 去查询 视觉浓缩图 (KV: B, 144, C)
            fused_tokens = cross_block(x_q=fused_tokens, x_kv=v_tokens)

        # --- 4. 最终单一预测 ---
        final_out = fused_tokens[:, -1, :]  # 取最后时刻的融合表征
        preds = self.predictor(final_out)

        return preds


# 测试块
if __name__ == "__main__":
    print("🚀 开始测试 [3层 SA + 3层 CA] 深层交叉融合网络...")
    batch_size = 2
    seq_len = 16

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dummy_imgs = torch.randn(batch_size, seq_len, 3, 96, 96).to(DEVICE)
    dummy_nums = torch.randn(batch_size, seq_len, 10).to(DEVICE)

    # 显式指定深度为 3
    model = MultiModalPVNet(self_depth=3, cross_depth=3, output_seq_len=4).to(DEVICE)
    model.eval()

    with torch.no_grad():
        # output, v_f, t_f = model(dummy_imgs, dummy_nums)
        output = model(dummy_imgs, dummy_nums)

    print(f"\n📥 输入云图 : {dummy_imgs.shape}")
    print(f"📥 输入数值 : {dummy_nums.shape}")
    print(f"📤 最终预测 : {output.shape} (预期为 Batch={batch_size}, 预测步数=4)")
    # print(f"🧬 DCCA 独立视觉特征: {v_f.shape}")
    # print(f"🧬 DCCA 独立时序特征: {t_f.shape}")

    if output.shape == (batch_size, 4):
        print("\n✅ 测试成功！模型已具备深层 3x3 Attention 结构！")
