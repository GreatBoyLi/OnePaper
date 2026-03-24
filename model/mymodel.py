import torch
import torch.nn as nn
from einops import rearrange

# from model.transformer import TransformerBlock, CrossTransformerBlock
from model.flashattn import TransformerBlock, CrossTransformerBlock
from model.fused import GatedFusion


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
        self.v_pos_embed = nn.Parameter(torch.randn(1, 16 * num_patches, transformer_dim))

        self.t_embed = nn.Linear(times_input_cha, transformer_dim)
        self.t_pos_embed = nn.Parameter(torch.randn(1, 16, transformer_dim))

        # ================= 2. Stage 1: 多层独立自注意力 =================
        self.visual_sa_layers = nn.ModuleList([
            TransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(self_depth)
        ])

        self.ts_sa_layers = nn.ModuleList([
            TransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(self_depth)
        ])

        # ================= 3. Stage 2: 多层交替交叉融合 (Co-Attention) =================
        # TS 作为 Q, 查询 Vis 的层
        self.ts_to_vis_layers = nn.ModuleList([
            CrossTransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(cross_depth)
        ])

        # Vis 作为 Q, 查询 TS 的层
        self.vis_to_ts_layers = nn.ModuleList([
            CrossTransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(cross_depth)
        ])

        # ================= 4. 最终融合与预测头 =================
        # 引入动态门控融合层
        self.fusion_layer = GatedFusion(dim=transformer_dim)

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
        x_v = x_images.transpose(1, 2)
        x_v = self.v_patch_embed(x_v)
        B, C, T, H_p, W_p = x_v.shape
        v_tokens = rearrange(x_v, 'b c t h w -> b (t h w) c')
        v_tokens = v_tokens + self.v_pos_embed[:, :v_tokens.shape[1], :]

        t_tokens = self.t_embed(x_numeric)
        t_tokens = t_tokens + self.t_pos_embed

        # --- 2. Stage 1: 深层独立自注意力 ---
        for sa_block in self.visual_sa_layers:
            v_tokens = sa_block(v_tokens)

        for sa_block in self.ts_sa_layers:
            t_tokens = sa_block(t_tokens)

        # --- 3. Stage 2: 早期交替交叉融合 (Alternating Co-Attention) ---
        fused_t = t_tokens
        fused_v = v_tokens

        # 交替互相查询、共同进化
        for i in range(len(self.ts_to_vis_layers)):
            fused_t = self.ts_to_vis_layers[i](x_q=fused_t, x_kv=fused_v)
            fused_v = self.vis_to_ts_layers[i](x_q=fused_v, x_kv=fused_t)

        # --- 4. 最终单一预测 ---
        # 提取时序的最后时刻特征
        final_t = fused_t[:, -1, :]  # (Batch, transformer_dim)
        # 提取视觉的全局平均特征
        final_v = fused_v.mean(dim=1)  # (Batch, transformer_dim)

        # 动态门控融合两股特征
        final_out = self.fusion_layer(final_t, final_v)

        preds = self.predictor(final_out)

        return preds


# 测试块
if __name__ == "__main__":
    print("🚀 开始测试 [交替 Co-Attention + 门控融合] 网络...")
    batch_size = 2
    seq_len = 16

    dummy_imgs = torch.randn(batch_size, seq_len, 3, 96, 96)
    dummy_nums = torch.randn(batch_size, seq_len, 10)

    model = MultiModalPVNet(self_depth=3, cross_depth=3, output_seq_len=4)
    model.eval()

    with torch.no_grad():
        output = model(dummy_imgs, dummy_nums)

    print(f"\n📥 输入云图 : {dummy_imgs.shape}")
    print(f"📥 输入数值 : {dummy_nums.shape}")
    print(f"📤 最终预测 : {output.shape} (预期为 Batch={batch_size}, 预测步数=4)")
    if output.shape == (batch_size, 4):
        print("\n✅ 门控网络测试成功！")
