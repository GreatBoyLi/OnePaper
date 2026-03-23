import torch
import torch.nn as nn
from einops import rearrange

from model.transformer import TransformerBlock, CrossTransformerBlock
from model.visual_branch import RICNN


class MultiModalPVNet(nn.Module):
    def __init__(self, visual_input_cha=3, times_input_cha=10, patch_size=8, img_size=96, transformer_dim=384,
                 self_depth=3, cross_depth=3,
                 ricnn_in_channels=384, roi_size=16, final_dim=256, output_seq_len=4, heads=6, dim_head=64,
                 dropout=0.1):
        super(MultiModalPVNet, self).__init__()
        self.img_size = img_size

        # ================= 1. Token 提取器 =================
        # 视觉 Patch 嵌入
        self.v_patch_embed = nn.Conv3d(visual_input_cha, transformer_dim, kernel_size=(1, patch_size, patch_size),
                                       stride=(1, patch_size, patch_size))
        num_patches = (img_size // patch_size) ** 2
        self.v_pos_embed = nn.Parameter(torch.randn(1, 16 * num_patches, transformer_dim))

        # 时序 Linear 嵌入
        self.t_embed = nn.Linear(times_input_cha, transformer_dim)
        self.t_pos_embed = nn.Parameter(torch.randn(1, 16, transformer_dim))

        # ================= 2. Stage 1: 多层独立自注意力 (深度特征提取) =================
        # 视觉支路：连续过 3 层自注意力
        self.visual_sa_layers = nn.ModuleList([
            TransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(self_depth)
        ])

        # 时序支路：连续过 3 层自注意力
        self.ts_sa_layers = nn.ModuleList([
            TransformerBlock(dim=transformer_dim, heads=heads, dim_head=dim_head, dropout=dropout)
            for _ in range(self_depth)
        ])

        # 🛠️ DCCA 约束用的辅助提取头 (截留融合前的深层独立特征)
        # self.v_to_hidden_map = nn.Conv2d(transformer_dim, ricnn_in_channels, kernel_size=1)
        # self.ricnn = RICNN(in_channels=ricnn_in_channels, roi_size=roi_size, out_dim=final_dim)
        # self.t_intermediate_head = nn.Sequential(nn.LayerNorm(transformer_dim), nn.Linear(transformer_dim, final_dim))

        # ================= 3. Stage 2: 多层交叉融合 (深度跨模态查询) =================
        # 让时间序列 (Q) 连续 3 次去跨模态查询云图 (K, V)，不断修正自己的特征
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
        x_v = x_images.transpose(1, 2)
        x_v = self.v_patch_embed(x_v)
        B, C, T, H_p, W_p = x_v.shape
        v_tokens = rearrange(x_v, 'b c t h w -> b (t h w) c')
        v_tokens = v_tokens + self.v_pos_embed[:, :v_tokens.shape[1], :]

        t_tokens = self.t_embed(x_numeric)
        t_tokens = t_tokens + self.t_pos_embed

        # --- 2. Stage 1: 3 层深层自注意力 ---
        # 视觉不断提炼云层的时空变化
        for sa_block in self.visual_sa_layers:
            v_tokens = sa_block(v_tokens)

        # 时序不断提炼发电量的历史趋势
        for sa_block in self.ts_sa_layers:
            t_tokens = sa_block(t_tokens)

        # # 🌟 截流点：在最高层（第 3 层）提取完美独立特征，送给 DCCA 计算正交约束！
        # v_img = rearrange(v_tokens, 'b (t h w) c -> b t h w c', t=T, h=H_p, w=W_p)
        # v_img = v_img[:, -1, :, :, :].permute(0, 3, 1, 2)
        # v_img = F.interpolate(v_img, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        # v_img = self.v_to_hidden_map(v_img)
        # v_feat = self.ricnn(v_img)  # -> (Batch, final_dim)
        #
        # t_feat = self.t_intermediate_head(t_tokens[:, -1, :])  # -> (Batch, final_dim)

        # --- 3. Stage 2: 3 层早期交叉融合 (Deep Cross Attention) ---
        # 此时的 t_tokens 变成了 fust_tokens，它将在 3 层网络中反复去视觉 v_tokens 里“淘宝”
        fused_tokens = t_tokens
        for cross_block in self.cross_attn_layers:
            fused_tokens = cross_block(x_q=fused_tokens, x_kv=v_tokens)

        # --- 4. 最终单一预测 ---
        # 此时的 fused_tokens 已经是一个彻底吸收了 3 层云图动态的究极体序列
        final_out = fused_tokens[:, -1, :]  # 取最后时刻的融合表征
        preds = self.predictor(final_out)

        # 返回 preds 给回归 Loss，返回独立的 v_feat 和 t_feat 给 DCCA Loss
        return preds  # , v_feat, t_feat


# 测试块
if __name__ == "__main__":
    print("🚀 开始测试 [3层 SA + 3层 CA] 深层交叉融合网络...")
    batch_size = 2
    seq_len = 16

    dummy_imgs = torch.randn(batch_size, seq_len, 3, 96, 96)
    dummy_nums = torch.randn(batch_size, seq_len, 10)

    # 显式指定深度为 3
    model = MultiModalPVNet(self_depth=3, cross_depth=3, output_seq_len=4)
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
