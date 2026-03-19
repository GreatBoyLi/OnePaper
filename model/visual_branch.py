import torch.nn as nn

from model.transformer import LinearSpatiotemporalTransformer


# ==========================================
# 3. 你的 RICNN 保持绝对原汁原味 (一行不改！)
# ==========================================
class RICNN(nn.Module):
    def __init__(self, in_channels, roi_size=16, out_dim=128):
        super(RICNN, self).__init__()
        self.roi_size = roi_size
        self.roi_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(128, out_dim)

    def forward(self, feature_map):
        b, c, h, w = feature_map.size()
        center_y, center_x = h // 2, w // 2
        half_roi = self.roi_size // 2
        roi_feature = feature_map[:, :,
        center_y - half_roi: center_y + half_roi,
        center_x - half_roi: center_x + half_roi]
        x = self.roi_conv(roi_feature)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ==========================================
# 4. 升级版的 VisualBranch
# ==========================================
class VisualBranch(nn.Module):
    def __init__(self, input_channels=1, patch_size=8, img_size=9, transformer_dim=384, transformer_depth=3,
                 ricnn_in_channels=16, roi_size=16, final_dim=128):
        super(VisualBranch, self).__init__()

        # 1. 全新强大的线性 Transformer (替代了原本的 ConvLSTM)
        self.transformer = LinearSpatiotemporalTransformer(
            in_channels=input_channels,
            patch_size=patch_size,  # 把 96x96 切成 8x8 的小块
            embed_dim=transformer_dim,
            img_size=img_size,
            depth=transformer_depth,  # 堆叠 3 层 Transformer
            out_channels=ricnn_in_channels  # 完美对齐 RICNN 所需的输入通道数
        )

        # 2. 负责聚焦电站的 RICNN
        self.ricnn = RICNN(in_channels=ricnn_in_channels, roi_size=roi_size, out_dim=final_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len=16, Channels=1, H=96, W=96)

        # Transformer 一次性看穿 16 帧，直接输出最后时刻的 96x96 空间图！
        # 完全抛弃了慢吞吞的 for t in range(seq_len) 循环！
        h_state = self.transformer(x)  # -> (Batch, 16, 96, 96)

        # 送入 RICNN 聚焦中心
        final_visual_feature = self.ricnn(h_state)  # -> (Batch, final_dim)

        return final_visual_feature
