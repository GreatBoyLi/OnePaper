import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

from model.mymodel import MultiModalPVNet
from dataset.dataset import SatellitePVDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SELF_DEPTH = 3
CROSS_DEPTH = 3
FINAL_DIM = 64
TRANSFORMER_DIM = 128
HEADS = 4
DROPOUT = 0.3

MODEL_PATH = "../checkpoints/Epoch:5-RMSE:0.0584-MAE:0.0283-MAPE:20.83%-R:97.73%.pth"
VAL_CSV_PATH = "../data/val/20240606.csv"
VAL_SAT_DIR = "../data/val/crop_himawari/15min"


def main():
    model = MultiModalPVNet(
        final_dim=FINAL_DIM, transformer_dim=TRANSFORMER_DIM, heads=HEADS,
        self_depth=SELF_DEPTH, cross_depth=CROSS_DEPTH, output_seq_len=4, dropout=DROPOUT
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

    val_dataset = SatellitePVDataset(VAL_CSV_PATH, VAL_SAT_DIR, mode="val")

    val_loader = DataLoader(val_dataset, batch_size=1, sampler=None, shuffle=False, num_workers=4)

    # 确保模型处于评估模式
    model.eval()
    g_values_history = []
    true_power_history = []
    pred_power_history = []

    for batch in val_loader:
        with torch.no_grad():  # 预测时务必关闭梯度计算
            imgs = batch['x_images'].to(DEVICE)
            nums = batch['x_numeric'].to(DEVICE)
            targets = batch['y_power'].to(DEVICE)
            zeniths = batch['y_zenith'].to(DEVICE)
            y_clearsky = batch['y_clear_sky_ghi'].to(DEVICE)

            preds_csi, v_feat, t_feat, t_attn_weights, g_weights = model(imgs, nums)

            # 计算最终的预测功率
            preds_power = preds_csi * y_clearsky

            # 根据天顶角进行物理约束（夜晚强制置零）
            night_mask = zeniths > 88
            preds_power[night_mask] = 0.0

            # 1. 统一转为 numpy 数组
            preds_power = preds_power.cpu().numpy()
            y_true = targets.cpu().numpy()
            g_val = g_weights.cpu().numpy()

            # 🚀 2. 关键修复区：统一维度，只取未来第一步！
            # 假设 DataLoader 的 batch_size=1
            # y_true 形状是 (1, 4)，preds_power 形状是 (1, 4)
            # 我们只取索引为 0 的那个值 (即未来 15 分钟的那一步)
            true_power_history.append(y_true[0, 0])
            pred_power_history.append(preds_power[0, 0])

            # g_val 通常是标量或形状为 (1,)，用 item() 取出唯一的数值
            g_values_history.append(g_val.item())

        # --- 循环结束 ---

        # 转换为 numpy 数组
    true_power = np.array(true_power_history)  # 现在长度肯定是 77
    pred_power = np.array(pred_power_history)  # 现在长度肯定是 77
    g_values = np.array(g_values_history)  # 现在长度肯定是 77

    # 动态生成时间轴，periods 自动为 77
    time_axis = pd.date_range("04:00", periods=len(true_power), freq="15min")

    # 设置全局字体和清晰度 (论文专用)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # ================= 绘制左轴 (光伏实际与预测功率) =================
    color_true = 'black'
    color_pred = '#d62728'  # 高级红

    ax1.set_xlabel('Time of Day', fontsize=12, fontweight='bold')
    ax1.set_ylabel('PV Power Generation (kW)', color=color_true, fontsize=12, fontweight='bold')
    ax1.plot(time_axis, true_power, color=color_true, label='True Power', linewidth=2)
    ax1.plot(time_axis, pred_power, color=color_pred, linestyle='--', label='Predicted Power', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_true)
    ax1.grid(True, linestyle=':', alpha=0.6)

    # ================= 绘制右轴 (动态门控权重 g) =================
    ax2 = ax1.twinx()  # 实例化共享 x 轴的第二个 y 轴
    color_g = '#1f77b4'  # 高级蓝

    ax2.set_ylabel('Temporal Routing Weight (g)', color=color_g, fontsize=12, fontweight='bold')
    # 使用带面积填充的线，视觉冲击力更强
    ax2.plot(time_axis, g_values, color=color_g, label='Gating Weight (g)', linewidth=2.5)
    ax2.fill_between(time_axis, 0, g_values, color=color_g, alpha=0.15)
    ax2.set_ylim(-0.1, 1.1)  # g 值范围是 0 到 1
    ax2.tick_params(axis='y', labelcolor=color_g)

    # ================= 论文加分项：标注云团突变区域 =================
    # 假设你在看数据时，发现中午 12:00 到 14:00 发生了云层遮挡，功率剧降
    # 用高亮背景色标出这段“多云突变区域”
    ax1.axvspan(time_axis[48], time_axis[56], color='gray', alpha=0.2, label='Cloudy / Ramp Event')

    # 合并图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4,
               frameon=False)

    # 格式化时间轴显示
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.title('Dynamic Modality Routing during a Clear-to-Cloudy Day', pad=30, fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Gated_Fusion_Analysis.png', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
