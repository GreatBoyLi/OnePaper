import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from model.mymodel import MultiModalPVNet
from dataset.dataset import SatellitePVDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SELF_DEPTH = 2
CROSS_DEPTH = 2
FINAL_DIM = 64
TRANSFORMER_DIM = 64
HEADS = 4
DROPOUT = 0.3

# 你的模型与数据路径
MODEL_PATH = "../checkpoints/Epoch:7-RMSE:0.0537-MAE:0.0223-MAPE:14.19%-R:98.00%.pth"
VAL_CSV_PATH = "../data/val/hardcore_ramp_weather_2days.csv"
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

    print("🏃 正在推断验证集数据...")
    for batch in val_loader:
        with torch.no_grad():
            imgs = batch['x_images'].to(DEVICE)
            nums = batch['x_numeric'].to(DEVICE)
            targets = batch['y_power'].to(DEVICE)
            zeniths = batch['y_zenith'].to(DEVICE)
            y_clearsky = batch['y_clear_sky_ghi'].to(DEVICE)

            preds_csi, v_feat, t_feat, t_attn_weights, g_weights = model(imgs, nums)

            # 计算最终的预测功率
            preds_power = preds_csi * y_clearsky

            # 物理约束
            night_mask = zeniths > 88
            preds_power[night_mask] = 0.0

            preds_power = preds_power.cpu().numpy()
            y_true = targets.cpu().numpy()
            g_val = g_weights.cpu().numpy()

            # 统一维度，只取未来第一步
            true_power_history.append(y_true[0, 0])
            pred_power_history.append(preds_power[0, 0])
            g_values_history.append(g_val.item())

    # --- 推理结束 ---
    print("📈 正在处理数据并绘制分析图...")

    true_power = np.array(true_power_history)
    pred_power = np.array(pred_power_history)
    g_values = np.array(g_values_history)

    # 1. 绝对时序对齐
    df_val = pd.read_csv(VAL_CSV_PATH, parse_dates=True, index_col=0)
    valid_df = df_val.dropna(subset=['Active_Power'])

    HISTORY_LEN = 16
    min_len = min(len(true_power), len(valid_df) - HISTORY_LEN)

    true_power = true_power[:min_len]
    pred_power = pred_power[:min_len]
    g_values = g_values[:min_len]
    real_time_axis = valid_df.index[HISTORY_LEN: HISTORY_LEN + min_len]

    # 2. 纯物理裁切
    day_mask = (true_power > 0.005) | (pred_power > 0.005)

    true_power = true_power[day_mask]
    pred_power = pred_power[day_mask]
    g_values = g_values[day_mask]
    time_axis = real_time_axis[day_mask]

    # 3. 跨天断层剪裁
    time_diffs = np.diff(time_axis).astype('timedelta64[m]')
    cross_day_idx = np.where(time_diffs > np.timedelta64(90, 'm'))[0] + 1

    true_power = np.insert(true_power, cross_day_idx, np.nan)
    pred_power = np.insert(pred_power, cross_day_idx, np.nan)
    g_values = np.insert(g_values, cross_day_idx, np.nan)

    time_labels = time_axis.strftime('%m-%d %H:%M').tolist()
    for idx in reversed(cross_day_idx):
        time_labels.insert(idx, "")

    x_axis = np.arange(len(true_power))

    # ================= 绘图部分 (美学升级) =================
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['figure.dpi'] = 300

    # 稍微加长画布，让波峰显得不那么拥挤
    fig, ax1 = plt.subplots(figsize=(15, 6))

    color_true = 'black'
    color_pred = '#d62728'

    ax1.set_xlabel('Time of Day (Effective Daylight Hours)', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('PV Power Generation (kW)', color=color_true, fontsize=12, fontweight='bold')

    ax1.plot(x_axis, true_power, color=color_true, label='True Power', linewidth=2.5)
    ax1.plot(x_axis, pred_power, color=color_pred, linestyle='--', label='Predicted Power', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color_true)

    # 增加网格线的质感
    ax1.grid(True, linestyle=':', alpha=0.5, color='gray')

    ax2 = ax1.twinx()
    color_g = '#1f77b4'

    ax2.set_ylabel('Temporal Routing Weight (g)', color=color_g, fontsize=12, fontweight='bold')
    ax2.plot(x_axis, g_values, color=color_g, label='Gating Weight (g)', linewidth=2.5)
    ax2.fill_between(x_axis, 0, g_values, color=color_g, alpha=0.15)
    ax2.set_ylim(-0.1, 1.1)
    ax2.tick_params(axis='y', labelcolor=color_g)

    # 🌟 关键美学优化：动态计算 X 轴的刻度密度
    # 确保图表上有 12 到 15 个时间标签，不至于太挤也不至于太疏
    num_points = len(x_axis)
    tick_spacing = max(1, num_points // 12)

    ax1.set_xticks(x_axis[::tick_spacing])
    ax1.set_xticklabels(time_labels[::tick_spacing], rotation=25, ha='right')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    # 将图例移到顶部，更符合排版习惯
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=4,
               frameon=False, prop={'size': 11})

    plt.title('Dynamic Modality Routing under Clear Sky Conditions', pad=35, fontsize=15, fontweight='bold')

    plt.tight_layout()
    plt.savefig('Gated_Fusion_ClearSky_Analysis.png', bbox_inches='tight')
    print("✅ 成功生成晴天完美版图表: Gated_Fusion_ClearSky_Analysis.png")
    plt.show()


if __name__ == "__main__":
    main()