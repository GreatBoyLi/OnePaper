import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.font_manager as fm

from model.mymodel import MultiModalPVNet
from dataset.dataset import SatellitePVDataset


# =========================
# 0. 基础配置
# =========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SELF_DEPTH = 2
CROSS_DEPTH = 2
FINAL_DIM = 64
TRANSFORMER_DIM = 64
HEADS = 4
DROPOUT = 0.3
OUTPUT_SEQ_LEN = 4

HISTORY_LEN = 16

MODEL_PATH = "../../checkpoints/0428_depth_2_mse-dcca/Epoch:23-RMSE:0.0537-MAE:0.0194-MAPE:14.42%-R:97.97%.pth"
VAL_SAT_DIR = "../../data/val/crop_himawari/15min"

SAVE_DIR = "./gating_weight_figures"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 1. 多天气 CSV 配置
# 你后面把路径换成自己的即可
# =========================
WEATHER_FILES = [
    {
        "name": "Clear",
        "csv_path": "../../data/val/hardcore_clear_weather_2days.csv",
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
    {
        "name": "Cloudy",
        "csv_path": "../../data/val/hardcore_mixed_weather_2days.csv",
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
    # {
    #     "name": "Overcast",
    #     "csv_path": "../../data/val/overcast_weather_2days.csv",
    #     "start_hour": "06:00",
    #     "end_hour": "20:00",
    # },
    {
        "name": "Ramp",
        "csv_path": "../../data/val/hardcore_ramp_weather_2days.csv",
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
]


# =========================
# 2. 字体设置：Times New Roman
# =========================
def setup_font():
    font_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"

    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_prop = fm.FontProperties(fname=font_path)
        font_name = font_prop.get_name()
        plt.rcParams["font.family"] = font_name
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["axes.unicode_minus"] = False
        return font_prop
    else:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["mathtext.fontset"] = "stix"
        plt.rcParams["axes.unicode_minus"] = False
        return None


FONT_PROP = setup_font()


# =========================
# 3. 加载模型
# =========================
def load_model():
    model = MultiModalPVNet(
        final_dim=FINAL_DIM,
        transformer_dim=TRANSFORMER_DIM,
        heads=HEADS,
        self_depth=SELF_DEPTH,
        cross_depth=CROSS_DEPTH,
        output_seq_len=OUTPUT_SEQ_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    return model


# =========================
# 4. 推理一个天气文件
# =========================
def infer_one_weather(model, csv_path):
    val_dataset = SatellitePVDataset(csv_path, VAL_SAT_DIR, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    true_power_history = []
    pred_power_history = []
    g_values_history = []

    print(f"正在推断：{csv_path}")

    for batch in val_loader:
        with torch.no_grad():
            imgs = batch["x_images"].to(DEVICE)
            nums = batch["x_numeric"].to(DEVICE)
            targets = batch["y_power"].to(DEVICE)
            zeniths = batch["y_zenith"].to(DEVICE)
            y_clearsky = batch["y_clear_sky_ghi"].to(DEVICE)

            # 模型输出
            # 注意：这里假设 model 返回：
            # preds_csi, v_feat, t_feat, t_attn_weights, g_weights
            preds_csi, v_feat, t_feat, t_attn_weights, g_weights = model(imgs, nums)

            # 预测 CSI 转预测功率
            preds_power = preds_csi * y_clearsky

            # 夜间物理约束
            night_mask = zeniths > 88
            preds_power[night_mask] = 0.0

            preds_power_np = preds_power.detach().cpu().numpy()
            y_true_np = targets.detach().cpu().numpy()

            # g_weights 可能是标量，也可能是向量，这里统一取均值
            g_val_np = g_weights.detach().cpu().numpy()
            g_mean = float(np.mean(g_val_np))

            # 只取未来第一个预测步长
            true_power_history.append(float(y_true_np[0, 0]))
            pred_power_history.append(float(preds_power_np[0, 0]))
            g_values_history.append(g_mean)

    return (
        np.array(true_power_history),
        np.array(pred_power_history),
        np.array(g_values_history),
    )


# =========================
# 5. 对齐真实时间轴
# =========================
def build_results_dataframe(csv_path, true_power, pred_power, g_values, start_hour="06:00", end_hour="20:00"):
    df_val = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    # 去掉时区，避免 reindex 不匹配
    if df_val.index.tz is not None:
        df_val.index = df_val.index.tz_localize(None)

    valid_df = df_val.dropna(subset=["Active_Power"])

    min_len = min(len(true_power), len(valid_df) - HISTORY_LEN)

    if min_len <= 0:
        raise ValueError(f"有效样本长度不足，请检查 CSV 文件：{csv_path}")

    real_time_axis = valid_df.index[HISTORY_LEN: HISTORY_LEN + min_len]

    results_df = pd.DataFrame(
        {
            "True_Power": true_power[:min_len],
            "Pred_Power": pred_power[:min_len],
            "G_Weight": g_values[:min_len],
        },
        index=real_time_axis,
    )

    unique_dates = np.unique(real_time_axis.date)
    start_time = f"{unique_dates[0]} {start_hour}"
    end_time = f"{unique_dates[-1]} {end_hour}"

    target_index = pd.date_range(start=start_time, end=end_time, freq="15min")

    results_df = results_df.reindex(target_index)

    # 功率夜间或缺失时段可以填 0
    results_df["True_Power"] = results_df["True_Power"].fillna(0.0)
    results_df["Pred_Power"] = results_df["Pred_Power"].fillna(0.0)

    # 门控权重不要填 0，否则会误导为“夜间更依赖视觉模态”
    results_df["G_Weight"] = results_df["G_Weight"].fillna(np.nan)

    return results_df


# =========================
# 6. 绘图
# =========================
def plot_gating_profile(results_df, weather_name):
    final_true = results_df["True_Power"].values
    final_pred = results_df["Pred_Power"].values
    final_g = results_df["G_Weight"].values

    time_labels = results_df.index.strftime("%m-%d %H:%M").tolist()
    x_axis = np.arange(len(final_true))

    plt.rcParams["figure.dpi"] = 300

    fig, ax1 = plt.subplots(figsize=(16, 6))

    color_true = "black"
    color_pred = "#d62728"
    color_g = "#1f77b4"

    ax1.set_xlabel(
        "Time",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
        fontproperties=FONT_PROP,
    )

    ax1.set_ylabel(
        "Normalized PV Output (p.u.)",
        color=color_true,
        fontsize=12,
        fontweight="bold",
        fontproperties=FONT_PROP,
    )

    ax1.plot(
        x_axis,
        final_true,
        color=color_true,
        label="True Power",
        linewidth=2.5,
    )

    ax1.plot(
        x_axis,
        final_pred,
        color=color_pred,
        linestyle="--",
        label="Predicted Power",
        linewidth=2,
    )

    ax1.tick_params(axis="y", labelcolor=color_true)
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")

    ax2 = ax1.twinx()

    ax2.set_ylabel(
        r"Temporal Modality Weight $\gamma$",
        color=color_g,
        fontsize=12,
        fontweight="bold",
        fontproperties=FONT_PROP,
    )

    ax2.plot(
        x_axis,
        final_g,
        color=color_g,
        label=r"Gating Weight $\gamma$",
        linewidth=2.5,
    )

    # fill_between 遇到 nan 会自动断开
    ax2.fill_between(
        x_axis,
        0,
        final_g,
        color=color_g,
        alpha=0.15,
        where=~np.isnan(final_g),
    )

    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor=color_g)

    # X 轴刻度，每 3 小时一个
    tick_spacing = 12
    ax1.set_xticks(x_axis[::tick_spacing])
    ax1.set_xticklabels(
        time_labels[::tick_spacing],
        rotation=25,
        ha="right",
        fontproperties=FONT_PROP,
    )

    # Y 轴字体
    for label in ax1.get_yticklabels():
        label.set_fontproperties(FONT_PROP)
    for label in ax2.get_yticklabels():
        label.set_fontproperties(FONT_PROP)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.12),
        ncol=3,
        frameon=False,
        prop=FONT_PROP,
    )

    # 不在图内放标题，论文中用图注说明
    plt.tight_layout()

    base_name = f"gating_weight_{weather_name.lower()}"
    png_path = os.path.join(SAVE_DIR, base_name + ".png")
    pdf_path = os.path.join(SAVE_DIR, base_name + ".pdf")
    svg_path = os.path.join(SAVE_DIR, base_name + ".svg")

    plt.savefig(png_path, dpi=600, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.savefig(svg_path, bbox_inches="tight")

    print(f"已保存：{png_path}")
    print(f"已保存：{pdf_path}")
    print(f"已保存：{svg_path}")

    plt.close(fig)


# =========================
# 7. 统计各天气下门控权重均值
# =========================
def summarize_gating(results_all):
    summary = []

    for weather_name, results_df in results_all.items():
        g = results_df["G_Weight"].dropna().values

        if len(g) == 0:
            summary.append(
                {
                    "Weather": weather_name,
                    "Mean_Gamma": np.nan,
                    "Std_Gamma": np.nan,
                    "Min_Gamma": np.nan,
                    "Max_Gamma": np.nan,
                    "Sample_Number": 0,
                }
            )
            continue

        summary.append(
            {
                "Weather": weather_name,
                "Mean_Gamma": float(np.mean(g)),
                "Std_Gamma": float(np.std(g)),
                "Min_Gamma": float(np.min(g)),
                "Max_Gamma": float(np.max(g)),
                "Sample_Number": int(len(g)),
            }
        )

    summary_df = pd.DataFrame(summary)
    save_path = os.path.join(SAVE_DIR, "gating_weight_summary.csv")
    summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n门控权重统计结果：")
    print(summary_df)
    print(f"统计结果已保存：{save_path}")

    return summary_df


# =========================
# 8. 主函数
# =========================
def main():
    model = load_model()

    results_all = {}

    for item in WEATHER_FILES:
        weather_name = item["name"]
        csv_path = item["csv_path"]
        start_hour = item.get("start_hour", "06:00")
        end_hour = item.get("end_hour", "20:00")

        if not os.path.exists(csv_path):
            print(f"跳过：文件不存在 -> {csv_path}")
            continue

        true_power, pred_power, g_values = infer_one_weather(model, csv_path)

        results_df = build_results_dataframe(
            csv_path=csv_path,
            true_power=true_power,
            pred_power=pred_power,
            g_values=g_values,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        results_all[weather_name] = results_df

        plot_gating_profile(results_df, weather_name)

    if results_all:
        summarize_gating(results_all)
    else:
        print("没有成功处理任何天气文件，请检查 WEATHER_FILES 中的路径。")


if __name__ == "__main__":
    main()