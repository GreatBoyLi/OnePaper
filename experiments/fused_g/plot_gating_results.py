import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# 0. 路径配置
# =========================
RESULT_DIR = "./gating_saved_results"
SAVE_DIR = "./gating_figures"
os.makedirs(SAVE_DIR, exist_ok=True)


PLOT_FILES = [
    {
        "name": "Clear",
        "csv_path": os.path.join(RESULT_DIR, "gating_results_clear.csv"),
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
    {
        "name": "Cloudy",
        "csv_path": os.path.join(RESULT_DIR, "gating_results_cloudy.csv"),
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
    {
        "name": "Ramp",
        "csv_path": os.path.join(RESULT_DIR, "gating_results_ramp.csv"),
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
]


# =========================
# 1. 字体设置
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
# 2. 构造连续时间轴
# =========================
def build_continuous_time_df(csv_path, start_hour="06:00", end_hour="20:00"):
    df = pd.read_csv(csv_path, parse_dates=["Time"])
    df = df.set_index("Time")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    unique_dates = np.unique(df.index.date)

    start_time = f"{unique_dates[0]} {start_hour}"
    end_time = f"{unique_dates[-1]} {end_hour}"

    target_index = pd.date_range(start=start_time, end=end_time, freq="15min")

    df = df.reindex(target_index)

    # 功率缺失时段填 0，门控权重不填 0
    df["True_Power_kW"] = df["True_Power_kW"].fillna(0.0)
    df["Pred_Power_kW"] = df["Pred_Power_kW"].fillna(0.0)
    df["Gamma"] = df["Gamma"].fillna(np.nan)

    return df


# =========================
# 3. 画单个天气图
# =========================
def plot_one_weather(df, weather_name):
    true_power = df["True_Power_kW"].values
    pred_power = df["Pred_Power_kW"].values
    gamma = df["Gamma"].values

    x_axis = np.arange(len(df))
    time_labels = df.index.strftime("%m-%d %H:%M").tolist()

    fig, ax1 = plt.subplots(figsize=(16, 6))

    color_true = "black"
    color_pred = "#d62728"
    color_gamma = "#1f77b4"

    ax1.set_xlabel(
        "Time",
        fontsize=12,
        fontweight="bold",
        labelpad=10,
        fontproperties=FONT_PROP,
    )

    ax1.set_ylabel(
        "PV Power (kW)",
        color=color_true,
        fontsize=12,
        fontweight="bold",
        fontproperties=FONT_PROP,
    )

    ax1.plot(
        x_axis,
        true_power,
        color=color_true,
        label="True Power",
        linewidth=2.5,
    )

    ax1.plot(
        x_axis,
        pred_power,
        color=color_pred,
        linestyle="--",
        label="Predicted Power",
        linewidth=2.0,
    )

    ax1.tick_params(axis="y", labelcolor=color_true)
    ax1.grid(True, linestyle=":", alpha=0.5, color="gray")

    ax2 = ax1.twinx()

    ax2.set_ylabel(
        r"Temporal Modality Weight $\gamma$",
        color=color_gamma,
        fontsize=12,
        fontweight="bold",
        fontproperties=FONT_PROP,
    )

    ax2.plot(
        x_axis,
        gamma,
        color=color_gamma,
        label=r"Gating Weight $\gamma$",
        linewidth=2.5,
    )

    ax2.fill_between(
        x_axis,
        0,
        gamma,
        color=color_gamma,
        alpha=0.15,
        where=~np.isnan(gamma),
    )

    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor=color_gamma)

    # 每 3 小时一个刻度
    tick_spacing = 12
    ax1.set_xticks(x_axis[::tick_spacing])
    ax1.set_xticklabels(
        time_labels[::tick_spacing],
        rotation=25,
        ha="right",
        fontproperties=FONT_PROP,
    )

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
# 4. 统计 Gamma
# =========================
def summarize_gamma(plot_files):
    rows = []

    for item in plot_files:
        weather_name = item["name"]
        csv_path = item["csv_path"]

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        gamma = df["Gamma"].dropna().values

        rows.append(
            {
                "Weather": weather_name,
                "Mean_Gamma": float(np.mean(gamma)),
                "Std_Gamma": float(np.std(gamma)),
                "Min_Gamma": float(np.min(gamma)),
                "Max_Gamma": float(np.max(gamma)),
                "Sample_Number": int(len(gamma)),
            }
        )

    summary_df = pd.DataFrame(rows)
    save_path = os.path.join(SAVE_DIR, "gamma_summary.csv")
    summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\nGamma 统计结果：")
    print(summary_df)
    print(f"已保存：{save_path}")


# =========================
# 5. 主函数
# =========================
def main():
    for item in PLOT_FILES:
        weather_name = item["name"]
        csv_path = item["csv_path"]
        start_hour = item.get("start_hour", "06:00")
        end_hour = item.get("end_hour", "20:00")

        if not os.path.exists(csv_path):
            print(f"跳过：文件不存在 -> {csv_path}")
            continue

        df = build_continuous_time_df(
            csv_path=csv_path,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        plot_one_weather(df, weather_name)

    summarize_gamma(PLOT_FILES)


if __name__ == "__main__":
    main()