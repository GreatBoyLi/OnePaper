import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# 0. 路径配置
# =========================
RESULT_DIR = "./gating_saved_results_two_days"
SAVE_DIR = "./gating_figures_two_days"
os.makedirs(SAVE_DIR, exist_ok=True)

# 读取上一步生成的两天数据
PLOT_FILES = [
    {
        "season": "Spring",
        "csv_path": os.path.join(RESULT_DIR, "gating_results_spring_two_days.csv"),
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
    {
        "season": "Summer",
        "csv_path": os.path.join(RESULT_DIR, "gating_results_summer_two_days.csv"),
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
    {
        "season": "Autumn",
        "csv_path": os.path.join(RESULT_DIR, "gating_results_autumn_two_days.csv"),
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
    {
        "season": "Winter",
        "csv_path": os.path.join(RESULT_DIR, "gating_results_winter_two_days.csv"),
        "start_hour": "06:00",
        "end_hour": "20:00",
    },
]

# 两天之间的横轴空隙，单位是 15 min 点数
GAP_POINTS = 8


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
# 2. 读取两天数据
# =========================
def load_two_day_data(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    if df["Time"].dt.tz is not None:
        df["Time"] = df["Time"].dt.tz_localize(None)

    if "Date" not in df.columns:
        df["Date"] = df["Time"].dt.date.astype(str)

    required_cols = [
        "Time",
        "Date",
        "True_Power_kW",
        "Pred_Power_kW",
        "Gamma",
        "Selected_Weather_Type",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{csv_path} 缺少必要列: {col}")

    return df


# =========================
# 3. 提取某一天 06:00-20:00 数据
# =========================
def build_one_day_df(df, date_str, start_hour="06:00", end_hour="20:00"):
    day_df = df[df["Date"].astype(str) == str(date_str)].copy()

    if len(day_df) == 0:
        raise ValueError(f"找不到日期 {date_str} 的数据。")

    day_df = day_df.set_index("Time").sort_index()

    start_time = pd.to_datetime(f"{date_str} {start_hour}")
    end_time = pd.to_datetime(f"{date_str} {end_hour}")

    day_df = day_df.loc[start_time:end_time].copy()

    target_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="15min"
    )

    day_df = day_df.reindex(target_index)

    # 功率缺失填 0；Gamma 保持 NaN，避免把缺失误解为权重为 0
    day_df["True_Power_kW"] = day_df["True_Power_kW"].fillna(0.0)
    day_df["Pred_Power_kW"] = day_df["Pred_Power_kW"].fillna(0.0)
    day_df["Gamma"] = day_df["Gamma"].fillna(np.nan)

    return day_df


# =========================
# 4. 构造两天的紧凑横轴
# =========================
def build_two_day_segments(df, start_hour="06:00", end_hour="20:00", gap_points=8):
    """
    把 Rainy day 和 Clear-sky day 分别截取 06:00-20:00，
    然后在横轴上拼接，中间留空隙。
    """
    weather_order = {
        "Rainy": 0,
        "Clear-sky": 1,
        "Clear": 1,
        "Cloudy": 2,
    }

    day_info = (
        df[["Date", "Selected_Weather_Type"]]
        .drop_duplicates()
        .copy()
    )

    day_info["Order"] = day_info["Selected_Weather_Type"].map(
        lambda x: weather_order.get(str(x), 9)
    )

    day_info = day_info.sort_values(["Order", "Date"]).reset_index(drop=True)

    # 通常只会有 Rainy + Clear-sky 两天
    day_info = day_info.iloc[:2].copy()

    segments = []
    separator_positions = []
    x_offset = 0

    for i, row in day_info.iterrows():
        date_str = str(row["Date"])
        weather_type = str(row["Selected_Weather_Type"])

        one_day_df = build_one_day_df(
            df,
            date_str=date_str,
            start_hour=start_hour,
            end_hour=end_hour,
        )

        n = len(one_day_df)
        x = np.arange(n) + x_offset

        segments.append(
            {
                "date": date_str,
                "weather_type": weather_type,
                "df": one_day_df,
                "x": x,
                "x_start": x[0],
                "x_end": x[-1],
            }
        )

        if i < len(day_info) - 1:
            separator_positions.append(x[-1] + gap_points / 2)

        x_offset = x[-1] + gap_points + 1

    return segments, separator_positions


# =========================
# 5. 绘制一个季节的两天 Gamma 图
# =========================
def plot_one_season_two_days(csv_path, season_name, start_hour="06:00", end_hour="20:00"):
    df = load_two_day_data(csv_path)

    segments, separator_positions = build_two_day_segments(
        df,
        start_hour=start_hour,
        end_hour=end_hour,
        gap_points=GAP_POINTS,
    )

    # IEEE 单栏偏宽图
    fig, ax1 = plt.subplots(figsize=(7.4, 3.8))

    color_true = "black"
    color_pred = "#d62728"
    color_gamma = "#1f77b4"

    # =========================
    # 左轴：真实功率与预测功率
    # =========================
    for idx, seg in enumerate(segments):
        x = seg["x"]
        one_day_df = seg["df"]

        true_power = one_day_df["True_Power_kW"].values
        pred_power = one_day_df["Pred_Power_kW"].values

        ax1.plot(
            x,
            true_power,
            color=color_true,
            label="Actual Power" if idx == 0 else None,
            linewidth=1.45,
            zorder=3,
        )

        ax1.plot(
            x,
            pred_power,
            color=color_pred,
            linestyle="--",
            label="Predicted Power" if idx == 0 else None,
            linewidth=1.35,
            zorder=3,
        )

    ax1.set_ylabel(
        "PV Power (kW)",
        color=color_true,
        fontsize=10,
        fontweight="bold",
        fontproperties=FONT_PROP,
    )

    ax1.tick_params(axis="y", labelcolor=color_true, labelsize=9)
    ax1.tick_params(axis="x", labelsize=8, pad=2)

    ax1.grid(
        True,
        linestyle=":",
        linewidth=0.55,
        alpha=0.45,
        color="gray"
    )

    # =========================
    # 右轴：Gamma
    # =========================
    ax2 = ax1.twinx()

    for idx, seg in enumerate(segments):
        x = seg["x"]
        one_day_df = seg["df"]

        gamma = one_day_df["Gamma"].values

        ax2.plot(
            x,
            gamma,
            color=color_gamma,
            label=r"Gating Weight $\gamma$" if idx == 0 else None,
            linewidth=1.55,
            zorder=2,
        )

        ax2.fill_between(
            x,
            0,
            gamma,
            color=color_gamma,
            alpha=0.10,
            where=~np.isnan(gamma),
            zorder=1,
        )

    ax2.set_ylabel(
        r"Temporal Weight $\gamma$",
        color=color_gamma,
        fontsize=10,
        fontweight="bold",
        fontproperties=FONT_PROP,
    )

    ax2.set_ylim(0.0, 1.0)
    ax2.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax2.tick_params(axis="y", labelcolor=color_gamma, labelsize=9)

    # =========================
    # 两天之间的分隔线
    # =========================
    for pos in separator_positions:
        ax1.axvline(
            pos,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
            zorder=0,
        )

    # =========================
    # 横轴刻度
    # =========================
    all_tick_positions = []
    all_tick_labels = []

    for seg in segments:
        x = seg["x"]
        one_day_df = seg["df"]

        # 15 min 一个点，4 小时 = 16 个点
        tick_spacing = 16
        tick_positions = x[::tick_spacing]
        tick_labels = one_day_df.index[::tick_spacing].strftime("%H:%M").tolist()

        all_tick_positions.extend(tick_positions)
        all_tick_labels.extend(tick_labels)

        # 在每一天下方标注天气类型和日期
        mid_x = (seg["x_start"] + seg["x_end"]) / 2.0

        ax1.text(
            mid_x,
            -0.23,
            f"{seg['weather_type']} ({seg['date']})",
            transform=ax1.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            fontproperties=FONT_PROP,
        )

    ax1.set_xticks(all_tick_positions)
    ax1.set_xticklabels(
        all_tick_labels,
        rotation=0,
        ha="center",
        fontsize=8,
        fontproperties=FONT_PROP,
    )

    ax1.set_xlabel(
        "Time",
        fontsize=10,
        fontweight="bold",
        labelpad=26,
        fontproperties=FONT_PROP,
    )

    # 去掉左右空白
    x_min = min(seg["x_start"] for seg in segments)
    x_max = max(seg["x_end"] for seg in segments)
    ax1.set_xlim(x_min, x_max)
    ax1.margins(x=0)

    # =========================
    # 图内季节标注
    # =========================
    ax1.text(
        0.01,
        0.95,
        season_name,
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        fontproperties=FONT_PROP,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.75,
            pad=1.5
        ),
    )

    # =========================
    # 图例
    # =========================
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()

    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.20),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        prop=FONT_PROP,
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.4,
    )

    # 字体统一
    for label in ax1.get_yticklabels() + ax2.get_yticklabels():
        label.set_fontproperties(FONT_PROP)

    for spine in ax1.spines.values():
        spine.set_linewidth(0.8)

    for spine in ax2.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout(pad=0.4)
    plt.subplots_adjust(bottom=0.24, top=0.84)

    # =========================
    # 保存图像
    # =========================
    base_name = f"gating_weight_{season_name.lower()}_two_days"

    png_path = os.path.join(SAVE_DIR, base_name + ".png")
    pdf_path = os.path.join(SAVE_DIR, base_name + ".pdf")
    svg_path = os.path.join(SAVE_DIR, base_name + ".svg")

    plt.savefig(
        png_path,
        dpi=900,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.02
    )

    plt.savefig(
        svg_path,
        bbox_inches="tight",
        pad_inches=0.02
    )

    print(f"已保存：{png_path}")
    print(f"已保存：{pdf_path}")
    print(f"已保存：{svg_path}")

    plt.close(fig)


# =========================
# 6. 统计 Gamma
# =========================
def summarize_gamma(plot_files):
    rows = []

    for item in plot_files:
        season_name = item["season"]
        csv_path = item["csv_path"]

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, parse_dates=["Time"])

        if "Date" not in df.columns:
            df["Date"] = df["Time"].dt.date.astype(str)

        if "Selected_Weather_Type" not in df.columns:
            df["Selected_Weather_Type"] = "Unknown"

        for (weather_type, date_str), sub in df.groupby(["Selected_Weather_Type", "Date"]):
            gamma = sub["Gamma"].dropna().values

            rows.append(
                {
                    "Season": season_name,
                    "Weather_Type": weather_type,
                    "Date": date_str,
                    "Mean_Gamma": float(np.mean(gamma)) if len(gamma) > 0 else np.nan,
                    "Std_Gamma": float(np.std(gamma)) if len(gamma) > 0 else np.nan,
                    "Min_Gamma": float(np.min(gamma)) if len(gamma) > 0 else np.nan,
                    "Max_Gamma": float(np.max(gamma)) if len(gamma) > 0 else np.nan,
                    "Sample_Number": int(len(gamma)),
                }
            )

    summary_df = pd.DataFrame(rows)
    save_path = os.path.join(SAVE_DIR, "gamma_summary_two_days.csv")
    summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\nGamma 统计结果：")
    print(summary_df)
    print(f"已保存：{save_path}")


# =========================
# 7. 主函数
# =========================
def main():
    for item in PLOT_FILES:
        season_name = item["season"]
        csv_path = item["csv_path"]
        start_hour = item.get("start_hour", "06:00")
        end_hour = item.get("end_hour", "20:00")

        if not os.path.exists(csv_path):
            print(f"跳过：文件不存在 -> {csv_path}")
            continue

        plot_one_season_two_days(
            csv_path=csv_path,
            season_name=season_name,
            start_hour=start_hour,
            end_hour=end_hour,
        )

    summarize_gamma(PLOT_FILES)


if __name__ == "__main__":
    main()