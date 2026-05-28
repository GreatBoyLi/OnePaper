import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# 0. 路径配置
# =========================
RESULT_DIR = "./season_prediction_results"
SAVE_DIR = "./season_clear_day_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

SEASON_RESULT_FILES = {
    "Spring": os.path.join(RESULT_DIR, "prediction_results_spring.csv"),
    "Summer": os.path.join(RESULT_DIR, "prediction_results_summer.csv"),
    "Autumn": os.path.join(RESULT_DIR, "prediction_results_autumn.csv"),
    "Winter": os.path.join(RESULT_DIR, "prediction_results_winter.csv"),
}

CLEAR_DAY_FILE = os.path.join(RESULT_DIR, "selected_clear_days.csv")


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
# 2. 读取自动筛选的晴天代表日
# =========================
def load_selected_clear_days(clear_day_file):
    if not os.path.exists(clear_day_file):
        raise FileNotFoundError(f"找不到晴天代表日文件: {clear_day_file}")

    df = pd.read_csv(clear_day_file)

    season_dates = {}
    for _, row in df.iterrows():
        season_dates[row["Season"]] = str(row["Date"])

    return season_dates


# =========================
# 3. 读取某个季节的预测结果
# =========================
def load_season_result(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到预测结果文件: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    if df["Time"].dt.tz is not None:
        df["Time"] = df["Time"].dt.tz_localize(None)

    df["Date"] = df["Time"].dt.date.astype(str)

    return df


# =========================
# 4. 提取某一天数据
# =========================
def get_one_day_data(df, date_str, start_hour="06:00", end_hour="20:00"):
    day_df = df[df["Date"] == date_str].copy()

    if len(day_df) == 0:
        raise ValueError(f"找不到日期 {date_str} 的数据。")

    day_df = day_df.set_index("Time")

    start_time = pd.to_datetime(f"{date_str} {start_hour}")
    end_time = pd.to_datetime(f"{date_str} {end_hour}")

    day_df = day_df.loc[start_time:end_time].copy()

    # 构造连续 15 min 时间轴
    target_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="15min"
    )

    day_df = day_df.reindex(target_index)

    return day_df


# =========================
# 5. 绘制春夏秋冬 2×2 图
# =========================
def plot_season_clear_days(season_dates):
    season_order = ["Spring", "Summer", "Autumn", "Winter"]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 5.2),
        sharex=False,
        sharey=True
    )

    axes = axes.flatten()

    color_true = "black"
    color_pred = "#d62728"

    for i, season in enumerate(season_order):
        ax = axes[i]

        csv_path = SEASON_RESULT_FILES[season]
        date_str = season_dates[season]

        df = load_season_result(csv_path)

        day_df = get_one_day_data(
            df,
            date_str=date_str,
            start_hour="06:00",
            end_hour="20:00"
        )

        x_axis = np.arange(len(day_df))

        true_power = day_df["True_Power_kW"].values
        pred_power = day_df["Pred_Power_kW"].values

        ax.plot(
            x_axis,
            true_power,
            color=color_true,
            linewidth=1.6,
            label="True Power"
        )

        ax.plot(
            x_axis,
            pred_power,
            color=color_pred,
            linestyle="--",
            linewidth=1.5,
            label="Predicted Power"
        )

        # 横轴每 3 小时一个刻度
        tick_spacing = 12
        tick_positions = x_axis[::tick_spacing]
        tick_labels = day_df.index[tick_positions].strftime("%H:%M")

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(
            tick_labels,
            fontsize=8,
            fontproperties=FONT_PROP
        )

        ax.set_xlim(0, len(day_df) - 1)
        ax.margins(x=0)

        ax.grid(
            True,
            linestyle=":",
            linewidth=0.6,
            alpha=0.45,
            color="gray"
        )

        ax.set_title(
            f"({chr(97 + i)}) {season}",
            fontsize=10,
            fontweight="bold",
            fontproperties=FONT_PROP,
            pad=4
        )

        # 日期标注
        ax.text(
            0.98,
            0.92,
            date_str,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            fontproperties=FONT_PROP,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.75,
                pad=1.5
            )
        )

        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(FONT_PROP)

        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # 统一横纵轴标题
    fig.text(
        0.5,
        0.04,
        "Time",
        ha="center",
        fontsize=10,
        fontweight="bold",
        fontproperties=FONT_PROP
    )

    fig.text(
        0.04,
        0.5,
        "PV Power (kW)",
        va="center",
        rotation="vertical",
        fontsize=10,
        fontweight="bold",
        fontproperties=FONT_PROP
    )

    # 统一图例
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
        ncol=2,
        frameon=False,
        prop=FONT_PROP,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.2
    )

    plt.tight_layout(rect=[0.06, 0.06, 1.00, 0.94])

    png_path = os.path.join(SAVE_DIR, "season_clear_day_power_prediction.png")
    pdf_path = os.path.join(SAVE_DIR, "season_clear_day_power_prediction.pdf")
    svg_path = os.path.join(SAVE_DIR, "season_clear_day_power_prediction.svg")

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

    print(f"PNG saved to: {png_path}")
    print(f"PDF saved to: {pdf_path}")
    print(f"SVG saved to: {svg_path}")

    plt.show()


# =========================
# 6. 主函数
# =========================
def main():
    season_dates = load_selected_clear_days(CLEAR_DAY_FILE)

    print("Selected clear-sky dates:")
    for season, date_str in season_dates.items():
        print(f"{season}: {date_str}")

    required_seasons = ["Spring", "Summer", "Autumn", "Winter"]
    for season in required_seasons:
        if season not in season_dates:
            raise ValueError(f"selected_clear_days.csv 中缺少 {season} 的代表日。")

    plot_season_clear_days(season_dates)


if __name__ == "__main__":
    main()