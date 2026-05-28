import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# =========================
# 0. 路径配置
# =========================
SAVE_DIR = "./all_weather_season_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# 三类气象结果目录
WEATHER_CONFIGS = {
    "Clear-sky": {
        "result_dir": "./season_prediction_results",
        "selected_day_file": "selected_clear_days.csv",
    },
    "Cloudy": {
        "result_dir": "./season_prediction_results",
        "selected_day_file": "selected_cloudy_days.csv",
    },
    "Rainy": {
        "result_dir": "./season_prediction_results",
        "selected_day_file": "selected_rainy_days.csv",
    },
}

SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]
WEATHER_ORDER = ["Clear-sky", "Cloudy", "Rainy"]


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
# 2. 读取代表日
# =========================
def load_selected_days(result_dir, selected_day_file):
    path = os.path.join(result_dir, selected_day_file)

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到代表日文件: {path}")

    df = pd.read_csv(path)

    season_dates = {}
    for _, row in df.iterrows():
        season_dates[row["Season"]] = str(row["Date"])

    return season_dates


# =========================
# 3. 读取预测结果
# =========================
def load_prediction_result(result_dir, season):
    csv_path = os.path.join(
        result_dir,
        f"prediction_results_{season.lower()}.csv"
    )

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

    target_index = pd.date_range(
        start=start_time,
        end=end_time,
        freq="15min"
    )

    day_df = day_df.reindex(target_index)

    return day_df


# =========================
# 5. 绘制 3×4 总图
# =========================
def plot_all_weather_season_power():
    # 3 行 × 4 列
    fig, axes = plt.subplots(
        3,
        4,
        figsize=(10.5, 6.8),
        sharex=True,
        sharey=True
    )

    color_true = "black"
    color_pred = "#d62728"

    subplot_idx = 0

    for row_idx, weather in enumerate(WEATHER_ORDER):
        result_dir = WEATHER_CONFIGS[weather]["result_dir"]
        selected_day_file = WEATHER_CONFIGS[weather]["selected_day_file"]

        selected_days = load_selected_days(result_dir, selected_day_file)

        for col_idx, season in enumerate(SEASON_ORDER):
            ax = axes[row_idx, col_idx]

            if season not in selected_days:
                raise ValueError(f"{weather} 的代表日文件中缺少 {season}")

            date_str = selected_days[season]

            df = load_prediction_result(result_dir, season)

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
                linewidth=1.2,
                label="Actual Power"
            )

            ax.plot(
                x_axis,
                pred_power,
                color=color_pred,
                linestyle="--",
                linewidth=1.2,
                label="Predicted Power"
            )

            # 横轴每 4 小时一个刻度：06:00, 10:00, 14:00, 18:00
            tick_spacing = 16
            tick_positions = x_axis[::tick_spacing]
            tick_labels = day_df.index[tick_positions].strftime("%H:%M")

            ax.set_xticks(tick_positions)
            ax.set_xticklabels(
                tick_labels,
                fontsize=7,
                fontproperties=FONT_PROP
            )

            ax.set_xlim(0, len(day_df) - 1)
            ax.margins(x=0)

            ax.grid(
                True,
                linestyle=":",
                linewidth=0.5,
                alpha=0.4,
                color="gray"
            )

            # 第一行显示季节标题
            if row_idx == 0:
                ax.set_title(
                    season,
                    fontsize=10,
                    fontweight="bold",
                    fontproperties=FONT_PROP,
                    pad=5
                )

            # 第一列显示天气类型
            if col_idx == 0:
                ax.set_ylabel(
                    weather,
                    fontsize=10,
                    fontweight="bold",
                    fontproperties=FONT_PROP
                )

            # 子图编号
            ax.text(
                0.03,
                0.92,
                f"({chr(97 + subplot_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                fontweight="bold",
                fontproperties=FONT_PROP,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                    pad=0.8
                )
            )

            subplot_idx += 1

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(FONT_PROP)

            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

    # 统一坐标轴标题
    fig.text(
        0.5,
        0.035,
        "Time",
        ha="center",
        fontsize=10,
        fontweight="bold",
        fontproperties=FONT_PROP
    )

    fig.text(
        0.025,
        0.5,
        "PV Power (kW)",
        va="center",
        rotation="vertical",
        fontsize=10,
        fontweight="bold",
        fontproperties=FONT_PROP
    )

    # 统一图例
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        prop=FONT_PROP,
        fontsize=9,
        handlelength=2.2,
        columnspacing=1.5
    )

    plt.tight_layout(rect=[0.045, 0.055, 1.0, 0.96])
    plt.subplots_adjust(wspace=0.12, hspace=0.20)

    png_path = os.path.join(SAVE_DIR, "all_weather_season_power_prediction.png")
    pdf_path = os.path.join(SAVE_DIR, "all_weather_season_power_prediction.pdf")
    svg_path = os.path.join(SAVE_DIR, "all_weather_season_power_prediction.svg")

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
    plot_all_weather_season_power()


if __name__ == "__main__":
    main()