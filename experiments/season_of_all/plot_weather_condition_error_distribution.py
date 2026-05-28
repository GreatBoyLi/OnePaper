import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde


# =========================
# 0. 路径配置
# =========================
SAVE_DIR = "./weather_error_distribution_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

# 你的三类天气代表日和预测结果都在同一个目录
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

WEATHER_ORDER = ["Clear-sky", "Cloudy", "Rainy"]
SEASON_ORDER = ["Spring", "Summer", "Autumn", "Winter"]


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
    selected_day_path = os.path.join(result_dir, selected_day_file)

    if not os.path.exists(selected_day_path):
        raise FileNotFoundError(f"找不到代表日文件: {selected_day_path}")

    df = pd.read_csv(selected_day_path)

    season_dates = {}
    for _, row in df.iterrows():
        season_dates[row["Season"]] = str(row["Date"])

    return season_dates


# =========================
# 3. 读取某个季节预测结果
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
# 4. 提取某一天 06:00-20:00 数据
# =========================
def get_one_day_data(df, date_str, start_hour="06:00", end_hour="20:00"):
    day_df = df[df["Date"] == date_str].copy()

    if len(day_df) == 0:
        raise ValueError(f"找不到日期 {date_str} 的数据。")

    day_df = day_df.set_index("Time")

    start_time = pd.to_datetime(f"{date_str} {start_hour}")
    end_time = pd.to_datetime(f"{date_str} {end_hour}")

    day_df = day_df.loc[start_time:end_time].copy()

    return day_df


# =========================
# 5. 指标计算
# =========================
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    error = y_pred - y_true

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))
    error_mean = np.mean(error)
    error_std = np.std(error)

    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        # R 使用小数形式，例如 0.98，不再乘以 100
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = np.nan

    return rmse, mae, r, error_mean, error_std


# =========================
# 6. KDE 绘制
# =========================
def draw_error_kde(ax, errors, color="#d62728", x_range=None, tick_step=0.25):
    errors = np.asarray(errors, dtype=float)
    errors = errors[np.isfinite(errors)]

    if len(errors) < 5:
        return

    if x_range is None:
        x_min = np.percentile(errors, 1)
        x_max = np.percentile(errors, 99)

        max_abs = max(abs(x_min), abs(x_max), 0.1)
        x_min, x_max = -max_abs, max_abs
    else:
        x_min, x_max = x_range

    # KDE 曲线更细腻
    x_grid = np.linspace(x_min, x_max, 500)

    kde = gaussian_kde(errors)
    density = kde(x_grid)

    ax.plot(
        x_grid,
        density,
        color=color,
        linewidth=1.9,
        label="Ours"
    )

    ax.fill_between(
        x_grid,
        0,
        density,
        color=color,
        alpha=0.22
    )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=0.9,
        label="Unbiased"
    )

    ax.set_xlim(x_min, x_max)

    # =========================
    # 横轴刻度更细
    # =========================
    tick_start = np.floor(x_min / tick_step) * tick_step
    tick_end = np.ceil(x_max / tick_step) * tick_step

    xticks = np.arange(tick_start, tick_end + tick_step, tick_step)

    # 避免太多刻度导致拥挤
    if len(xticks) <= 9:
        ax.set_xticks(xticks)
    else:
        # 如果范围过大，自动放宽刻度
        larger_step = tick_step * 2
        tick_start = np.floor(x_min / larger_step) * larger_step
        tick_end = np.ceil(x_max / larger_step) * larger_step
        xticks = np.arange(tick_start, tick_end + larger_step, larger_step)
        ax.set_xticks(xticks)


# =========================
# 7. 右上角 GT-Pred 小散点图
# =========================
def add_scatter_inset(ax, y_true, y_pred):
    inset = ax.inset_axes([0.58, 0.53, 0.38, 0.38])

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return

    inset.scatter(
        y_true,
        y_pred,
        s=5,
        alpha=0.45,
        color="#d62728",
        edgecolors="none",
        rasterized=True
    )

    min_v = min(y_true.min(), y_pred.min())
    max_v = max(y_true.max(), y_pred.max())

    if abs(max_v - min_v) < 1e-8:
        min_v -= 0.1
        max_v += 0.1
    else:
        margin = (max_v - min_v) * 0.05
        min_v -= margin
        max_v += margin

    inset.plot(
        [min_v, max_v],
        [min_v, max_v],
        color="black",
        linestyle="--",
        linewidth=0.7
    )

    inset.set_xlim(min_v, max_v)
    inset.set_ylim(min_v, max_v)

    inset.set_xlabel("GT", fontsize=5.5, fontproperties=FONT_PROP)
    inset.set_ylabel("Pred", fontsize=5.5, fontproperties=FONT_PROP)

    inset.tick_params(axis="both", labelsize=5, pad=1)
    inset.grid(True, linestyle=":", linewidth=0.35, alpha=0.45)

    for label in inset.get_xticklabels() + inset.get_yticklabels():
        label.set_fontproperties(FONT_PROP)

    for spine in inset.spines.values():
        spine.set_linewidth(0.55)


# =========================
# 8. 收集全部 3×4 数据
# =========================
def collect_weather_season_data():
    data_dict = {}
    all_errors = []

    for weather in WEATHER_ORDER:
        result_dir = WEATHER_CONFIGS[weather]["result_dir"]
        selected_day_file = WEATHER_CONFIGS[weather]["selected_day_file"]

        selected_days = load_selected_days(result_dir, selected_day_file)

        for season in SEASON_ORDER:
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

            if "True_Power_kW" not in day_df.columns:
                raise ValueError(f"{weather}-{season} 缺少 True_Power_kW 列")

            if "Pred_Power_kW" not in day_df.columns:
                raise ValueError(f"{weather}-{season} 缺少 Pred_Power_kW 列")

            y_true = day_df["True_Power_kW"].values.astype(float)
            y_pred = day_df["Pred_Power_kW"].values.astype(float)
            error = y_pred - y_true

            data_dict[(weather, season)] = {
                "date": date_str,
                "y_true": y_true,
                "y_pred": y_pred,
                "error": error,
            }

            all_errors.extend(error[np.isfinite(error)].tolist())

    # 统一所有子图的 error 横轴范围，便于横向比较
    all_errors = np.asarray(all_errors, dtype=float)
    all_errors = all_errors[np.isfinite(all_errors)]

    if len(all_errors) > 0:
        low = np.percentile(all_errors, 1)
        high = np.percentile(all_errors, 99)
        max_abs = max(abs(low), abs(high), 0.1)

        # 稍微扩大边界，让曲线不贴边
        max_abs = max_abs * 1.10

        x_range = (-max_abs, max_abs)
    else:
        x_range = (-1.0, 1.0)

    return data_dict, x_range


# =========================
# 9. 绘制 3×4 Fig.3 风格图
# =========================
def plot_weather_season_fig3_style():
    data_dict, x_range = collect_weather_season_data()

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(14.5, 8.2),
        sharex=True,
        sharey=False
    )

    metrics_rows = []
    subplot_idx = 0

    for row_idx, weather in enumerate(WEATHER_ORDER):
        for col_idx, season in enumerate(SEASON_ORDER):
            ax = axes[row_idx, col_idx]

            item = data_dict[(weather, season)]
            date_str = item["date"]
            y_true = item["y_true"]
            y_pred = item["y_pred"]
            errors = item["error"]

            rmse, mae, r, error_mean, error_std = calc_metrics(y_true, y_pred)

            metrics_rows.append({
                "Weather": weather,
                "Season": season,
                "Date": date_str,
                "Sample_Number": int(np.isfinite(errors).sum()),
                "RMSE": rmse,
                "MAE": mae,
                "R": r,
                "Error_Mean": error_mean,
                "Error_STD": error_std,
            })

            draw_error_kde(
                ax,
                errors,
                color="#d62728",
                x_range=x_range,
                tick_step=0.25
            )

            add_scatter_inset(ax, y_true, y_pred)

            # 第一行显示季节标题
            if row_idx == 0:
                ax.set_title(
                    season,
                    fontsize=13,
                    fontweight="bold",
                    fontproperties=FONT_PROP,
                    pad=8
                )

            # 第一列显示天气类型和 Density
            if col_idx == 0:
                ax.set_ylabel(
                    f"{weather}\nDensity",
                    fontsize=11,
                    fontweight="bold",
                    fontproperties=FONT_PROP
                )

            # 最后一行显示横轴标题
            if row_idx == len(WEATHER_ORDER) - 1:
                ax.set_xlabel(
                    "Error (kW)",
                    fontsize=11,
                    fontproperties=FONT_PROP
                )

            # 子图编号
            ax.text(
                0.03,
                0.93,
                f"({chr(97 + subplot_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                fontweight="bold",
                fontproperties=FONT_PROP,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.65,
                    pad=0.8
                )
            )

            # 日期标注
            ax.text(
                0.97,
                0.93,
                date_str,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                fontproperties=FONT_PROP,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.70,
                    pad=0.8
                )
            )

            # 只显示 R 指标，小数形式，不显示百分号
            ax.text(
                0.05,
                0.18,
                f"$R$={r:.2f}",
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=8.8,
                fontproperties=FONT_PROP,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.78,
                    pad=1.2
                )
            )

            ax.grid(
                True,
                linestyle="-",
                linewidth=0.42,
                alpha=0.32,
                color="gray"
            )

            ax.tick_params(axis="both", labelsize=10)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(FONT_PROP)

            for spine in ax.spines.values():
                spine.set_linewidth(0.7)

            subplot_idx += 1

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
        fontsize=10,
        handlelength=2.2,
        columnspacing=1.5
    )

    plt.tight_layout(rect=[0.03, 0.03, 1.0, 0.96])
    plt.subplots_adjust(wspace=0.20, hspace=0.28)

    png_path = os.path.join(SAVE_DIR, "fig3_style_weather_season_error_distribution.png")
    pdf_path = os.path.join(SAVE_DIR, "fig3_style_weather_season_error_distribution.pdf")
    svg_path = os.path.join(SAVE_DIR, "fig3_style_weather_season_error_distribution.svg")

    plt.savefig(
        png_path,
        dpi=1200,
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

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(SAVE_DIR, "fig3_style_weather_season_error_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    print(f"Metrics saved to: {metrics_path}")
    print(metrics_df)

    plt.show()


# =========================
# 10. 主函数
# =========================
def main():
    plot_weather_season_fig3_style()


if __name__ == "__main__":
    main()