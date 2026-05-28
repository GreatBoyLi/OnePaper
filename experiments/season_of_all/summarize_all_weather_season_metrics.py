import os
import numpy as np
import pandas as pd


# =========================
# 0. 路径配置
# =========================
SAVE_DIR = "./all_weather_season_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

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
# 1. 指标计算
# =========================
def calc_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE(%)": np.nan,
            "R(%)": np.nan,
            "Error_Mean": np.nan,
            "Error_STD": np.nan,
            "Sample_Number": 0,
        }

    error = y_pred - y_true

    rmse = np.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    valid_mape_mask = y_true > 0.01
    if valid_mape_mask.sum() > 0:
        mape = np.mean(np.abs(error[valid_mape_mask] / y_true[valid_mape_mask])) * 100.0
    else:
        mape = np.nan

    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        r = np.corrcoef(y_true, y_pred)[0, 1] * 100.0
    else:
        r = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "R(%)": r,
        "Error_Mean": np.mean(error),
        "Error_STD": np.std(error),
        "Sample_Number": len(y_true),
    }


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

    return day_df


# =========================
# 5. 统计每类天气和季节的指标
# =========================
def summarize_selected_days():
    rows = []
    all_true = []
    all_pred = []

    for weather in WEATHER_ORDER:
        result_dir = WEATHER_CONFIGS[weather]["result_dir"]
        selected_day_file = WEATHER_CONFIGS[weather]["selected_day_file"]

        selected_days = load_selected_days(result_dir, selected_day_file)

        weather_true = []
        weather_pred = []

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

            y_true = day_df["True_Power_kW"].values
            y_pred = day_df["Pred_Power_kW"].values

            metrics = calc_metrics(y_true, y_pred)

            rows.append({
                "Weather": weather,
                "Season": season,
                "Date": date_str,
                **metrics,
            })

            weather_true.extend(y_true)
            weather_pred.extend(y_pred)

            all_true.extend(y_true)
            all_pred.extend(y_pred)

        # 每个天气条件汇总
        weather_metrics = calc_metrics(weather_true, weather_pred)
        rows.append({
            "Weather": weather,
            "Season": "Average",
            "Date": "-",
            **weather_metrics,
        })

    # 总体汇总
    total_metrics = calc_metrics(all_true, all_pred)
    rows.append({
        "Weather": "Total",
        "Season": "Average",
        "Date": "-",
        **total_metrics,
    })

    result_df = pd.DataFrame(rows)

    save_path = os.path.join(SAVE_DIR, "all_weather_season_metrics.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(result_df)
    print(f"Metrics saved to: {save_path}")


if __name__ == "__main__":
    summarize_selected_days()