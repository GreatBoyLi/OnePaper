import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

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

# 光伏系统额定容量，用于将标幺值还原为 kW
PV_CAPACITY_KW = 5.0

MODEL_PATH = "../../../checkpoints/0428_depth_2_mse-dcca/Epoch:23-RMSE:0.0537-MAE:0.0194-MAPE:14.42%-R:97.97%.pth"

# 卫星图像路径
VAL_SAT_DIR = "../../../data/val/crop_himawari/15min"

SAVE_DIR = "./season_rainy_clear_prediction_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 1. 春夏秋冬月份 CSV 配置
# Alice Springs 在南半球：
# Spring: 10月
# Summer: 1月
# Autumn: 4月
# Winter: 7月
# =========================
SEASON_FILES = [
    {
        "name": "Spring",
        "csv_path": "../../../data/val/validate_chun.csv",
    },
    {
        "name": "Summer",
        "csv_path": "../../../data/val/validate_xia.csv",
    },
    {
        "name": "Autumn",
        "csv_path": "../../../data/val/validate_qiu.csv",
    },
    {
        "name": "Winter",
        "csv_path": "../../../data/val/validate_dong.csv",
    },
]


# =========================
# 2. 加载模型
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

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型权重不存在: {MODEL_PATH}")

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # 兼容 DDP 保存的权重
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "", 1)] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    print(f"Loaded model weights from: {MODEL_PATH}")
    return model


# =========================
# 3. 推理一个月份文件
# =========================
def infer_one_file(model, csv_path):
    dataset = SatellitePVDataset(csv_path, VAL_SAT_DIR, mode="val")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    true_power_history = []
    pred_power_history = []
    gamma_history = []

    print(f"正在推断：{csv_path}")

    for batch in loader:
        with torch.no_grad():
            imgs = batch["x_images"].to(DEVICE)
            nums = batch["x_numeric"].to(DEVICE)
            targets = batch["y_power"].to(DEVICE)
            zeniths = batch["y_zenith"].to(DEVICE)
            y_clearsky = batch["y_clear_sky_ghi"].to(DEVICE)

            # 模型输出：
            # preds_csi, v_feat, t_feat, t_attn_weights, g_weights
            preds_csi, v_feat, t_feat, t_attn_weights, g_weights = model(imgs, nums)

            # CSI 转功率，当前仍是标幺值
            preds_power = preds_csi * y_clearsky

            # 夜间置零
            night_mask = zeniths > 88
            preds_power = preds_power.clone()
            preds_power[night_mask] = 0.0

            preds_power_np = preds_power.detach().cpu().numpy()
            y_true_np = targets.detach().cpu().numpy()

            # gamma 可能是标量，也可能是向量，这里统一取均值
            gamma_np = g_weights.detach().cpu().numpy()
            gamma_mean = float(np.mean(gamma_np))

            # 只取未来第一个预测步长，并将标幺值还原为 kW
            true_power_kw = float(y_true_np[0, 0]) * PV_CAPACITY_KW
            pred_power_kw = float(preds_power_np[0, 0]) * PV_CAPACITY_KW

            true_power_history.append(true_power_kw)
            pred_power_history.append(pred_power_kw)
            gamma_history.append(gamma_mean)

    return (
        np.array(true_power_history),
        np.array(pred_power_history),
        np.array(gamma_history),
    )


# =========================
# 4. 对齐时间轴并保存完整季节预测结果
# =========================
def build_and_save_dataframe(season_name, csv_path, true_power_kw, pred_power_kw, gamma_values):
    df_raw = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    if df_raw.index.tz is not None:
        df_raw.index = df_raw.index.tz_localize(None)

    # 和 Dataset 内部尽量保持一致：剔除 Active_Power 缺失
    if "Active_Power" in df_raw.columns:
        valid_df = df_raw.dropna(subset=["Active_Power"]).copy()
    else:
        valid_df = df_raw.copy()

    min_len = min(len(true_power_kw), len(valid_df) - HISTORY_LEN)

    if min_len <= 0:
        raise ValueError(f"有效样本长度不足，请检查 CSV 文件：{csv_path}")

    real_time_axis = valid_df.index[HISTORY_LEN: HISTORY_LEN + min_len]

    result_df = pd.DataFrame(
        {
            "Time": real_time_axis,
            "True_Power_kW": true_power_kw[:min_len],
            "Pred_Power_kW": pred_power_kw[:min_len],
            "Gamma": gamma_values[:min_len],
            "Season": season_name,
        }
    )

    # 保留辅助字段，方便后续筛选 Rainy/Clear-sky 日和画图
    aux_df = valid_df.iloc[HISTORY_LEN: HISTORY_LEN + min_len].copy().reset_index(drop=True)

    possible_cols = [
        "CSI",
        "Solar_Zenith_Raw",
        "Active_Power",
        "Global_Horizontal_Radiation",
        "Clear_Sky_GHI",
        "GHI",
        "clear_sky_ghi",
        "y_clear_sky_ghi",
        "Temperature",
        "Air_Temperature",
        "Relative_Humidity",
        "Wind_Speed",
    ]

    for col in possible_cols:
        if col in aux_df.columns:
            result_df[col] = aux_df[col].values

    save_path = os.path.join(
        SAVE_DIR,
        f"prediction_results_{season_name.lower()}.csv"
    )

    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"已保存：{save_path}")
    print(f"样本数：{len(result_df)}")
    print(f"时间范围：{result_df['Time'].iloc[0]}  ->  {result_df['Time'].iloc[-1]}")

    return save_path


# =========================
# 5. 统计每日天气特征
# =========================
def build_daily_weather_stats(result_csv_path, zenith_threshold=85):
    df = pd.read_csv(result_csv_path, parse_dates=["Time"])

    if "CSI" not in df.columns:
        raise ValueError(f"{result_csv_path} 中没有 CSI 列，无法筛选 Rainy/Clear-sky 日期。")

    if "Solar_Zenith_Raw" not in df.columns:
        raise ValueError(f"{result_csv_path} 中没有 Solar_Zenith_Raw 列，无法筛选白天样本。")

    df["Date"] = df["Time"].dt.date.astype(str)

    # 只统计白天样本
    df_day = df[df["Solar_Zenith_Raw"] <= zenith_threshold].copy()
    df_day = df_day.dropna(subset=["CSI"])

    if len(df_day) == 0:
        raise ValueError(f"{result_csv_path} 没有可用白天 CSI 数据。")

    daily_stats = (
        df_day
        .groupby("Date")
        .agg(
            Mean_CSI=("CSI", "mean"),
            Min_CSI=("CSI", "min"),
            Max_CSI=("CSI", "max"),
            Sample_Number=("CSI", "count"),
            Rainy_Ratio=("CSI", lambda x: np.mean(x < 0.4)),
            Cloudy_Ratio=("CSI", lambda x: np.mean((x >= 0.4) & (x < 0.8))),
            Clear_Ratio=("CSI", lambda x: np.mean(x >= 0.8)),
        )
        .reset_index()
    )

    return df, daily_stats


# =========================
# 6. 自动筛选每个季节的 Rainy + Clear-sky 两天
# =========================
def select_rainy_and_clear_days(
    result_csv_path,
    rainy_threshold=0.4,
    clear_threshold=0.8,
    zenith_threshold=85,
    min_day_samples=20,
):
    df, daily_stats = build_daily_weather_stats(
        result_csv_path=result_csv_path,
        zenith_threshold=zenith_threshold,
    )

    # 保证样本数量充足
    valid_stats = daily_stats[daily_stats["Sample_Number"] >= min_day_samples].copy()

    if len(valid_stats) == 0:
        print(f"警告：没有满足 min_day_samples={min_day_samples} 的日期，改用全部日期。")
        valid_stats = daily_stats.copy()

    # ---------- 1) 选 Rainy day ----------
    # 优先：Rainy_Ratio 最高
    # 其次：Mean_CSI 最低
    # 再次：样本数最多
    rainy_candidates = valid_stats.copy()
    rainy_candidates = rainy_candidates.sort_values(
        by=["Rainy_Ratio", "Mean_CSI", "Sample_Number"],
        ascending=[False, True, False]
    )

    rainy_row = rainy_candidates.iloc[0]
    rainy_date = rainy_row["Date"]

    # ---------- 2) 选 Clear-sky day ----------
    # 避免和 Rainy day 是同一天
    clear_candidates = valid_stats[valid_stats["Date"] != rainy_date].copy()

    if len(clear_candidates) == 0:
        print("警告：除 Rainy day 外没有其他日期，Clear-sky day 可能与 Rainy day 相同。")
        clear_candidates = valid_stats.copy()

    # 优先：Clear_Ratio 最高
    # 其次：Mean_CSI 最高
    # 再次：样本数最多
    clear_candidates = clear_candidates.sort_values(
        by=["Clear_Ratio", "Mean_CSI", "Sample_Number"],
        ascending=[False, False, False]
    )

    clear_row = clear_candidates.iloc[0]
    clear_date = clear_row["Date"]

    print("\n筛选结果：")
    print(
        f"Rainy day      : {rainy_date} | "
        f"Mean CSI={rainy_row['Mean_CSI']:.4f}, "
        f"Rainy Ratio={rainy_row['Rainy_Ratio']:.2%}, "
        f"Clear Ratio={rainy_row['Clear_Ratio']:.2%}, "
        f"Samples={int(rainy_row['Sample_Number'])}"
    )

    print(
        f"Clear-sky day  : {clear_date} | "
        f"Mean CSI={clear_row['Mean_CSI']:.4f}, "
        f"Rainy Ratio={clear_row['Rainy_Ratio']:.2%}, "
        f"Clear Ratio={clear_row['Clear_Ratio']:.2%}, "
        f"Samples={int(clear_row['Sample_Number'])}"
    )

    selected = [
        {
            "Weather_Type": "Rainy",
            "Date": rainy_date,
            "Mean_CSI": rainy_row["Mean_CSI"],
            "Min_CSI": rainy_row["Min_CSI"],
            "Max_CSI": rainy_row["Max_CSI"],
            "Rainy_Ratio": rainy_row["Rainy_Ratio"],
            "Cloudy_Ratio": rainy_row["Cloudy_Ratio"],
            "Clear_Ratio": rainy_row["Clear_Ratio"],
            "Sample_Number": int(rainy_row["Sample_Number"]),
        },
        {
            "Weather_Type": "Clear-sky",
            "Date": clear_date,
            "Mean_CSI": clear_row["Mean_CSI"],
            "Min_CSI": clear_row["Min_CSI"],
            "Max_CSI": clear_row["Max_CSI"],
            "Rainy_Ratio": clear_row["Rainy_Ratio"],
            "Cloudy_Ratio": clear_row["Cloudy_Ratio"],
            "Clear_Ratio": clear_row["Clear_Ratio"],
            "Sample_Number": int(clear_row["Sample_Number"]),
        },
    ]

    return selected, df, daily_stats


# =========================
# 7. 保存每个季节筛选出的两天数据
# =========================
def save_selected_two_day_data(season_name, result_df, selected_days):
    result_df = result_df.copy()
    result_df["Date"] = result_df["Time"].dt.date.astype(str)

    selected_parts = []

    for item in selected_days:
        weather_type = item["Weather_Type"]
        date_str = item["Date"]

        one_day_df = result_df[result_df["Date"] == date_str].copy()
        one_day_df["Selected_Weather_Type"] = weather_type

        selected_parts.append(one_day_df)

    selected_df = pd.concat(selected_parts, axis=0, ignore_index=True)

    save_path = os.path.join(
        SAVE_DIR,
        f"selected_two_days_{season_name.lower()}.csv"
    )

    selected_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"两天数据已保存：{save_path}")
    print(f"样本数：{len(selected_df)}")

    return save_path


# =========================
# 8. 主函数
# =========================
def main():
    model = load_model()

    saved_prediction_files = []
    saved_two_day_files = []
    selected_all_rows = []

    for item in SEASON_FILES:
        season_name = item["name"]
        csv_path = item["csv_path"]

        if not os.path.exists(csv_path):
            print(f"跳过：文件不存在 -> {csv_path}")
            continue

        print("\n" + "=" * 70)
        print(f"开始处理季节：{season_name}")
        print("=" * 70)

        true_power_kw, pred_power_kw, gamma_values = infer_one_file(
            model=model,
            csv_path=csv_path,
        )

        result_csv_path = build_and_save_dataframe(
            season_name=season_name,
            csv_path=csv_path,
            true_power_kw=true_power_kw,
            pred_power_kw=pred_power_kw,
            gamma_values=gamma_values,
        )

        saved_prediction_files.append(result_csv_path)

        selected_days, result_df, daily_stats = select_rainy_and_clear_days(
            result_csv_path=result_csv_path,
            rainy_threshold=0.4,
            clear_threshold=0.8,
            zenith_threshold=85,
            min_day_samples=20,
        )

        # 保存每日统计，方便检查
        daily_stats_path = os.path.join(
            SAVE_DIR,
            f"daily_weather_stats_{season_name.lower()}.csv"
        )
        daily_stats.to_csv(daily_stats_path, index=False, encoding="utf-8-sig")
        print(f"每日天气统计已保存：{daily_stats_path}")

        # 保存当前季节的两天数据
        two_day_path = save_selected_two_day_data(
            season_name=season_name,
            result_df=result_df,
            selected_days=selected_days,
        )
        saved_two_day_files.append(two_day_path)

        # 汇总代表日信息
        for order_idx, selected in enumerate(selected_days, start=1):
            selected_all_rows.append({
                "Season": season_name,
                "Day_Order": order_idx,
                "Weather_Type": selected["Weather_Type"],
                "Date": selected["Date"],
                "Mean_CSI": selected["Mean_CSI"],
                "Min_CSI": selected["Min_CSI"],
                "Max_CSI": selected["Max_CSI"],
                "Rainy_Ratio": selected["Rainy_Ratio"],
                "Cloudy_Ratio": selected["Cloudy_Ratio"],
                "Clear_Ratio": selected["Clear_Ratio"],
                "Sample_Number": selected["Sample_Number"],
            })

    print("\n" + "=" * 70)
    print("全部预测结果保存完成：")
    for f in saved_prediction_files:
        print(f)

    print("\n全部两天筛选数据保存完成：")
    for f in saved_two_day_files:
        print(f)

    if len(selected_all_rows) > 0:
        selected_days_path = os.path.join(
            SAVE_DIR,
            "selected_rainy_clear_days.csv"
        )

        selected_days_df = pd.DataFrame(selected_all_rows)
        selected_days_df.to_csv(
            selected_days_path,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\n春夏秋冬 Rainy + Clear-sky 代表日已保存：{selected_days_path}")
        print(selected_days_df)


if __name__ == "__main__":
    main()