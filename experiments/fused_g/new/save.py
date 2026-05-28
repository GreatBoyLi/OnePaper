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

# 上一步筛选出的 Rainy + Clear-sky 日期表
SELECTED_DAYS_CSV = "./season_rainy_clear_prediction_results/selected_rainy_clear_days.csv"

# 保存推理后的 gating 数据
SAVE_DIR = "./gating_saved_results_two_days"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 1. 春夏秋冬原始月份 CSV 配置
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
# 2. 读取代表日文件
# =========================
def load_selected_days(selected_days_csv):
    """
    读取 selected_rainy_clear_days.csv。

    期望列：
    Season, Day_Order, Weather_Type, Date, ...
    """
    if not os.path.exists(selected_days_csv):
        raise FileNotFoundError(f"找不到代表日文件: {selected_days_csv}")

    df = pd.read_csv(selected_days_csv)

    required_cols = ["Season", "Weather_Type", "Date"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{selected_days_csv} 缺少必要列: {col}")

    df["Season"] = df["Season"].astype(str)
    df["Weather_Type"] = df["Weather_Type"].astype(str)
    df["Date"] = df["Date"].astype(str)

    selected_map = {}

    for season, sub in df.groupby("Season"):
        selected_map[season] = sub[["Weather_Type", "Date"]].to_dict("records")

    print("已读取代表日：")
    for season, items in selected_map.items():
        print(f"{season}:")
        for item in items:
            print(f"  {item['Weather_Type']}: {item['Date']}")

    return selected_map


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
# 4. 从 Dataset 中获取样本对应的预测时间
# =========================
def get_sample_target_time(dataset, sample_idx):
    """
    对于 Dataset 的第 sample_idx 个样本：
    real_idx = dataset.valid_indices[sample_idx]
    历史窗口: real_idx : real_idx + input_len
    预测窗口第一个时刻: real_idx + input_len

    因为我们只保存未来第一个预测步，所以时间轴用这个 target_time。
    """
    real_idx = dataset.valid_indices[sample_idx]
    target_idx = real_idx + dataset.input_len
    target_time = dataset.data.index[target_idx]
    return target_time, target_idx


# =========================
# 5. 推理一个季节文件，返回完整月度结果
# =========================
def infer_one_season(model, season_name, csv_path):
    dataset = SatellitePVDataset(csv_path, VAL_SAT_DIR, mode="val")

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    rows = []

    print(f"\n正在推断季节：{season_name}")
    print(f"CSV: {csv_path}")
    print(f"有效样本数: {len(dataset)}")

    for sample_idx, batch in enumerate(loader):
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

            target_time, target_idx = get_sample_target_time(dataset, sample_idx)

            row = {
                "Time": target_time,
                "True_Power_kW": true_power_kw,
                "Pred_Power_kW": pred_power_kw,
                "Gamma": gamma_mean,
                "Season": season_name,
            }

            # 从 Dataset 内部清洗后的 data 中取辅助字段，保证和样本时间完全对齐
            possible_cols = [
                "CSI",
                "Solar_Zenith_Raw",
                "Active_Power",
                "Power_Norm",
                "Global_Horizontal_Radiation",
                "Clear_Sky_GHI",
                "GHI",
                "clear_sky_ghi",
                "y_clear_sky_ghi",
                "Weather_Temperature_Celsius",
                "Weather_Relative_Humidity",
                "Temperature",
                "Air_Temperature",
                "Relative_Humidity",
                "Wind_Speed",
            ]

            for col in possible_cols:
                if col in dataset.data.columns:
                    row[col] = dataset.data.iloc[target_idx][col]

            rows.append(row)

    result_df = pd.DataFrame(rows)

    if len(result_df) == 0:
        raise ValueError(f"{season_name} 没有生成任何推理结果，请检查数据。")

    result_df["Time"] = pd.to_datetime(result_df["Time"])
    result_df["Date"] = result_df["Time"].dt.date.astype(str)

    return result_df


# =========================
# 6. 根据代表日筛选两天数据并保存
# =========================
def save_selected_two_days(season_name, full_result_df, selected_days):
    """
    selected_days 示例：
    [
        {"Weather_Type": "Rainy", "Date": "2024-10-xx"},
        {"Weather_Type": "Clear-sky", "Date": "2024-10-xx"},
    ]
    """
    selected_parts = []

    for item in selected_days:
        weather_type = item["Weather_Type"]
        date_str = item["Date"]

        one_day_df = full_result_df[full_result_df["Date"] == date_str].copy()

        if len(one_day_df) == 0:
            print(f"警告：{season_name} 中找不到日期 {date_str} 的推理结果。")
            continue

        one_day_df["Selected_Weather_Type"] = weather_type
        selected_parts.append(one_day_df)

        # 单独保存每个季节每种天气的一天
        single_save_name = f"gating_results_{season_name.lower()}_{weather_type.lower().replace('-', '_')}.csv"
        single_save_path = os.path.join(SAVE_DIR, single_save_name)

        one_day_df.to_csv(single_save_path, index=False, encoding="utf-8-sig")

        print(f"已保存单日数据：{single_save_path}")
        print(f"  {weather_type} | {date_str} | 样本数: {len(one_day_df)}")

    if len(selected_parts) == 0:
        print(f"警告：{season_name} 没有任何代表日数据被保存。")
        return None

    selected_df = pd.concat(selected_parts, axis=0, ignore_index=True)
    selected_df = selected_df.sort_values(["Selected_Weather_Type", "Time"]).reset_index(drop=True)

    save_path = os.path.join(
        SAVE_DIR,
        f"gating_results_{season_name.lower()}_two_days.csv"
    )

    selected_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"已保存两天合并数据：{save_path}")
    print(f"样本数：{len(selected_df)}")

    return save_path


# =========================
# 7. 保存 Gamma 统计
# =========================
def save_gamma_summary(all_selected_files):
    rows = []

    for csv_path in all_selected_files:
        if csv_path is None or not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, parse_dates=["Time"])

        if "Selected_Weather_Type" not in df.columns:
            df["Selected_Weather_Type"] = "Unknown"

        for (season, weather_type, date_str), sub in df.groupby(
            ["Season", "Selected_Weather_Type", "Date"]
        ):
            gamma = sub["Gamma"].dropna().values

            rows.append(
                {
                    "Season": season,
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

    save_path = os.path.join(SAVE_DIR, "gating_gamma_summary_two_days.csv")
    summary_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\nGamma 统计结果：")
    print(summary_df)
    print(f"统计结果已保存：{save_path}")

    return summary_df


# =========================
# 8. 主函数
# =========================
def main():
    selected_map = load_selected_days(SELECTED_DAYS_CSV)

    model = load_model()

    saved_files = []

    for item in SEASON_FILES:
        season_name = item["name"]
        csv_path = item["csv_path"]

        if not os.path.exists(csv_path):
            print(f"跳过：文件不存在 -> {csv_path}")
            continue

        if season_name not in selected_map:
            print(f"跳过：代表日文件中没有 {season_name}")
            continue

        full_result_df = infer_one_season(
            model=model,
            season_name=season_name,
            csv_path=csv_path,
        )

        # 可选：保存完整季节推理结果，方便检查
        full_save_path = os.path.join(
            SAVE_DIR,
            f"gating_results_{season_name.lower()}_full.csv"
        )
        full_result_df.to_csv(full_save_path, index=False, encoding="utf-8-sig")
        print(f"完整季节推理结果已保存：{full_save_path}")

        two_day_save_path = save_selected_two_days(
            season_name=season_name,
            full_result_df=full_result_df,
            selected_days=selected_map[season_name],
        )

        saved_files.append(two_day_save_path)

    print("\n全部两天 gating 数据保存完成：")
    for f in saved_files:
        if f is not None:
            print(f)

    save_gamma_summary(saved_files)


if __name__ == "__main__":
    main()