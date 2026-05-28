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

MODEL_PATH = "../../checkpoints/0428_depth_2_mse-dcca/Epoch:23-RMSE:0.0537-MAE:0.0194-MAPE:14.42%-R:97.97%.pth"

# 卫星图像路径
VAL_SAT_DIR = "../../data/val/crop_himawari/15min"

SAVE_DIR = "./season_prediction_results"
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
        "csv_path": "../../data/val/validate_chun.csv",
    },
    {
        "name": "Summer",
        "csv_path": "../../data/val/validate_xia.csv",
    },
    {
        "name": "Autumn",
        "csv_path": "../../data/val/validate_qiu.csv",
    },
    {
        "name": "Winter",
        "csv_path": "../../data/val/validate_dong.csv",
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
# 4. 对齐时间轴并保存
# =========================
def build_and_save_dataframe(season_name, csv_path, true_power_kw, pred_power_kw, gamma_values):
    # 尽量兼容你的原始 CSV：默认第一列是时间索引
    df_raw = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    if df_raw.index.tz is not None:
        df_raw.index = df_raw.index.tz_localize(None)

    # 和 Dataset 内部保持一致：剔除 Active_Power 缺失
    if "Active_Power" in df_raw.columns:
        valid_df = df_raw.dropna(subset=["Active_Power"]).copy()
    else:
        valid_df = df_raw.copy()

    min_len = min(len(true_power_kw), len(valid_df) - HISTORY_LEN)

    if min_len <= 0:
        raise ValueError(f"有效样本长度不足，请检查 CSV 文件：{csv_path}")

    # 第一个预测样本对应 HISTORY_LEN 之后的时刻
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

    # 保留原 CSV 中的辅助字段，方便后面筛选晴天日和画图
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
# 5. 可选：从每个季节结果中自动筛选晴天日
# =========================
def select_clear_day(result_csv_path, csi_threshold=0.8, zenith_threshold=85):
    df = pd.read_csv(result_csv_path, parse_dates=["Time"])

    if "CSI" not in df.columns:
        print(f"跳过晴天日筛选：{result_csv_path} 中没有 CSI 列")
        return None

    if "Solar_Zenith_Raw" not in df.columns:
        print(f"跳过晴天日筛选：{result_csv_path} 中没有 Solar_Zenith_Raw 列")
        return None

    df["Date"] = df["Time"].dt.date.astype(str)

    df_day = df[df["Solar_Zenith_Raw"] <= zenith_threshold].copy()
    df_day = df_day.dropna(subset=["CSI"])

    daily_csi = (
        df_day
        .groupby("Date")["CSI"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(daily_csi) == 0:
        print(f"没有可用白天 CSI 数据：{result_csv_path}")
        return None

    clear_days = daily_csi[daily_csi >= csi_threshold]

    if len(clear_days) > 0:
        selected_date = clear_days.index[0]
        selected_csi = clear_days.iloc[0]
    else:
        selected_date = daily_csi.index[0]
        selected_csi = daily_csi.iloc[0]

    print(f"晴天代表日：{selected_date}, daily mean CSI = {selected_csi:.4f}")

    return selected_date


# =========================
# 6. 主函数
# =========================
def main():
    model = load_model()

    saved_files = []
    selected_clear_days = {}

    for item in SEASON_FILES:
        season_name = item["name"]
        csv_path = item["csv_path"]

        if not os.path.exists(csv_path):
            print(f"跳过：文件不存在 -> {csv_path}")
            continue

        true_power_kw, pred_power_kw, gamma_values = infer_one_file(
            model=model,
            csv_path=csv_path,
        )

        save_path = build_and_save_dataframe(
            season_name=season_name,
            csv_path=csv_path,
            true_power_kw=true_power_kw,
            pred_power_kw=pred_power_kw,
            gamma_values=gamma_values,
        )

        saved_files.append(save_path)

        selected_date = select_clear_day(save_path)
        if selected_date is not None:
            selected_clear_days[season_name] = selected_date

    print("\n全部预测结果保存完成：")
    for f in saved_files:
        print(f)

    print("\n各季节自动筛选的晴天代表日：")
    for season, date_str in selected_clear_days.items():
        print(f"{season}: {date_str}")

    # 保存晴天代表日，后面画图代码可以直接读取
    if len(selected_clear_days) > 0:
        clear_day_path = os.path.join(SAVE_DIR, "selected_clear_days.csv")
        pd.DataFrame(
            [
                {"Season": season, "Date": date_str}
                for season, date_str in selected_clear_days.items()
            ]
        ).to_csv(clear_day_path, index=False, encoding="utf-8-sig")

        print(f"\n晴天代表日已保存：{clear_day_path}")


if __name__ == "__main__":
    main()