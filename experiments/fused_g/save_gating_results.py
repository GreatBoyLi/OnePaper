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

MODEL_PATH = "../../checkpoints/good/Epoch:7-RMSE:0.0537-MAE:0.0223-MAPE:14.19%-R:98.00%.pth"
VAL_SAT_DIR = "../../data/val/crop_himawari/15min"

SAVE_DIR = "./gating_saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)


# =========================
# 1. 多天气 CSV 配置
# =========================
WEATHER_FILES = [
    {
        "name": "Clear",
        "csv_path": "../../data/val/hardcore_clear_weather_2days.csv",
    },
    {
        "name": "Cloudy",
        "csv_path": "../../data/val/hardcore_mixed_weather_2days.csv",
    },
    {
        "name": "Ramp",
        "csv_path": "../../data/val/hardcore_ramp_weather_2days.csv",
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
# 3. 推理一个天气文件
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
    gamma_history = []

    print(f"正在推断：{csv_path}")

    for batch in val_loader:
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
def build_and_save_dataframe(weather_name, csv_path, true_power_kw, pred_power_kw, gamma_values):
    df_val = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    if df_val.index.tz is not None:
        df_val.index = df_val.index.tz_localize(None)

    valid_df = df_val.dropna(subset=["Active_Power"])

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
        }
    )

    # 尽量保留原 CSV 中的辅助字段，方便后面分析
    aux_df = valid_df.iloc[HISTORY_LEN: HISTORY_LEN + min_len].copy().reset_index()

    possible_cols = [
        "CSI",
        "Solar_Zenith_Raw",
        "Active_Power",
        "Global_Horizontal_Radiation",
        "Clear_Sky_GHI",
        "GHI",
        "clear_sky_ghi",
        "y_clear_sky_ghi",
    ]

    for col in possible_cols:
        if col in aux_df.columns:
            result_df[col] = aux_df[col].values

    save_path = os.path.join(SAVE_DIR, f"gating_results_{weather_name.lower()}.csv")
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"已保存：{save_path}")
    print(f"样本数：{len(result_df)}")

    return save_path


# =========================
# 5. 主函数
# =========================
def main():
    model = load_model()

    saved_files = []

    for item in WEATHER_FILES:
        weather_name = item["name"]
        csv_path = item["csv_path"]

        if not os.path.exists(csv_path):
            print(f"跳过：文件不存在 -> {csv_path}")
            continue

        true_power_kw, pred_power_kw, gamma_values = infer_one_weather(model, csv_path)

        save_path = build_and_save_dataframe(
            weather_name=weather_name,
            csv_path=csv_path,
            true_power_kw=true_power_kw,
            pred_power_kw=pred_power_kw,
            gamma_values=gamma_values,
        )

        saved_files.append(save_path)

    print("\n全部保存完成：")
    for f in saved_files:
        print(f)


if __name__ == "__main__":
    main()