import os
import math
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from tqdm import tqdm

from model.mymodel import MultiModalPVNet
from dataset.dataset import SatellitePVDataset
from utils.config import load_config


# =========================
# 0. 基础配置
# =========================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

config = load_config("../../config/config.yaml")

TEST_CSV_PATH = os.path.join("..", config["val_file_paths"]["series_file"])
TEST_SAT_DIR = os.path.join("..", config["val_file_paths"]["aligned_satellite_path"])

SAVE_DIR = "./noise_robustness_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# 模型权重路径：改成你自己的 pth
MODEL_PATH = "../../checkpoints/good/Epoch:7-RMSE:0.0537-MAE:0.0223-MAPE:14.19%-R:98.00%.pth"

# 模型参数：必须和训练时一致
SELF_DEPTH = 2
CROSS_DEPTH = 2
FINAL_DIM = 64
TRANSFORMER_DIM = 64
HEADS = 4
DROPOUT = 0.3
OUTPUT_SEQ_LEN = 4

BATCH_SIZE = 32
NUM_WORKERS = 4

# x_numeric 中 Power_Norm 的列索引
POWER_COL = 0

# 太阳天顶角大于该阈值时置零
NIGHT_ZENITH_THRESHOLD = 88

# MAPE 计算时过滤过小功率
MAPE_THRESHOLD = 0.01

# 噪声等级
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

# 每个噪声等级重复次数
NUM_REPEATS = 3

# 三类扰动
NOISE_TYPES = ["gaussian", "random_drop", "structured_drop"]

# 是否只统计白天预测点
DAYTIME_ONLY = True
DAY_ZENITH_THRESHOLD = 86.0

# 随机种子
BASE_SEED = 42


# =========================
# 0.1 每个指标的绘图配置
# =========================
# 说明：
# 1. ylim = None 表示自动纵坐标范围；
# 2. ylim = (下限, 上限) 表示手动固定纵坐标范围；
# 3. ytick_num 控制纵轴刻度数量；
# 4. lower_margin_ratio / upper_margin_ratio 控制自动纵坐标放宽程度；
# 5. legend_loc 控制图例位置。
METRIC_PLOT_CONFIGS = {
    "RMSE": {
        "ylabel": "RMSE",
        "save_name": "noise_robustness_rmse",
        "mean_col": "RMSE_mean",
        "std_col": "RMSE_std",

        # 例如想固定范围，可改成："ylim": (0.03, 0.20)
        "ylim": (0.03, 0.21),

        "ytick_num": 7,
        "lower_margin_ratio": 0.0,
        "upper_margin_ratio": 0.8,
        "legend_loc": "best",
    },

    "MAE": {
        "ylabel": "MAE",
        "save_name": "noise_robustness_mae",
        "mean_col": "MAE_mean",
        "std_col": "MAE_std",

        # 例如想固定范围，可改成："ylim": (0.02, 0.14)
        "ylim": (0.02, 0.14),

        "ytick_num": 7,
        "lower_margin_ratio": 0.8,
        "upper_margin_ratio": 1.2,
        "legend_loc": "best",
    },

    "MAPE": {
        "ylabel": "MAPE",
        "save_name": "noise_robustness_mape",
        "mean_col": "MAPE_mean",
        "std_col": "MAPE_std",

        # 现在 MAPE 是小数，例如 0.13 表示 13%
        # 例如想固定范围，可改成："ylim": (0.13, 0.70)
        "ylim": (0.13, 0.73),

        "ytick_num": 7,
        "lower_margin_ratio": 0.6,
        "upper_margin_ratio": 1.5,
        "legend_loc": "best",
    },

    "R": {
        "ylabel": "R",
        "save_name": "noise_robustness_r",
        "mean_col": "R_mean",
        "std_col": "R_std",

        # 现在 R 是小数，例如 0.98 表示 98%
        # 如果看起来下降太陡，建议手动固定，例如："ylim": (0.94, 1.00)
        "ylim": (0.40, 1.00),

        "ytick_num": 7,
        "lower_margin_ratio": 2.0,
        "upper_margin_ratio": 0.8,
        "legend_loc": "best",
    },
}


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
# 2. 固定随机种子
# =========================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    # 兼容 DDP 保存的 module.xxx
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
# 4. 构造噪声
# =========================
def apply_gaussian_noise(nums, noise_level, power_col=0):
    """
    对历史 Power_Norm 加高斯噪声。
    noise_level 是归一化功率尺度下的标准差。
    """
    nums_noisy = nums.clone()

    if noise_level <= 0:
        return nums_noisy

    power_seq = nums_noisy[:, :, power_col]
    noise = torch.randn_like(power_seq) * noise_level

    nums_noisy[:, :, power_col] = power_seq + noise
    nums_noisy[:, :, power_col] = torch.clamp(
        nums_noisy[:, :, power_col],
        0.0,
        1.0
    )

    return nums_noisy


def apply_random_drop(nums, drop_ratio, power_col=0):
    """
    随机缺失：按 drop_ratio 随机把历史 Power_Norm 的部分点置 0。
    """
    nums_noisy = nums.clone()

    if drop_ratio <= 0:
        return nums_noisy

    power_seq = nums_noisy[:, :, power_col]
    mask = torch.rand_like(power_seq) < drop_ratio

    power_seq = power_seq.masked_fill(mask, 0.0)
    nums_noisy[:, :, power_col] = power_seq

    return nums_noisy


def apply_structured_drop(nums, drop_ratio, power_col=0):
    """
    连续片段缺失：每个样本随机选择一段连续历史 Power_Norm 置 0。
    drop_ratio 控制缺失长度占输入长度比例。
    """
    nums_noisy = nums.clone()

    if drop_ratio <= 0:
        return nums_noisy

    batch_size, seq_len, _ = nums_noisy.shape

    drop_len = max(1, int(round(seq_len * drop_ratio)))
    drop_len = min(drop_len, seq_len)

    for b in range(batch_size):
        start = torch.randint(
            low=0,
            high=seq_len - drop_len + 1,
            size=(1,),
            device=nums_noisy.device
        ).item()

        nums_noisy[b, start:start + drop_len, power_col] = 0.0

    return nums_noisy


def apply_noise(nums, noise_type, noise_level, power_col=0):
    if noise_type == "clean":
        return nums.clone()

    if noise_type == "gaussian":
        return apply_gaussian_noise(nums, noise_level, power_col)

    if noise_type == "random_drop":
        return apply_random_drop(nums, noise_level, power_col)

    if noise_type == "structured_drop":
        return apply_structured_drop(nums, noise_level, power_col)

    raise ValueError(f"未知噪声类型: {noise_type}")


# =========================
# 5. 指标计算
# =========================
def compute_metrics_from_arrays(y_true, y_pred, zenith=None):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    if zenith is not None and DAYTIME_ONLY:
        zenith = np.asarray(zenith, dtype=float).reshape(-1)
        mask = mask & np.isfinite(zenith) & (zenith <= DAY_ZENITH_THRESHOLD)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "RMSE": np.nan,
            "MAE": np.nan,
            "MAPE": np.nan,
            "R": np.nan,
            "Sample_Number": 0,
        }

    error = y_pred - y_true

    rmse = math.sqrt(np.mean(error ** 2))
    mae = np.mean(np.abs(error))

    mape_mask = y_true > MAPE_THRESHOLD
    if np.sum(mape_mask) > 0:
        # 不乘 100，保持小数形式
        mape = np.mean(np.abs(error[mape_mask] / y_true[mape_mask]))
    else:
        mape = np.nan

    if len(y_true) > 1 and np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        # 不乘 100，保持小数形式
        r = np.corrcoef(y_true, y_pred)[0, 1]
    else:
        r = np.nan

    return {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R": r,
        "Sample_Number": len(y_true),
    }


# =========================
# 6. 单次推理评估
# =========================
@torch.no_grad()
def evaluate_one_setting(model, loader, noise_type, noise_level):
    model.eval()

    all_true = []
    all_pred = []
    all_zenith = []

    for batch in tqdm(loader, desc=f"{noise_type}-{noise_level:.2f}", leave=False):
        imgs = batch["x_images"].to(DEVICE)
        nums = batch["x_numeric"].to(DEVICE)
        targets = batch["y_power"].to(DEVICE)
        zeniths = batch["y_zenith"].to(DEVICE)
        y_clearsky = batch["y_clear_sky_ghi"].to(DEVICE)

        nums_noisy = apply_noise(
            nums=nums,
            noise_type=noise_type,
            noise_level=noise_level,
            power_col=POWER_COL
        )

        outputs = model(imgs, nums_noisy)

        if isinstance(outputs, tuple):
            preds_csi = outputs[0]
        else:
            preds_csi = outputs

        preds_power = preds_csi * y_clearsky

        # 夜间预测置零
        night_mask = zeniths > NIGHT_ZENITH_THRESHOLD
        preds_power = preds_power.clone()
        preds_power[night_mask] = 0.0

        all_true.append(targets.detach().cpu().numpy())
        all_pred.append(preds_power.detach().cpu().numpy())
        all_zenith.append(zeniths.detach().cpu().numpy())

    all_true = np.concatenate(all_true, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_zenith = np.concatenate(all_zenith, axis=0)

    metrics = compute_metrics_from_arrays(
        y_true=all_true,
        y_pred=all_pred,
        zenith=all_zenith
    )

    return metrics


# =========================
# 7. 鲁棒性实验主循环
# =========================
def run_noise_robustness_experiment():
    set_seed(BASE_SEED)

    model = load_model()

    dataset = SatellitePVDataset(
        TEST_CSV_PATH,
        TEST_SAT_DIR,
        mode="val"
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    print(f"Test samples: {len(dataset)}")
    print(f"Power noise column: x_numeric[:, :, {POWER_COL}] = Power_Norm")

    rows = []

    for noise_type in NOISE_TYPES:
        for noise_level in NOISE_LEVELS:
            repeat_metrics = []

            for repeat_idx in range(NUM_REPEATS):
                seed = BASE_SEED + repeat_idx
                set_seed(seed)

                metrics = evaluate_one_setting(
                    model=model,
                    loader=loader,
                    noise_type=noise_type,
                    noise_level=noise_level
                )

                repeat_metrics.append(metrics)

                rows.append({
                    "Noise_Type": noise_type,
                    "Noise_Level": noise_level,
                    "Repeat": repeat_idx,
                    "RMSE": metrics["RMSE"],
                    "MAE": metrics["MAE"],
                    "MAPE": metrics["MAPE"],
                    "R": metrics["R"],
                    "Sample_Number": metrics["Sample_Number"],
                })

                print(
                    f"[{noise_type}] level={noise_level:.2f}, repeat={repeat_idx} | "
                    f"RMSE={metrics['RMSE']:.6f}, "
                    f"MAE={metrics['MAE']:.6f}, "
                    f"MAPE={metrics['MAPE']:.6f}, "
                    f"R={metrics['R']:.6f}"
                )

            avg_rmse = np.nanmean([m["RMSE"] for m in repeat_metrics])
            avg_mae = np.nanmean([m["MAE"] for m in repeat_metrics])
            avg_mape = np.nanmean([m["MAPE"] for m in repeat_metrics])
            avg_r = np.nanmean([m["R"] for m in repeat_metrics])

            print(
                f"==> AVG [{noise_type}] level={noise_level:.2f} | "
                f"RMSE={avg_rmse:.6f}, "
                f"MAE={avg_mae:.6f}, "
                f"MAPE={avg_mape:.6f}, "
                f"R={avg_r:.6f}"
            )

    result_df = pd.DataFrame(rows)

    result_df = result_df[
        [
            "Noise_Type",
            "Noise_Level",
            "Repeat",
            "RMSE",
            "MAE",
            "MAPE",
            "R",
            "Sample_Number",
        ]
    ]

    csv_path = os.path.join(SAVE_DIR, "robustness_metrics.csv")
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"\nMetrics saved to: {csv_path}")

    return result_df


# =========================
# 8. 聚合结果
# =========================
def aggregate_results(result_df):
    agg_df = (
        result_df
        .groupby(["Noise_Type", "Noise_Level"], as_index=False)
        .agg(
            RMSE_mean=("RMSE", "mean"),
            RMSE_std=("RMSE", "std"),
            MAE_mean=("MAE", "mean"),
            MAE_std=("MAE", "std"),
            MAPE_mean=("MAPE", "mean"),
            MAPE_std=("MAPE", "std"),
            R_mean=("R", "mean"),
            R_std=("R", "std"),
            Sample_Number=("Sample_Number", "mean"),
        )
    )

    agg_df = agg_df[
        [
            "Noise_Type",
            "Noise_Level",
            "RMSE_mean",
            "RMSE_std",
            "MAE_mean",
            "MAE_std",
            "MAPE_mean",
            "MAPE_std",
            "R_mean",
            "R_std",
            "Sample_Number",
        ]
    ]

    save_path = os.path.join(SAVE_DIR, "robustness_metrics_summary.csv")
    agg_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Summary saved to: {save_path}")

    return agg_df


# =========================
# 9. 绘图函数：每个指标单独配置
# =========================
def plot_metric_curve(agg_df, config):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))

    metric_mean_col = config["mean_col"]
    metric_std_col = config["std_col"]
    ylabel = config["ylabel"]
    save_name = config["save_name"]

    line_styles = {
        "gaussian": "-",
        "random_drop": "--",
        "structured_drop": "-.",
    }

    labels = {
        "gaussian": "Gaussian noise",
        "random_drop": "Random drop",
        "structured_drop": "Structured drop",
    }

    all_y_values = []

    for noise_type in NOISE_TYPES:
        sub = agg_df[agg_df["Noise_Type"] == noise_type].copy()
        sub = sub.sort_values("Noise_Level")

        x = sub["Noise_Level"].values
        y = sub[metric_mean_col].values
        y_std = sub[metric_std_col].fillna(0.0).values

        all_y_values.extend(y[np.isfinite(y)].tolist())

        ax.plot(
            x,
            y,
            linestyle=line_styles.get(noise_type, "-"),
            marker="o",
            linewidth=1.8,
            markersize=4.2,
            label=labels.get(noise_type, noise_type),
        )

        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            alpha=0.12
        )

    # =========================
    # 每个指标单独控制纵坐标
    # =========================
    if config.get("ylim") is not None:
        lower, upper = config["ylim"]
        ax.set_ylim(lower, upper)

        yticks = np.linspace(lower, upper, config.get("ytick_num", 7))
        ax.set_yticks(yticks)

    else:
        all_y_values = np.asarray(all_y_values, dtype=float)
        all_y_values = all_y_values[np.isfinite(all_y_values)]

        if len(all_y_values) > 0:
            y_min = np.min(all_y_values)
            y_max = np.max(all_y_values)
            y_range = y_max - y_min

            if y_range < 1e-8:
                y_range = max(abs(y_max), 1e-3)

            lower_margin_ratio = config.get("lower_margin_ratio", 0.8)
            upper_margin_ratio = config.get("upper_margin_ratio", 1.2)

            if ylabel == "R":
                lower = max(0.0, y_min - lower_margin_ratio * y_range)
                upper = min(1.0, y_max + upper_margin_ratio * y_range)

                if upper - lower < 0.03:
                    center = (upper + lower) / 2.0
                    lower = max(0.0, center - 0.015)
                    upper = min(1.0, center + 0.015)
            else:
                lower = max(0.0, y_min - lower_margin_ratio * y_range)
                upper = y_max + upper_margin_ratio * y_range

                if upper - lower < 0.01:
                    upper = y_max + 0.01

            ax.set_ylim(lower, upper)

            yticks = np.linspace(lower, upper, config.get("ytick_num", 7))
            ax.set_yticks(yticks)

    ax.set_xlabel(
        r"Perturbation Level $\sigma$",
        fontsize=11,
        fontweight="bold",
        fontproperties=FONT_PROP
    )

    ax.set_ylabel(
        ylabel,
        fontsize=11,
        fontweight="bold",
        fontproperties=FONT_PROP
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.45
    )

    ax.legend(
        loc=config.get("legend_loc", "best"),
        frameon=True,
        prop=FONT_PROP,
        fontsize=9
    )

    ax.tick_params(axis="both", labelsize=10)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FONT_PROP)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()

    png_path = os.path.join(SAVE_DIR, save_name + ".png")
    pdf_path = os.path.join(SAVE_DIR, save_name + ".pdf")
    svg_path = os.path.join(SAVE_DIR, save_name + ".svg")

    plt.savefig(png_path, dpi=900, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.02)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    print(f"Saved: {svg_path}")

    plt.close(fig)


def plot_all_curves(agg_df):
    for metric_name in ["RMSE", "MAE", "MAPE", "R"]:
        plot_metric_curve(
            agg_df=agg_df,
            config=METRIC_PLOT_CONFIGS[metric_name]
        )


# =========================
# 10. 主函数
# =========================
def main():
    result_df = run_noise_robustness_experiment()
    agg_df = aggregate_results(result_df)
    plot_all_curves(agg_df)


if __name__ == "__main__":
    main()