import os
import math
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import SatellitePVDataset
from model.mymodel import MultiModalPVNet
from utils.config import load_config

# ================= 配置区域 =================

config = load_config("../../config/config.yaml")

# 这里改成你想测试的数据集路径
TEST_CSV_PATH = os.path.join("..", config["val_file_paths"]["series_file"])
TEST_SAT_DIR = os.path.join("..", config["val_file_paths"]["aligned_satellite_path"])
print(TEST_CSV_PATH)
print(TEST_SAT_DIR)

# 加载的模型权重路径
MODEL_WEIGHT_PATH = "../../checkpoints/good/Epoch:7-RMSE:0.0537-MAE:0.0223-MAPE:14.19%-R:98.00%.pth"

BATCH_SIZE = 32
NUM_WORKERS = 4

DROPOUT = 0.1
SELF_DEPTH = 2
CROSS_DEPTH = 2
FINAL_DIM = 64
TRANSFORMER_DIM = 64
HEADS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 指标计算函数 =================

@torch.no_grad()
def evaluate_metrics(model, loader, device):
    model.eval()

    sum_se = torch.tensor(0.0, device=device)
    sum_ae = torch.tensor(0.0, device=device)

    sum_ape = torch.tensor(0.0, device=device)
    sum_mape_count = torch.tensor(0.0, device=device)

    sum_x = torch.tensor(0.0, device=device)
    sum_y = torch.tensor(0.0, device=device)
    sum_xy = torch.tensor(0.0, device=device)
    sum_x2 = torch.tensor(0.0, device=device)
    sum_y2 = torch.tensor(0.0, device=device)

    total_count = torch.tensor(0.0, device=device)

    for batch in tqdm(loader, desc="Evaluating"):
        imgs = batch["x_images"].to(device)
        nums = batch["x_numeric"].to(device)
        targets = batch["y_power"].to(device)
        zeniths = batch["y_zenith"].to(device)
        y_clearsky = batch["y_clear_sky_ghi"].to(device)

        # 模型输出 CSI
        preds_csi, _, _, _, _ = model(imgs, nums)

        # 转成功率预测值
        preds_power = preds_csi * y_clearsky

        # 夜间置零，保持和原 validate_distributed 中一致
        night_mask = zeniths > 88
        preds_power = preds_power.clone()
        preds_power[night_mask] = 0.0

        error = preds_power - targets

        # RMSE / MAE
        sum_se += torch.sum(error ** 2)
        sum_ae += torch.sum(torch.abs(error))
        total_count += targets.numel()

        # MAPE：只统计 target 大于阈值的位置
        threshold = 0.01
        valid_mask = targets > threshold

        if valid_mask.sum() > 0:
            sum_ape += torch.sum(torch.abs(error[valid_mask] / targets[valid_mask]))
            sum_mape_count += valid_mask.sum()

        # Pearson R 相关系数
        sum_x += torch.sum(preds_power)
        sum_y += torch.sum(targets)
        sum_xy += torch.sum(preds_power * targets)
        sum_x2 += torch.sum(preds_power ** 2)
        sum_y2 += torch.sum(targets ** 2)

    N = total_count.item()

    rmse = math.sqrt(sum_se.item() / N)
    mae = sum_ae.item() / N

    if sum_mape_count.item() > 0:
        mape = sum_ape.item() / sum_mape_count.item() * 100.0
    else:
        mape = float("nan")

    numerator = N * sum_xy.item() - sum_x.item() * sum_y.item()
    denominator = math.sqrt(max(N * sum_x2.item() - sum_x.item() ** 2, 1e-8)) * \
                  math.sqrt(max(N * sum_y2.item() - sum_y.item() ** 2, 1e-8))

    r = numerator / denominator * 100.0 if denominator > 0 else 0.0

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE(%)": mape,
        "R(%)": r,
    }

    return metrics


# ================= 主函数 =================

def main():
    print(f"Using device: {DEVICE}")

    # 1. 加载测试数据集
    test_dataset = SatellitePVDataset(
        TEST_CSV_PATH,
        TEST_SAT_DIR,
        mode="val"
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=False
    )

    # 2. 构建模型
    model = MultiModalPVNet(
        final_dim=FINAL_DIM,
        transformer_dim=TRANSFORMER_DIM,
        heads=HEADS,
        self_depth=SELF_DEPTH,
        cross_depth=CROSS_DEPTH,
        output_seq_len=4,
        dropout=DROPOUT
    ).to(DEVICE)

    # 3. 加载权重
    if not os.path.exists(MODEL_WEIGHT_PATH):
        raise FileNotFoundError(f"模型权重不存在: {MODEL_WEIGHT_PATH}")

    state_dict = torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE)

    # 如果权重是 DDP 保存的，可能带有 module. 前缀，这里自动兼容
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k.replace("module.", "", 1)] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=True)

    print(f"Loaded model weights from: {MODEL_WEIGHT_PATH}")

    # 4. 计算指标
    metrics = evaluate_metrics(model, test_loader, DEVICE)

    print("-" * 60)
    print("Evaluation Results:")
    print(f"RMSE     : {metrics['RMSE']:.6f}")
    print(f"MAE      : {metrics['MAE']:.6f}")
    print(f"MAPE(%)  : {metrics['MAPE(%)']:.2f}%")
    print(f"R(%)     : {metrics['R(%)']:.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()
