import os
import torch
import random
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 导入你自己写的模块 (与 train.py 保持一致)
from dataset.dataset import SatellitePVDataset
from model.mymodel import MultiModalPVNet
from utils.config import load_config, setup_logger

# ================= 1. 配置区域 =================
# 建议通过显卡 0 或 CPU 进行简单推理
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载配置
config = load_config("../config/config.yaml")

# 验证集/测试集路径 (这里复用验证集，如果你有单独的测试集路径请替换)
VAL_CSV_PATH = config["val_file_paths"]["series_file"]
VAL_SAT_DIR = config["val_file_paths"]["aligned_satellite_path"]

# 🌟 请将这里替换为你训练出来的、表现最好的模型权重文件名！
MODEL_WEIGHTS_PATH = os.path.join(config["pkg_path"], "Epoch:33-best_rmse_model-RMSE:0.0629-MAE:0.0275-MAPE:14.62%-R:97.39%.pth")

# 模型结构参数 (必须与 train.py 中完全一致)
FINAL_DIM = 64
TRANSFORMER_DIM = 128
HEADS = 4
SELF_DEPTH = 3
CROSS_DEPTH = 3
DROPOUT = 0.3
SEQ_LEN = 4  # 预测未来4个时间步

# 可视化样本数量
NUM_SAMPLES_TO_PLOT = 5


# ================= 2. 可视化绘图函数 =================
def plot_prediction(y_true, y_pred, sample_idx, save_dir):
    """
    绘制单样本的真实功率与预测功率对比图
    y_true: 真实功率序列 (长度为 SEQ_LEN)
    y_pred: 预测功率序列 (长度为 SEQ_LEN)
    """
    plt.figure(figsize=(8, 5))

    # x 轴的时间步长
    x_axis = np.arange(1, SEQ_LEN + 1)

    plt.plot(x_axis, y_true, marker='o', linestyle='-', color='blue', label='True Power', linewidth=2)
    plt.plot(x_axis, y_pred, marker='x', linestyle='--', color='red', label='Predicted Power', linewidth=2)

    plt.title(f'Photovoltaic Power Forecasting (Sample {sample_idx})', fontsize=14)
    plt.xlabel('Future Time Steps', fontsize=12)
    plt.ylabel('Power (W/m²)', fontsize=12)

    # 强制 x 轴显示整数时间步
    plt.xticks(x_axis)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(loc='best', fontsize=11)

    # 保存图片
    plot_path = os.path.join(save_dir, f'inference_sample_{sample_idx}.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    return plot_path


# ================= 3. 主推理逻辑 =================
def main():
    logger = setup_logger("./inference_logs")

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        logger.error(f"❌ 找不到模型权重文件: {MODEL_WEIGHTS_PATH}")
        logger.error("请手动修改 MODEL_WEIGHTS_PATH 变量，填入你真实生成的 .pth 文件名！")
        return

    logger.info("📂 正在加载测试/验证数据...")
    dataset = SatellitePVDataset(VAL_CSV_PATH, VAL_SAT_DIR, mode="val")
    # shuffle=True 方便每次运行都能随机抽到不同的样本看效果
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    logger.info("🏗️ 正在初始化模型并加载权重...")
    model = MultiModalPVNet(
        final_dim=FINAL_DIM,
        transformer_dim=TRANSFORMER_DIM,
        heads=HEADS,
        self_depth=SELF_DEPTH,
        cross_depth=CROSS_DEPTH,
        output_seq_len=SEQ_LEN,
        dropout=DROPOUT
    ).to(DEVICE)

    # 加载权重 (设置 map_location 以防你在单卡/CPU上推理多卡训练的模型)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=DEVICE))
    model.eval()  # 必须开启 eval 模式，关闭 Dropout 和 BatchNorm 的动态更新
    logger.info("✅ 模型加载成功，准备进行推理...")

    output_dir = os.path.join(config["pkg_path"], "inference_plots")
    os.makedirs(output_dir, exist_ok=True)

    plot_count = 0

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if plot_count >= NUM_SAMPLES_TO_PLOT:
                break

            imgs = batch['x_images'].to(DEVICE)
            nums = batch['x_numeric'].to(DEVICE)
            targets = batch['y_power'].to(DEVICE)
            zeniths = batch['y_zenith'].to(DEVICE)
            y_clearsky = batch['y_clear_sky_ghi'].to(DEVICE)

            # 1. 模型预测 CSI
            preds_csi = model(imgs, nums)

            # 2. 物理还原：CSI 还原为功率
            preds_power = preds_csi * y_clearsky

            # 3. 物理后处理：夜晚掩码强行置零
            night_mask = zeniths > 86
            preds_power[night_mask] = 0.0

            # --- 准备画图数据 ---
            # 剥离 batch 维度 (batch_size=1)，转到 cpu 并转为 numpy 数组
            true_seq = targets.squeeze(0).cpu().numpy()
            pred_seq = preds_power.squeeze(0).cpu().numpy()
            zenith_seq = zeniths.squeeze(0).cpu().numpy()

            # 过滤掉完全是晚上的无聊样本（如果未来 4 个时间步天顶角都大于 86，图上全是 0，没观赏性）
            if (zenith_seq > 86).all():
                continue

            # 绘制并保存曲线
            saved_path = plot_prediction(true_seq, pred_seq, i, output_dir)
            logger.info(f"📊 样本 {i} 的预测曲线已保存至: {saved_path}")

            plot_count += 1

    logger.info("🎉 推理可视化完成！快去文件夹里看看图画得怎么样吧。")


if __name__ == "__main__":
    main()