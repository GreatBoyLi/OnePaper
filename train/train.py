import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import warnings

# 忽略所有 FutureWarning
# warnings.filterwarnings("ignore", category=FutureWarning)

# 或者更精确地只忽略包含 "custom_fwd" 或 "custom_bwd" 的警告
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_bwd.*")

# 导入自己写的模块
from dataset.dataset import SatellitePVDataset
from model.mymodel import MultiModalPVNet
from utils.config import load_config, setup_logger, plot_metrics_curve, plot_loss_curve, set_seed, init_weights
from utils.metrics import evaluate_metrics
from loss.loss import masked_mse_loss, gradient_rmse_loss, physics_constraint_loss, NRDCCALoss
from loss.optimizer import create_mamba_optimizer

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ================= 配置区域 (Hyperparameters) =================
# 加载配置
config = load_config("../config/config.yaml")

# 路径设置
TRAIN_CSV_PATH = config["train_file_paths"]["series_file"]
TRAIN_SAT_DIR = config["train_file_paths"]["aligned_satellite_path"]
VAL_CSV_PATH = config["val_file_paths"]["series_file"]
VAL_SAT_DIR = config["val_file_paths"]["aligned_satellite_path"]
SAVE_DIR = config["pkg_path"]

# 🌟 1. 初始化日志
logger = setup_logger(SAVE_DIR)

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
NUM_EPOCHS = 100
PATIENCE = 100
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3
SELF_DEPTH = 3
CROSS_DEPTH = 3
FINAL_DIM = 64
TRANSFORMER_DIM = 128
HEADS = 4
ALPHA = 1.0  # Masked MSE 的权重 (主 Loss)
BETA = 0.5  # Grad Loss (爬坡) 的权重 (通常设在 0.1~0.5)
GAMMA = 0.1  # Physics Loss 的权重 (通常不需要太大，能约束住就行)
LAMBDA_DCCA = 1e-3

# 硬件设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 使用设备: {DEVICE}")


# ============================================================
def train_one_epoch(model, loader, optimizer, device, scheduler, dcca_criterion):
    model.train()
    running_loss = 0.0
    running_mse = 0.0
    running_grad = 0.0
    running_phy = 0.0
    running_dcca = 0.0

    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        imgs = batch['x_images'].to(device)
        nums = batch['x_numeric'].to(device)
        targets = batch['y_power'].to(device)
        zeniths = batch['y_zenith'].to(device)  # 🌟 拿到天顶角

        optimizer.zero_grad()

        # 1. 模型输出是 CSI
        preds_csi, v_feat, t_feat, t_attn_weights = model(imgs, nums)

        # 🌟 2. 拿到预测窗口对应的理论晴空功率 (确保 Dataset 里有返回这个字段)
        y_clearsky = batch['y_clear_sky_ghi'].to(device)

        # 🌟 3. 物理还原：将 CSI 转换为实际预测功率
        preds_power = preds_csi * y_clearsky

        # 🌟 3. 分别计算三个子 Loss
        # (A) Masked MSE (主损失)
        loss_mse = masked_mse_loss(preds_power, targets, zeniths)

        # (B) Grad Loss (捕获爬坡趋势)
        loss_grad = gradient_rmse_loss(preds_power, targets, zeniths)

        # (C) Physics Loss (约束上限)
        loss_phy = physics_constraint_loss(preds_power, y_clearsky, zeniths)

        # 🌟 计算 NR-DCCA 表征对齐损失
        loss_dcca = dcca_criterion(v_feat, t_feat)

        # 4. 混合损失
        # 注意：DCCA 是特征层面的 loss，数值量级可能与 MSE 不同，通常给一个较小的系数
        total_loss = ALPHA * loss_mse + BETA * loss_grad + GAMMA * loss_phy + LAMBDA_DCCA * loss_dcca

        total_loss.backward()

        # 🛡️ 补丁：强行把梯度范数剪裁到 1.0 (这是 Mamba 训练的刚需！)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # ✅ 新增：每个 Batch 更新完权重后，微调一次学习率
        scheduler.step()

        # 记录日志
        running_loss += total_loss.item()
        running_mse += loss_mse.item()
        running_grad += loss_grad.item()
        running_phy += loss_phy.item()
        running_dcca += loss_dcca.item()

        # tqdm 进度条显示详细 Loss
        loop.set_postfix(
            T=f"{total_loss.item():.4f}",
            M=f"{loss_mse.item():.4f}",
            G=f"{loss_grad.item():.4f}",
            P=f"{loss_phy.item():.4f}",
            D=f"{loss_dcca.item():.4f}"
        )

    # 计算 Epoch 平均 Loss
    num_batches = len(loader)
    avg_total_loss = running_loss / num_batches

    # Logger 里打印详细的子 Loss 平均值
    logger.info(
        f"Avg MSE: {running_mse / num_batches:.6f}, Avg Grad: {running_grad / num_batches:.6f}, Avg Phy: {running_phy / num_batches:.6f}")

    return avg_total_loss


def validate(model, loader, device, dcca_criterion):
    model.eval()
    running_loss = 0.0

    # 🌟 新增：用于收集整个验证集的预测值和真实值
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['x_images'].to(device)
            nums = batch['x_numeric'].to(device)
            targets = batch['y_power'].to(device)
            zeniths = batch['y_zenith'].to(device)  # 🌟 拿到天顶角

            # 1. 模型预测 CSI
            preds_csi, v_feat, t_feat, t_attn_weights = model(imgs, nums)

            # 🌟 2. 拿到预测窗口对应的理论晴空功率 (确保 Dataset 里有返回这个字段)
            y_clearsky = batch['y_clear_sky_ghi'].to(device)

            # 🌟 3. 物理还原：将 CSI 转换为实际预测功率
            preds_power = preds_csi * y_clearsky

            # 🌟 3. 分别计算三个子 Loss
            # (A) Masked MSE (主损失)
            loss_mse = masked_mse_loss(preds_power, targets, zeniths)

            # (B) Grad Loss (捕获爬坡趋势)
            loss_grad = gradient_rmse_loss(preds_power, targets, zeniths)

            # (C) Physics Loss (约束上限)
            loss_phy = physics_constraint_loss(preds_power, y_clearsky, zeniths)

            # 🌟 计算 NR-DCCA 表征对齐损失
            loss_dcca = dcca_criterion(v_feat, t_feat)

            # 4. 混合损失
            # 注意：DCCA 是特征层面的 loss，数值量级可能与 MSE 不同，通常给一个较小的系数
            total_loss = ALPHA * loss_mse + BETA * loss_grad + GAMMA * loss_phy + LAMBDA_DCCA * loss_dcca

            # 3. 物理后处理：创建夜晚掩码
            # 如果天顶角 > 88°，说明太阳已落山或在地平线以下
            night_mask = zeniths > 88

            # 🌟 抹除夜晚的时段 (对 preds_power 操作)
            preds_power[night_mask] = 0.0

            loss = total_loss
            running_loss += loss.item()

            # 🌟 将还原并掩码后的【预测功率】存入列表，用于后续计算指标
            all_preds.append(preds_power.cpu())
            all_targets.append(targets.cpu())

    # 🌟 新增：将列表中的 Tensor 在第 0 维度（Batch 维度）拼接起来
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 🌟 新增：调用评估函数，计算四个指标
    metrics = evaluate_metrics(all_preds, all_targets)

    # 修改返回值，现在把 loss 和 指标字典 一起返回
    return running_loss / len(loader), metrics


def main():
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # 🌟 1. 在这里调用设置随机种子 (例如设为 42)
    set_seed(logger, 42)

    logger.info("📂 正在加载数据集...")
    if not os.path.exists(TRAIN_CSV_PATH) or not os.path.exists(TRAIN_SAT_DIR):
        logger.info(f"❌ 错误: 找不到数据文件。请检查路径:\n CSV: {TRAIN_CSV_PATH}\n SAT: {TRAIN_SAT_DIR}")
        return
    if not os.path.exists(VAL_CSV_PATH) or not os.path.exists(VAL_SAT_DIR):
        logger.info(f"❌ 错误: 找不到数据文件。请检查路径:\n CSV: {VAL_CSV_PATH}\n SAT: {VAL_SAT_DIR}")
        return

    train_dataset = SatellitePVDataset(TRAIN_CSV_PATH, TRAIN_SAT_DIR, mode="train")
    val_dataset = SatellitePVDataset(VAL_CSV_PATH, VAL_SAT_DIR, mode="val")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    logger.info(f"✅ 数据集加载完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    model = MultiModalPVNet(
        final_dim=FINAL_DIM,
        transformer_dim=TRANSFORMER_DIM,
        heads=HEADS,
        self_depth=SELF_DEPTH,
        cross_depth=CROSS_DEPTH,
        output_seq_len=4,  # 预测未来4个时间步
        dropout=DROPOUT
    ).to(DEVICE)

    # 🌟 2. 在这里应用权重初始化
    model.apply(init_weights)
    logger.info("✨ 模型权重初始化完成 (Xavier/Kaiming)")

    optimizer = create_mamba_optimizer(model, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 使用 AdamW 优化器
    # optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 2. 使用 OneCycleLR (自带 Warmup 和平滑余弦衰减)
    # max_lr 可以比原来稍微激进一点，比如 3e-4，因为有了 Warmup 保护
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),  # 必须传入每个 epoch 的 batch 数量
        pct_start=0.1,  # 前 10% 的时间 (即前 10 个 Epoch) 用于 Warmup 预热
        anneal_strategy='cos'  # 后续 90% 的时间进行余弦衰减
    )

    # 实例化 NR-DCCA (建议权重不要太大，作为一个辅助约束)
    dcca_criterion = NRDCCALoss(lambd=5e-3, noise_std=0.05, nr_weight=0.5).to(DEVICE)

    # 分别初始化四个指标的历史最佳记录
    # RMSE, MAE, MAPE 是越小越好，所以初始值设为正无穷大
    best_rmse = float('inf')
    best_mae = float('inf')
    best_mape = float('inf')
    # R (相关系数) 是越大越好，所以初始值设为负无穷大
    best_r = -float('inf')

    patience_counter = 0

    # 【新增 3】初始化列表用于存储 Loss
    train_loss_history = []
    val_loss_history = []

    # 🌟 新增：用于存储每轮的四个指标
    rmse_hist = []
    mae_hist = []
    mape_hist = []
    r_hist = []

    logger.info(f"🔥 开始训练 (Epochs: {NUM_EPOCHS})")
    logger.info("-" * 60)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, scheduler, dcca_criterion)
        val_loss, val_metrics = validate(model, val_loader, DEVICE, dcca_criterion)
        logger.info(val_metrics)
        # 获取当前刚刚被降下来的学习率
        current_lr = optimizer.param_groups[0]['lr']

        # 【新增 4】记录每一轮的 Loss
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        logger.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr}")

        # 获取当前 Epoch 的各项指标
        current_rmse = val_metrics['RMSE']
        current_mae = val_metrics['MAE']
        current_mape = val_metrics['MAPE(%)']
        current_r = val_metrics['R(%)']

        # 🌟 新增：记录到历史列表中
        rmse_hist.append(current_rmse)
        mae_hist.append(current_mae)
        mape_hist.append(current_mape)
        r_hist.append(current_r)

        # 设置一个标志位，只要有任何一个指标破纪录了，就重置早停计数器
        any_improvement = False

        # 🏆 1. 评判 RMSE (越小越好)
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR,
                                                        f"Epoch:{epoch + 1}-best_rmse_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))
            logger.info(
                f"Epoch:{epoch + 1}-best_rmse_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth")
            logger.info(f"   ⭐ [RMSE 冠军] 创新低: {best_rmse:.4f}，模型已保存！")

        # 🏆 2. 评判 MAE (越小越好)
        if current_mae < best_mae:
            best_mae = current_mae
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR,
                                                        f"Epoch:{epoch + 1}-best_mae_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))
            logger.info(
                f"Epoch:{epoch + 1}-best_mae_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth")
            logger.info(f"   ⭐ [MAE  冠军] 创新低: {best_mae:.4f}，模型已保存！")

        # 🏆 3. 评判 MAPE (越小越好)
        if current_mape < best_mape:
            best_mape = current_mape
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR,
                                                        f"Epoch:{epoch + 1}-best_mape_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))
            logger.info(
                f"Epoch:{epoch + 1}-best_mape_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth")
            logger.info(f"   ⭐ [MAPE 冠军] 创新低: {best_mape:.2f}%，模型已保存！")

        # 🏆 4. 评判 R (越大越好)
        if current_r > best_r:
            best_r = current_r
            any_improvement = True
            torch.save(model.state_dict(), os.path.join(SAVE_DIR,
                                                        f"Epoch:{epoch + 1}-best_r_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))
            logger.info(
                f"Epoch:{epoch + 1}-best_r_model-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth")
            logger.info(f"   🚀 [R 相关性冠军] 创新高: {best_r:.2f}%，模型已保存！")

        # 早停机制 (Early Stopping) 逻辑更新：
        # 只要这四个指标中有一个还在变好，我们就继续给模型机会
        if any_improvement:
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"   ⏳ 所有四项指标均未提升 ({patience_counter}/{PATIENCE})")

        if patience_counter >= PATIENCE:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            break

    logger.info("-" * 60)
    logger.info("🎉 训练结束！各指标的最佳模型已保存在: %s 目录下", SAVE_DIR)

    # 【新增 5】调用绘图函数
    plot_save_path = os.path.join(SAVE_DIR, "loss_curve.png")
    metrics_save_path = os.path.join(SAVE_DIR, "metrics_curve.png")
    plot_loss_curve(train_loss_history, val_loss_history, plot_save_path, logger)
    plot_metrics_curve(rmse_hist, mae_hist, mape_hist, r_hist, metrics_save_path, logger)


if __name__ == "__main__":
    main()
