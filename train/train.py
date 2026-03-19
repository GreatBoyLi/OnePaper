import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# 导入自己写的模块
from dataset.dataset import SatellitePVDataset
from model.mymodel import MultiModalPVNet
from utils.config import load_config, setup_logger, plot_metrics_curve, plot_loss_curve
from utils.metrics import evaluate_metrics
from loss.loss import masked_mse_loss, DCCALoss

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ================= 配置区域 (Hyperparameters) =================
# 加载配置
config = load_config("../config/config.yaml")

# 路径设置
CSV_PATH = config["file_paths"]["series_file"]
SAT_DIR = config["file_paths"]["aligned_satellite_path"]
SAVE_DIR = "../checkpoints/"

# 🌟 1. 初始化日志
logger = setup_logger(SAVE_DIR)

# 训练参数
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 100
WEIGHT_DECAY = 1e-3
DROPOUT = 0.3
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
SELF_DEPTH = 3
CROSS_DEPTH = 3
FINAL_DIM = 64
TRANSFORMER_DIM = 128
HEADS = 4

# 硬件设置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"🚀 使用设备: {DEVICE}")


# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device, scheduler):
    model.train()
    running_loss = 0.0
    loop = tqdm(loader, desc="Training", leave=False)

    for batch in loop:
        imgs = batch['x_images'].to(device)
        nums = batch['x_numeric'].to(device)
        targets = batch['y'].to(device)
        zeniths = batch['y_zenith'].to(device)  # 🌟 拿到天顶角

        optimizer.zero_grad()
        # preds, v_feat, t_feat = model(imgs, nums)
        preds = model(imgs, nums)

        # 🌟 1. 计算 Masked MSE (不管黑夜的死活，只算白天的预测误差)
        # loss_mse = criterion_mse(preds, targets)  # 预测准不准
        loss_mse = masked_mse_loss(preds, targets, zeniths)

        # 计算多目标 Loss
        # 🌟 2. 筛选出属于“白天”的样本，才送去算 DCCA
        # 只要这个样本的预测窗口里有任意一个时刻 <= 85°，我们就认为它包含了白天特征
        # daytime_sample_mask = (zeniths <= 86.0).any(dim=1)
        # # 提取白天样本的特征
        # valid_v_feat = v_feat[daytime_sample_mask]
        # valid_t_feat = t_feat[daytime_sample_mask]
        #
        # # 如果这个 Batch 里至少有 2 个白天样本 (DCCA 算相关性至少需要 2 个样本)
        # if valid_v_feat.size(0) > 1:
        #     loss_dcca = criterion_dcca(valid_v_feat, valid_t_feat)
        # else:
        #     loss_dcca = torch.tensor(0.0, device=device)
        # loss = criterion(preds, targets)
        # 论文总损失公式
        loss = loss_mse  # + lambda_c * loss_dcca
        loss.backward()
        optimizer.step()

        # ✅ 新增：每个 Batch 更新完权重后，微调一次学习率
        scheduler.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())  # , loss_mse=loss_mse.item(), loss_dcca=loss_dcca.item())

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    # 🌟 新增：用于收集整个验证集的预测值和真实值
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch['x_images'].to(device)
            nums = batch['x_numeric'].to(device)
            targets = batch['y'].to(device)
            zeniths = batch['y_zenith'].to(device)  # 🌟 拿到天顶角

            # preds, v_feat, t_feat = model(imgs, nums)
            preds = model(imgs, nums)

            # 计算 Loss
            # 🌟 1. 计算 Masked MSE (不管黑夜的死活，只算白天的预测误差)
            # loss_mse = criterion_mse(preds, targets)  # 预测准不准
            loss_mse = masked_mse_loss(preds, targets, zeniths)

            # 3. 物理后处理：创建夜晚掩码
            # 如果天顶角 > 86°，说明太阳已落山或在地平线以下
            night_mask = zeniths > 86
            # 把夜晚的时段抹除
            preds[night_mask] = 0.0

            # # 计算多目标 Loss
            # # 🌟 2. 筛选出属于“白天”的样本，才送去算 DCCA
            # # 只要这个样本的预测窗口里有任意一个时刻 <= 85°，我们就认为它包含了白天特征
            # daytime_sample_mask = (zeniths <= 86.0).any(dim=1)
            # # 提取白天样本的特征
            # valid_v_feat = v_feat[daytime_sample_mask]
            # valid_t_feat = t_feat[daytime_sample_mask]
            #
            # # 如果这个 Batch 里至少有 2 个白天样本 (DCCA 算相关性至少需要 2 个样本)
            # if valid_v_feat.size(0) > 1:
            #     loss_dcca = criterion_dcca(valid_v_feat, valid_t_feat)
            # else:
            #     loss_dcca = torch.tensor(0.0, device=device)
            loss = loss_mse  # + lambda_c * loss_dcca
            running_loss += loss.item()

            # 🌟 新增：把数据转移到 CPU 并存入列表，防止撑爆显存
            all_preds.append(preds.cpu())
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

    logger.info("📂 正在加载数据集...")
    if not os.path.exists(CSV_PATH) or not os.path.exists(SAT_DIR):
        logger.info(f"❌ 错误: 找不到数据文件。请检查路径:\n CSV: {CSV_PATH}\n SAT: {SAT_DIR}")
        return

    train_dataset = SatellitePVDataset(CSV_PATH, SAT_DIR, mode='train', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)
    val_dataset = SatellitePVDataset(CSV_PATH, SAT_DIR, mode='val', train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO)

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
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    # 2. 使用 OneCycleLR (自带 Warmup 和平滑余弦衰减)
    # max_lr 可以比原来稍微激进一点，比如 3e-4，因为有了 Warmup 保护
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=NUM_EPOCHS,
        steps_per_epoch=len(train_loader),  # 必须传入每个 epoch 的 batch 数量
        pct_start=0.1,  # 前 10% 的时间 (即前 10 个 Epoch) 用于 Warmup 预热
        anneal_strategy='cos'  # 后续 90% 的时间进行余弦衰减
    )

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
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE, scheduler)
        val_loss, val_metrics = validate(model, val_loader, criterion, DEVICE)
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
    logger.info("🎉 训练结束！最佳模型已保存在: %s", os.path.join(SAVE_DIR, "best_model.pth"))

    # 【新增 5】调用绘图函数
    plot_save_path = os.path.join(SAVE_DIR, "loss_curve.png")
    metrics_save_path = os.path.join(SAVE_DIR, "metrics_curve.png")
    plot_loss_curve(train_loss_history, val_loss_history, plot_save_path, logger)
    plot_metrics_curve(rmse_hist, mae_hist, mape_hist, r_hist, metrics_save_path, logger)


if __name__ == "__main__":
    criterion_mse = nn.MSELoss()
    criterion_dcca = DCCALoss()
    lambda_c = 0.01  # 论文中的平衡权重 \lambda_C，通常取 0.01 到 0.1
    main()
