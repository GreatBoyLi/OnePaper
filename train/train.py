import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import math
import warnings

# 忽略所有特定的警告
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_fwd.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.custom_bwd.*")

# 导入自己写的模块
from dataset.dataset import SatellitePVDataset
from model.mymodel import MultiModalPVNet
from utils.config import load_config, setup_logger, plot_metrics_curve, plot_loss_curve, set_seed, init_weights
from loss.loss import masked_mse_loss, gradient_rmse_loss, physics_constraint_loss, NRDCCALoss, AdaptiveLossWeighter
from loss.optimizer import create_mamba_optimizer

# ================= 配置区域 (Hyperparameters) =================
config = load_config("../config/config.yaml")

TRAIN_CSV_PATH = config["train_file_paths"]["series_file"]
TRAIN_SAT_DIR = config["train_file_paths"]["aligned_satellite_path"]
VAL_CSV_PATH = config["val_file_paths"]["series_file"]
VAL_SAT_DIR = config["val_file_paths"]["aligned_satellite_path"]
SAVE_DIR = config["pkg_path"]

# 🌟 预训练模型路径 (如果是微调，填入 pth 文件路径；如果是从头训练，保持为空字符串 "")
PRETRAINED_MODEL_PATH = "../checkpoints/Epoch:4-RMSE:0.0392-MAE:0.0211-MAPE:7.51%-R:99.28%.pth"
LEARNING_RATE = 2e-8

BATCH_SIZE = 32
# ⚠️ 注意：如果是微调 (加载了模型)，建议将学习率调小，例如 3e-5！
# LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
PATIENCE = 100
WEIGHT_DECAY = 1e-2
DROPOUT = 0.3
SELF_DEPTH = 3
CROSS_DEPTH = 3
FINAL_DIM = 64
TRANSFORMER_DIM = 128
HEADS = 4

# 🌟 总开关：是否开启自适应动态权重
AUTO_LOSS = False

# 固定 Loss 权重 (当 AUTO_LOSS = False 时生效)
ALPHA = 10.0  # MSE
BETA = 1.0  # Grad MSE
GAMMA = 0.5  # Physics
LAMBDA_DCCA = 0.004  # NR-DCCA


# ================= 分布式初始化 =================
def init_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    else:
        return 0, 1, 0


# ================= 训练与验证逻辑 =================
def train_one_epoch(model, loss_weighter, loader, optimizer, device, scheduler, dcca_criterion, rank):
    model.train()

    # 仅当使用了自适应权重时，才设置为训练模式
    if AUTO_LOSS and loss_weighter is not None:
        loss_weighter.train()

    running_loss = torch.tensor(0.0).to(device)

    loop = tqdm(loader, desc="Training", leave=False, disable=(rank != 0))

    for batch in loop:
        imgs = batch['x_images'].to(device)
        nums = batch['x_numeric'].to(device)
        targets = batch['y_power'].to(device)
        zeniths = batch['y_zenith'].to(device)
        y_clearsky = batch['y_clear_sky_ghi'].to(device)

        optimizer.zero_grad()

        # 模型预测
        preds_csi, v_feat, t_feat, _ = model(imgs, nums)
        preds_power = preds_csi * y_clearsky

        # 计算各个基础 Loss
        loss_mse = masked_mse_loss(preds_power, targets, zeniths)
        loss_grad = gradient_rmse_loss(preds_power, targets, zeniths)
        loss_phy = physics_constraint_loss(preds_power, y_clearsky, zeniths)
        loss_dcca = dcca_criterion(v_feat, t_feat)

        # 🌟 根据开关选择 Loss 融合方式
        if AUTO_LOSS:
            losses = [loss_mse, loss_grad, loss_phy, loss_dcca]
            total_loss = loss_weighter(losses)
        else:
            # total_loss = ALPHA * loss_mse + BETA * loss_grad + GAMMA * loss_phy + LAMBDA_DCCA * loss_dcca
            total_loss = ALPHA * loss_mse + LAMBDA_DCCA * loss_dcca
        total_loss.backward()

        # 裁剪模型参数梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        running_loss += total_loss.detach()

        if rank == 0:
            if AUTO_LOSS:
                # 动态获取当前的自适应权重 (兼容 DDP)
                lw_module = loss_weighter.module if dist.is_initialized() else loss_weighter
                cur_weights = lw_module.get_current_weights()
                loop.set_postfix(
                    Loss=f"{total_loss.item():.4f}",
                    W_M=f"{cur_weights[0]:.4f}",
                    W_G=f"{cur_weights[1]:.4f}",
                    W_P=f"{cur_weights[2]:.4f}",
                    W_D=f"{cur_weights[3]:.4f}"
                )
            else:
                loop.set_postfix(Loss=f"{total_loss.item():.4f}")

    if dist.is_initialized():
        dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1

    avg_total_loss = running_loss.item() / (len(loader) * world_size)
    return avg_total_loss


def validate_distributed(model, loss_weighter, loader, device, dcca_criterion, rank):
    model.eval()

    if AUTO_LOSS and loss_weighter is not None:
        loss_weighter.eval()

    sum_loss = torch.tensor(0.0).to(device)
    total_count = torch.tensor(0.0).to(device)
    sum_se = torch.tensor(0.0).to(device)
    sum_ae = torch.tensor(0.0).to(device)
    sum_ape = torch.tensor(0.0).to(device)
    sum_mape_count = torch.tensor(0.0).to(device)
    sum_x = torch.tensor(0.0).to(device)
    sum_y = torch.tensor(0.0).to(device)
    sum_xy = torch.tensor(0.0).to(device)
    sum_x2 = torch.tensor(0.0).to(device)
    sum_y2 = torch.tensor(0.0).to(device)

    with torch.no_grad():
        for batch in loader:
            imgs = batch['x_images'].to(device)
            nums = batch['x_numeric'].to(device)
            targets = batch['y_power'].to(device)
            zeniths = batch['y_zenith'].to(device)
            y_clearsky = batch['y_clear_sky_ghi'].to(device)

            preds_csi, v_feat, t_feat, _ = model(imgs, nums)
            preds_power = preds_csi * y_clearsky

            loss_mse = masked_mse_loss(preds_power, targets, zeniths)
            loss_grad = gradient_rmse_loss(preds_power, targets, zeniths)
            loss_phy = physics_constraint_loss(preds_power, y_clearsky, zeniths)
            loss_dcca = dcca_criterion(v_feat, t_feat)

            # 🌟 验证集同样根据开关选择 Loss 融合方式
            if AUTO_LOSS:
                losses = [loss_mse, loss_grad, loss_phy, loss_dcca]
                total_loss = loss_weighter(losses)
            else:
                # total_loss = ALPHA * loss_mse + BETA * loss_grad + GAMMA * loss_phy + LAMBDA_DCCA * loss_dcca
                total_loss = ALPHA * loss_mse + LAMBDA_DCCA * loss_dcca
            sum_loss += total_loss.detach()

            night_mask = zeniths > 88
            preds_power[night_mask] = 0.0

            error = preds_power - targets
            sum_se += torch.sum(error ** 2)
            sum_ae += torch.sum(torch.abs(error))

            # ========= 修改 MAPE 的计算方式 =========
            THRESHOLD = 0.01
            valid_mask = targets > THRESHOLD

            if valid_mask.sum() > 0:
                sum_ape += torch.sum(torch.abs(error[valid_mask] / targets[valid_mask]))

            sum_x += torch.sum(preds_power)
            sum_y += torch.sum(targets)
            sum_xy += torch.sum(preds_power * targets)
            sum_x2 += torch.sum(preds_power ** 2)
            sum_y2 += torch.sum(targets ** 2)

            total_count += targets.numel()

    if dist.is_initialized():
        dist.all_reduce(sum_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_se, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_ae, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_ape, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_mape_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_x, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_y, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_xy, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_x2, op=dist.ReduceOp.SUM)
        dist.all_reduce(sum_y2, op=dist.ReduceOp.SUM)
        world_size = dist.get_world_size()
    else:
        world_size = 1

    avg_loss = sum_loss.item() / (len(loader) * world_size)
    metrics = {}

    if rank == 0:
        N = total_count.item()
        final_rmse = math.sqrt(sum_se.item() / N)
        final_mae = sum_ae.item() / N
        final_mape = (sum_ape.item() / N) * 100.0

        numerator = N * sum_xy.item() - (sum_x.item() * sum_y.item())
        denominator = math.sqrt(max(N * sum_x2.item() - sum_x.item() ** 2, 1e-8)) * \
                      math.sqrt(max(N * sum_y2.item() - sum_y.item() ** 2, 1e-8))

        final_r = (numerator / denominator) * 100.0 if denominator > 0 else 0.0

        metrics = {
            'RMSE': final_rmse,
            'MAE': final_mae,
            'MAPE(%)': final_mape,
            'R(%)': final_r
        }

    return avg_loss, metrics


# ================= 主函数 =================
def main():
    rank, world_size, local_rank = init_ddp()
    DEVICE = torch.device(f"cuda:{local_rank}")

    logger = None
    if rank == 0:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        logger = setup_logger(SAVE_DIR)
        set_seed(logger, 42)
        mode_str = "自适应权重版" if AUTO_LOSS else "固定权重版"
        if dist.is_initialized():
            logger.info(f"🚀 启动分布式训练 ({mode_str}) | World Size: {world_size} | 设备: {DEVICE}")
        else:
            logger.info(f"🚀 启动单卡直通训练 ({mode_str}) | 设备: {DEVICE}")

    if dist.is_initialized():
        train_dataset = SatellitePVDataset(TRAIN_CSV_PATH, TRAIN_SAT_DIR, mode="train")
        val_dataset = SatellitePVDataset(VAL_CSV_PATH, VAL_SAT_DIR, mode="val")
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
        train_shuffle = False
    else:
        train_dataset = SatellitePVDataset(TRAIN_CSV_PATH, TRAIN_SAT_DIR, mode="train")
        val_dataset = SatellitePVDataset(VAL_CSV_PATH, VAL_SAT_DIR, mode="val")
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=train_shuffle,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, shuffle=False, num_workers=4)

    # 1. 构建主模型并初始化权重
    model = MultiModalPVNet(
        final_dim=FINAL_DIM, transformer_dim=TRANSFORMER_DIM, heads=HEADS,
        self_depth=SELF_DEPTH, cross_depth=CROSS_DEPTH, output_seq_len=4, dropout=DROPOUT
    ).to(DEVICE)
    model.apply(init_weights)

    # 🌟 1.5 尝试加载预训练模型 (微调模式)
    is_finetuning = False
    if PRETRAINED_MODEL_PATH and os.path.exists(PRETRAINED_MODEL_PATH):
        if rank == 0:
            logger.info(f"🔄 正在加载预训练权重进行微调: {PRETRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
        is_finetuning = True
    elif PRETRAINED_MODEL_PATH:
        if rank == 0:
            logger.warning(f"⚠️ 未找到预训练权重文件: {PRETRAINED_MODEL_PATH}，将从头开始训练！")

    loss_weighter = None

    # 🌟 2. 根据开关决定是否构建自适应权重模块
    if AUTO_LOSS:
        loss_weighter = AdaptiveLossWeighter(num_losses=4).to(DEVICE)

    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        # 如果启用了自适应，权重模块也需要 DDP 同步
        if AUTO_LOSS:
            loss_weighter = DDP(loss_weighter, device_ids=[local_rank], output_device=local_rank)

    # 🌟 3. 配置优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 如果开启自适应，将权重模块参数放进优化器
    if AUTO_LOSS:
        optimizer.add_param_group({'params': loss_weighter.parameters(), 'lr': LEARNING_RATE * 5})

    # 🌟 4. 智能选择调度器
    if is_finetuning:
        if rank == 0:
            logger.info("📉 检测到微调模式，使用 CosineAnnealingLR 调度器 (无预热，缓慢衰减)")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS * len(train_loader),  # 按照步数退火
            eta_min=1e-10
        )
    else:
        if rank == 0:
            logger.info("📈 检测到从头训练模式，使用 OneCycleLR 调度器 (含预热)")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=LEARNING_RATE, epochs=NUM_EPOCHS,
            steps_per_epoch=len(train_loader), pct_start=0.1, anneal_strategy='cos'
        )

    dcca_criterion = NRDCCALoss(lambd=1e-3, noise_std=0.05, nr_weight=0.5).to(DEVICE)

    best_rmse = float('inf')
    best_mae = float('inf')
    best_mape = float('inf')
    best_r = -float('inf')
    patience_counter = 0

    train_loss_history = []
    val_loss_history = []
    rmse_hist, mae_hist, mape_hist, r_hist = [], [], [], []

    if rank == 0:
        logger.info("-" * 60)
        logger.info(f"🔥 开始训练 (Epochs: {NUM_EPOCHS})")

    for epoch in range(NUM_EPOCHS):
        if dist.is_initialized():
            train_sampler.set_epoch(epoch)

        # 传入 loss_weighter（关闭自适应时为 None）
        train_loss = train_one_epoch(model, loss_weighter, train_loader, optimizer, DEVICE, scheduler, dcca_criterion,
                                     rank)
        val_loss, val_metrics = validate_distributed(model, loss_weighter, val_loader, DEVICE, dcca_criterion, rank)

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            current_rmse = val_metrics['RMSE']
            current_mae = val_metrics['MAE']
            current_mape = val_metrics['MAPE(%)']
            current_r = val_metrics['R(%)']

            rmse_hist.append(current_rmse)
            mae_hist.append(current_mae)
            mape_hist.append(current_mape)
            r_hist.append(current_r)

            logger.info(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.2e}")
            logger.info(
                f"   => RMSE: {current_rmse:.4f} | MAE: {current_mae:.4f} | MAPE: {current_mape:.2f}% | R: {current_r:.2f}%")

            # 🌟 根据开关决定日志打印内容
            if AUTO_LOSS:
                lw_module = loss_weighter.module if dist.is_initialized() else loss_weighter
                cur_weights = lw_module.get_current_weights()
                logger.info(
                    f"   => 动态权重: [MSE:{cur_weights[0]:.4f}, Grad:{cur_weights[1]:.4f}, Phy:{cur_weights[2]:.4f}, DCCA:{cur_weights[3]:.4f}]")
            else:
                logger.info(
                    f"   => 固定权重: [MSE:{ALPHA:.4f}, Grad:{BETA:.4f}, Phy:{GAMMA:.4f}, DCCA:{LAMBDA_DCCA:.4f}]")

            any_improvement = False
            model_to_save = model.module if dist.is_initialized() else model

            if current_rmse < best_rmse:
                best_rmse = current_rmse
                any_improvement = True
                torch.save(model_to_save.state_dict(),
                           os.path.join(SAVE_DIR,
                                        f"Epoch:{epoch + 1}-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))
            if current_mae < best_mae:
                best_mae = current_mae
                any_improvement = True
                torch.save(model_to_save.state_dict(),
                           os.path.join(SAVE_DIR,
                                        f"Epoch:{epoch + 1}-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))
            if current_mape < best_mape:
                best_mape = current_mape
                any_improvement = True
                torch.save(model_to_save.state_dict(),
                           os.path.join(SAVE_DIR,
                                        f"Epoch:{epoch + 1}-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))
            if current_r > best_r:
                best_r = current_r
                any_improvement = True
                torch.save(model_to_save.state_dict(),
                           os.path.join(SAVE_DIR,
                                        f"Epoch:{epoch + 1}-RMSE:{current_rmse:.4f}-MAE:{current_mae:.4f}-MAPE:{current_mape:.2f}%-R:{current_r:.2f}%.pth"))

            if any_improvement:
                patience_counter = 0
            else:
                patience_counter += 1
                logger.info(f"   ⏳ 所有四项指标均未提升 ({patience_counter}/{PATIENCE})")

            if patience_counter >= PATIENCE:
                early_stop_signal = torch.tensor(1.0).to(DEVICE)
            else:
                early_stop_signal = torch.tensor(0.0).to(DEVICE)
        else:
            early_stop_signal = torch.tensor(0.0).to(DEVICE)

        if dist.is_initialized():
            dist.all_reduce(early_stop_signal, op=dist.ReduceOp.SUM)
        if early_stop_signal.item() > 0:
            if rank == 0: logger.info(f"🛑 触发早停机制，停止于 Epoch {epoch + 1}")
            break

    if rank == 0:
        logger.info("-" * 60)
        logger.info("🎉 训练结束！开始绘制并保存图表...")
        plot_loss_curve(train_loss_history, val_loss_history, os.path.join(SAVE_DIR, "loss_curve.png"), logger)
        plot_metrics_curve(rmse_hist, mae_hist, mape_hist, r_hist, os.path.join(SAVE_DIR, "metrics_curve.png"), logger)
        logger.info(f"📂 所有文件已保存在: {SAVE_DIR}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
