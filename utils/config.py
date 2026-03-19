import os
import yaml
import logging
import datetime
import matplotlib.pyplot as plt  # 【新增 1】导入绘图库


def load_config(config_path):
    """加载YAML配置文件"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ 配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # safe_load避免安全风险
    return config


def setup_logger(save_dir):
    # 确保文件夹存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 生成类似 train_20260312_103000.log 的文件名
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(save_dir, f"train_{current_time}.log")

    # 配置 logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',  # 自动加上时间戳
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 写入文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)


def plot_metrics_curve(rmse_hist, mae_hist, mape_hist, r_hist, save_path, logger):
    """
    在一个大图里画出四个指标的独立变化曲线
    """
    # 创建一个 2 行 2 列的画布，尺寸为 12x8
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Validation Metrics Over Epochs', fontsize=16)

    # 1. 绘制 RMSE (越小越好)
    axs[0, 0].plot(rmse_hist, label='RMSE', color='blue', linewidth=2)
    axs[0, 0].set_title('RMSE (Lower is Better)')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('RMSE')
    axs[0, 0].grid(True, linestyle='--', linewidth=0.5)

    # 2. 绘制 MAE (越小越好)
    axs[0, 1].plot(mae_hist, label='MAE', color='green', linewidth=2)
    axs[0, 1].set_title('MAE (Lower is Better)')
    axs[0, 1].set_xlabel('Epochs')
    axs[0, 1].set_ylabel('MAE')
    axs[0, 1].grid(True, linestyle='--', linewidth=0.5)

    # 3. 绘制 MAPE (越小越好)
    axs[1, 0].plot(mape_hist, label='MAPE', color='orange', linewidth=2)
    axs[1, 0].set_title('MAPE % (Lower is Better)')
    axs[1, 0].set_xlabel('Epochs')
    axs[1, 0].set_ylabel('MAPE (%)')
    axs[1, 0].grid(True, linestyle='--', linewidth=0.5)

    # 4. 绘制 R (越大越好)
    axs[1, 1].plot(r_hist, label='R', color='red', linewidth=2)
    axs[1, 1].set_title('R % (Higher is Better)')
    axs[1, 1].set_xlabel('Epochs')
    axs[1, 1].set_ylabel('R (%)')
    axs[1, 1].grid(True, linestyle='--', linewidth=0.5)

    # 调整布局防止重叠，并保存图片
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭画布释放显存
    logger.info(f"📈 四项指标变化曲线图已保存至: {save_path}")


# 【新增 2】绘图函数
def plot_loss_curve(train_losses, val_losses, save_path, logger):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linestyle='--', linewidth=2)

    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # 保存图片
    plt.savefig(save_path, dpi=300)
    plt.close()  # 关闭画布释放内存
    logger.info(f"📈 损失曲线图已保存至: {save_path}")
