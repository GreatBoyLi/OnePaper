import numpy as np
import matplotlib.pyplot as plt


def validate_satellite_npy(npy_path):
    # 1. 加载数据
    data = np.load(npy_path)

    # 2. 基础属性打印
    print(f"📊 正在检查文件: {npy_path}")
    print(f"  ➤ 数组维度 (Shape): {data.shape}")
    print(f"  ➤ 数据类型 (Dtype): {data.dtype}")
    print(f"  ➤ 包含 NaN 数量: {np.isnan(data).sum()}")
    print(f"  ➤ 包含 Inf 数量: {np.isinf(data).sum()}\n")

    # 定义通道名称
    channels = ['albedo_03 (Visible)', 'tbb_07 (Shortwave IR)', 'tbb_13 (Longwave IR)']

    # 3. 统计信息打印
    for i, name in enumerate(channels):
        channel_data = data[i]
        print(f"🔹 通道 {i} - {name}:")
        print(f"    Min: {np.nanmin(channel_data):.2f}")
        print(f"    Max: {np.nanmax(channel_data):.2f}")
        print(f"    Mean: {np.nanmean(channel_data):.2f}")

    # 4. 绘图可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # albedo_03 通常在 0-1 之间
    im0 = axes[0].imshow(data[0], cmap='gray', vmin=0, vmax=1)
    axes[0].set_title("Ch0: albedo_03")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # tbb_07 和 tbb_13 通常在 200K - 340K 之间
    im1 = axes[1].imshow(data[1], cmap='jet', vmin=220, vmax=340)
    axes[1].set_title("Ch1: tbb_07")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(data[2], cmap='jet', vmin=220, vmax=340)
    axes[2].set_title("Ch2: tbb_13")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # 替换成你刚刚生成的一个【白天】的实测文件路径
    sample_file = "../data/train/crop_himawari/202001/01/NC_H08_20200101_0430_R21_FLDK.02401_02401_crop.npy"
    validate_satellite_npy(sample_file)