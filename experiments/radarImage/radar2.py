import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 全局字体设置
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 1. 手动录入实验数据
# =========================
data = [
    # Method, Weather, RMSE, MAE, MAPE, R

    # 自己的方法
    ["Ours", "Clear-sky", 0.016621, 0.011793, 0.0204, 0.9928],
    ["Ours", "Cloudy", 0.045979, 0.034498, 0.0756, 0.9455],
    ["Ours", "Rainy", 0.028359, 0.023343, 0.1506, 0.9649],

    # 2025年方法
    ["CR-DCCA", "Clear-sky", 0.019075, 0.015142, 0.0269, 0.9870],
    ["CR-DCCA", "Cloudy", 0.054665, 0.037993, 0.0793, 0.9421],
    ["CR-DCCA", "Rainy", 0.053333, 0.042538, 0.2156, 0.9055],

    # 2024年方法
    ["ROI-Hybrid", "Clear-sky", 0.018993, 0.010883, 0.0198, 0.9894],
    ["ROI-Hybrid", "Cloudy", 0.065007, 0.046798, 0.0931, 0.9268],
    ["ROI-Hybrid", "Rainy", 0.053688, 0.043748, 0.2595, 0.8067],

    # 2023年方法
    ["IV-CMA", "Clear-sky", 0.026132, 0.017910, 0.0357, 0.9897],
    ["IV-CMA", "Cloudy", 0.054705, 0.036286, 0.0777, 0.9429],
    ["IV-CMA", "Rainy", 0.044236, 0.033700, 0.2090, 0.9521],

    # 2022年方法
    ["STUNet", "Clear-sky", 0.016397, 0.012172, 0.0209, 0.9919],
    ["STUNet", "Cloudy", 0.071146, 0.049082, 0.1048, 0.9070],
    ["STUNet", "Rainy", 0.057354, 0.045935, 0.2516, 0.9294],
]

df = pd.DataFrame(data, columns=["Method", "Weather", "RMSE", "MAE", "MAPE", "R"])

# =========================
# 2. 指标归一化
# RMSE、MAE、MAPE 越小越好，反向归一化
# R 越大越好，正向归一化
# =========================
metrics_neg = ["RMSE", "MAE", "MAPE"]
metrics_pos = ["R"]

df_score = df.copy()

for weather in df["Weather"].unique():
    idx = df["Weather"] == weather

    # 误差类指标：越小越好
    for m in metrics_neg:
        x = df.loc[idx, m]
        if x.max() == x.min():
            df_score.loc[idx, m + "_score"] = 1.0
        else:
            df_score.loc[idx, m + "_score"] = (x.max() - x) / (x.max() - x.min())

    # 相关性指标：越大越好
    for m in metrics_pos:
        x = df.loc[idx, m]
        if x.max() == x.min():
            df_score.loc[idx, m + "_score"] = 1.0
        else:
            df_score.loc[idx, m + "_score"] = (x - x.min()) / (x.max() - x.min())

# =========================
# 3. 加权综合得分
# 更强调 RMSE 和 R
# =========================
df_score["TotalScore"] = (
    0.35 * df_score["RMSE_score"] +
    0.20 * df_score["MAE_score"] +
    0.20 * df_score["MAPE_score"] +
    0.25 * df_score["R_score"]
)

# 可选：打印综合得分，便于检查
print(
    df_score[["Method", "Weather", "TotalScore"]]
    .sort_values(["Weather", "TotalScore"], ascending=[True, False])
)

# =========================
# 4. 整理雷达图数据
# =========================
radar_data = df_score.pivot(index="Method", columns="Weather", values="TotalScore")

labels = ["Clear-sky", "Cloudy", "Rainy"]
radar_data = radar_data[labels]

# 按图例顺序排列
method_order = [
    "STUNet",
    "IV-CMA",
    "ROI-Hybrid",
    "CR-DCCA",
    "Ours"
]
radar_data = radar_data.loc[method_order]

# =========================
# 5. 绘制雷达图
# =========================
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(8.5, 7.5))
ax = plt.subplot(111, polar=True)

for method in radar_data.index:
    values = radar_data.loc[method].tolist()
    values += values[:1]

    if method == "Ours":
        ax.plot(
            angles,
            values,
            linewidth=3.5,
            linestyle="-",
            label=method
        )
        ax.fill(angles, values, alpha=0.18)
    else:
        ax.plot(
            angles,
            values,
            linewidth=1.8,
            linestyle="--",
            label=method
        )
        ax.fill(angles, values, alpha=0.05)

# =========================
# 6. 坐标轴设置
# =========================
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=13)

# 外圈标签向外偏移，避免被折线挡住
ax.tick_params(axis="x", pad=14)

ax.set_ylim(0, 1.05)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)

# 把径向刻度标签挪到右上侧，避免和顶部标签重叠
ax.set_rlabel_position(22.5)

# 不设置标题，论文里用图注说明即可
# ax.set_title("", fontsize=14, pad=22)

# 图例放到底部，分成两列
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.02),
    ncol=2,
    fontsize=10,
    frameon=True
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)

# =========================
# 7. 保存图像
# =========================
save_dir = "/data0/ybf/lee/workspace/OnePaper/experiments/radarImage"
os.makedirs(save_dir, exist_ok=True)

png_path = os.path.join(save_dir, "weather_weighted_radar1.png")
pdf_path = os.path.join(save_dir, "weather_weighted_radar2.pdf")
svg_path = os.path.join(save_dir, "weather_weighted_radar1.svg")

plt.savefig(png_path, dpi=900, bbox_inches="tight", pad_inches=0.05)
plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
plt.savefig(svg_path, bbox_inches="tight", pad_inches=0.05)

print(f"PNG saved to: {png_path}")
print(f"PDF saved to: {pdf_path}")
print(f"SVG saved to: {svg_path}")

plt.show()