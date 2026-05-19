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
    # Method, Season, RMSE, MAE, MAPE, R

    # 自己的方法
    ["Ours", "Summer", 0.061694, 0.031947, 0.2733, 0.9710],
    ["Ours", "Autumn", 0.041214, 0.015726, 0.1692, 0.9906],
    ["Ours", "Winter", 0.045403, 0.018576, 0.2385, 0.9858],
    ["Ours", "Spring", 0.086411, 0.036702, 0.7692, 0.9418],
    ["Ours", "Total", 0.062531, 0.025803, 0.3866, 0.9788],

    # 2025年方法
    ["ConvLSTM-RICNN+DCCA-LF", "Summer", 0.065672, 0.033244, 0.4108, 0.9656],
    ["ConvLSTM-RICNN+DCCA-LF", "Autumn", 0.045169, 0.017152, 0.2519, 0.9881],
    ["ConvLSTM-RICNN+DCCA-LF", "Winter", 0.049129, 0.017916, 0.2840, 0.9825],
    ["ConvLSTM-RICNN+DCCA-LF", "Spring", 0.092428, 0.037129, 0.7807, 0.9311],
    ["ConvLSTM-RICNN+DCCA-LF", "Total", 0.065925, 0.026436, 0.4459, 0.9688],

    # 2024年方法
    ["ROI-ROIsurr Hybrid", "Summer", 0.065362, 0.031821, 0.3212, 0.9657],
    ["ROI-ROIsurr Hybrid", "Autumn", 0.041230, 0.014757, 0.2319, 0.9902],
    ["ROI-ROIsurr Hybrid", "Winter", 0.048295, 0.016698, 0.2718, 0.9828],
    ["ROI-ROIsurr Hybrid", "Spring", 0.089900, 0.034991, 0.7431, 0.9367],
    ["ROI-ROIsurr Hybrid", "Total", 0.064158, 0.024647, 0.4033, 0.9702],

    # 2023年方法
    ["Informer-ViT-CMA", "Summer", 0.066924, 0.033021, 0.4255, 0.9650],
    ["Informer-ViT-CMA", "Autumn", 0.043372, 0.015829, 0.2380, 0.9890],
    ["Informer-ViT-CMA", "Winter", 0.049590, 0.017771, 0.3185, 0.9818],
    ["Informer-ViT-CMA", "Spring", 0.095255, 0.038897, 0.8206, 0.9320],
    ["Informer-ViT-CMA", "Total", 0.067040, 0.026466, 0.4652, 0.9681],

    # 2022年方法
    ["STUNet", "Summer", 0.073177, 0.035679, 0.3269, 0.9569],
    ["STUNet", "Autumn", 0.047533, 0.016286, 0.1935, 0.9868],
    ["STUNet", "Winter", 0.046663, 0.014579, 0.2308, 0.9839],
    ["STUNet", "Spring", 0.102221, 0.038636, 0.8002, 0.9196],
    ["STUNet", "Total", 0.071296, 0.026377, 0.4024, 0.9634],
]

df = pd.DataFrame(data, columns=["Method", "Season", "RMSE", "MAE", "MAPE", "R"])

# =========================
# 2. 指标归一化
# RMSE、MAE、MAPE 越小越好，反向归一化
# R 越大越好，正向归一化
# =========================
metrics_neg = ["RMSE", "MAE", "MAPE"]
metrics_pos = ["R"]

df_score = df.copy()

for season in df["Season"].unique():
    idx = df["Season"] == season

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
print(df_score[["Method", "Season", "TotalScore"]].sort_values(["Season", "TotalScore"], ascending=[True, False]))

# =========================
# 4. 整理雷达图数据
# =========================
radar_data = df_score.pivot(index="Method", columns="Season", values="TotalScore")

labels = ["Spring", "Summer", "Autumn", "Winter", "Total"]
radar_data = radar_data[labels]

# 按图例顺序排列
method_order = [
    "STUNet",
    "Informer-ViT-CMA",
    "ROI-ROIsurr Hybrid",
    "ConvLSTM-RICNN+DCCA-LF",
    "Ours"
]
radar_data = radar_data.loc[method_order]

# =========================
# 5. 绘制雷达图
# =========================
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig = plt.figure(figsize=(9, 8))
ax = plt.subplot(111, polar=True)

for method in radar_data.index:
    values = radar_data.loc[method].tolist()
    values += values[:1]

    if method == "Ours":
        ax.plot(angles, values, linewidth=3.5, label=method)
        ax.fill(angles, values, alpha=0.18)
    else:
        ax.plot(angles, values, linewidth=1.8, label=method)
        ax.fill(angles, values, alpha=0.05)

# 坐标轴设置
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=12)

ax.set_ylim(0, 1)
ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=10)

# 可选标题；论文里通常可以不放英文标题，直接用图注说明
ax.set_title("", fontsize=14, pad=22)

# 图例放到底部，分成两列
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=2,
    fontsize=10,
    frameon=True
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.20)

# =========================
# 6. 保存图像
# =========================
save_dir = "/home/ubuntu/workspace/OnePaper/experiments/radarImage"
os.makedirs(save_dir, exist_ok=True)

png_path = os.path.join(save_dir, "seasonal_weighted_radar.png")
pdf_path = os.path.join(save_dir, "seasonal_weighted_radar.pdf")
svg_path = os.path.join(save_dir, "seasonal_weighted_radar.svg")

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(svg_path, bbox_inches="tight")

print(f"PNG saved to: {png_path}")
print(f"PDF saved to: {pdf_path}")
print(f"SVG saved to: {svg_path}")

plt.show()
