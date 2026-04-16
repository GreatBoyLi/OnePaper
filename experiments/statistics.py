import pandas as pd


def calculate_cloud_distribution(df):
    """
    根据 CSI (Clear Sky Index) 计算云层状况的分布。
    """
    # 过滤掉夜间或无效数据 (CSI == 0 的情况)
    # 你也可以根据 df['Clear_Sky_GHI'] > 0 或天顶角来过滤白天数据
    df_valid = df[df['CSI'] > 0].copy()

    # 按照表格中的条件进行分类统计
    clear_sky_count = len(df_valid[df_valid['CSI'] > 0.9])
    cloudy_count = len(df_valid[(df_valid['CSI'] >= 0.3) & (df_valid['CSI'] <= 0.9)])
    overcast_count = len(df_valid[(df_valid['CSI'] > 0) & (df_valid['CSI'] < 0.3)])

    # 计算总数
    total = clear_sky_count + cloudy_count + overcast_count

    return [clear_sky_count, cloudy_count, overcast_count, total]


# 1. 加载你的训练集和测试集 (这里假设你已经分好了 csv 文件)
# 如果你是在一个 DataFrame 里通过时间划分的，可以先对 DataFrame 进行切片
df_train = pd.read_csv("../data/train/series_file.csv")
df_test = pd.read_csv("../data/val/series_file.csv")

# 2. 获取统计数据
train_counts = calculate_cloud_distribution(df_train)
test_counts = calculate_cloud_distribution(df_test)

# 3. 构建与图像结构一致的输出表格
distribution_table = pd.DataFrame({
    "Cloud Condition": [
        "Clear-sky (CSI > 0.9)",
        "Cloudy (0.3 <= CSI <= 0.9)",
        "Overcast (0 < CSI < 0.3)",
        "Total"
    ],
    "Number of Train": train_counts,
    "Number of Test": test_counts
})

# 打印最终结果
print("TABLE II")
print("DISTRIBUTION OF CLOUD CONDITION TYPES IN THE DATASET")
print("-" * 65)
print(distribution_table.to_string(index=False))
print("-" * 65)