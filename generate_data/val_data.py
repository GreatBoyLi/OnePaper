import pandas as pd
import numpy as np
import os


def full_28_day(path, output_path):
    # 1. 读取你原本完整的 2025 年验证集
    df_val_full = pd.read_csv(path, parse_dates=True, index_col=0)

    # 2. 定义我们选好的四个典型时间段
    target_periods = [
        ('2024-01-15', '2024-01-21'),
        ('2024-04-15', '2024-04-21'),
        ('2024-07-15', '2024-07-21'),
        ('2024-10-15', '2024-10-21')
    ]

    # 3. 把这四个时间段的数据拼接起来
    sampled_dfs = []
    for start, end in target_periods:
        # 包含该日期的全天数据 (从 00:00 到 23:59)
        period_df = df_val_full.loc[start:end]
        sampled_dfs.append(period_df)

    df_val_sampled = pd.concat(sampled_dfs)

    # 4. 保存为精简版验证集
    df_val_sampled.to_csv(output_path)
    print(f"✅ 精简版验证集生成完毕！总行数: {len(df_val_sampled)}")


def generate_hardcore_validation_set(csv_path, output_path, target_col='CSI', top_k=10):
    """
    扫描全年数据，自动挑选出白天波动最剧烈的 K 天作为验证集。
    """
    print(f"📂 正在读取全年数据: {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0)

    # 确保时间索引是有序的
    df = df.sort_index()

    # 1. 过滤出“白天”的数据进行波动率计算
    # 晚上 CSI 是 0，没有任何波动，如果不剔除晚上，会严重稀释白天的方差
    if 'Solar_Zenith' in df.columns:
        # 使用天顶角 < 85 度作为白天的严格定义
        daylight_df = df[df['Solar_Zenith'] < 85]
    else:
        # 备用方案：通过固定时间段过滤（比如 Alice Springs 的早 8 点到晚 5 点）
        daylight_df = df.between_time('08:00', '17:00')

    # 2. 按天分组
    daily_groups = daylight_df.groupby(daylight_df.index.date)

    # 3. 计算每一天的波动得分 (Volatility Score)
    # 综合指标：标准差 (整体离散度) + 一阶差分绝对值总和 (锯齿状波动的剧烈程度)
    def calculate_volatility(series):
        if len(series) < 10:  # 排除数据缺失严重的无效天数
            return 0.0

        # 归一化 CSI 的标准差
        std_val = series.std()
        # 归一化 一阶差分绝对值求和 (代表爬坡/突降的频率)
        ramp_val = series.diff().abs().mean()

        # 综合得分 (你可以根据需要调整权重)
        return std_val + ramp_val

    print("🧮 正在计算每日波动率得分...")
    daily_scores = daily_groups[target_col].apply(calculate_volatility)

    # 4. 排序并取出得分最高的 top_k 天
    hardest_dates = daily_scores.sort_values(ascending=False).head(top_k).index

    print(f"\n🔥 筛选出的 {top_k} 个最具挑战性的日期及其波动得分:")
    for rank, date in enumerate(hardest_dates):
        print(f"   Top {rank + 1}: {date} (得分: {daily_scores[date]:.4f})")

    # 5. 提取这 top_k 天的【全天完整数据】(从 00:00 到 23:59)
    # 注意：虽然我们只用白天计算波动，但导出验证集时必须带上黑夜，
    # 否则你的 Dataset 里面的序列滑动窗口 (16+4) 会因为时间断层而报错！
    sampled_dfs = []
    for d in hardest_dates:
        date_str = d.strftime('%Y-%m-%d')
        # 获取这一整天的数据
        full_day_data = df.loc[date_str]
        sampled_dfs.append(full_day_data)

    # 合并并保存
    hardcore_val_df = pd.concat(sampled_dfs)
    hardcore_val_df.to_csv(output_path)

    print(f"\n✅ 困难样本提取完成！已保存至: {output_path}")
    print(f"   共计 {len(hardcore_val_df)} 个时间步。")

    return


def merge_validation_sets(file1, file2, output_file):
    if not os.path.exists(file1) or not os.path.exists(file2):
        print("❌ 找不到输入文件，请检查当前目录下是否存在这两个 CSV 文件。")
        return

    print("📂 正在读取文件...")
    # 读取文件，将第一列作为时间索引，并解析为日期格式
    df1 = pd.read_csv(file1, index_col=0, parse_dates=True)
    df2 = pd.read_csv(file2, index_col=0, parse_dates=True)

    print(f"   - 季节代表集 (28天) 包含 {len(df1)} 行")
    print(f"   - 困难样本集 (10天) 包含 {len(df2)} 行")

    print("\n🔗 正在合并数据...")
    # 纵向拼接两个数据集
    df_merged = pd.concat([df1, df2])

    print("🧹 正在去重并按时间排序...")
    # 1. 剔除完全重合的时间点（如果 10 天里有某天刚好在 28 天的范围内，保留一份即可）
    # ~ 表示取反，保留那些没有重复的索引，或者重复索引中第一次出现的记录
    df_merged = df_merged[~df_merged.index.duplicated(keep='first')]

    # 2. 按时间索引从小到大严格排序 (Ascending=True)
    df_merged = df_merged.sort_index(ascending=True)

    print("\n💾 正在保存最终验证集...")
    df_merged.to_csv(output_file)

    print(f"🎉 搞定！最终的验证集已保存为: {output_file}")
    print(f"   最终有效总行数: {len(df_merged)}")


if __name__ == "__main__":
    # 配置你的文件路径
    INPUT_CSV = "../data/val/test_series_file.csv"
    OUTPUT_CSV1 = "../data/val_test_28days_sampled.csv"
    OUTPUT_CSV2 = "../data/val_test_10days_hardcore.csv"  # 输出的高难度验证集
    OUTPUT_CSV3 = "../data/val/series_file_test.csv"
    full_28_day(INPUT_CSV, OUTPUT_CSV1)

    # 确保文件存在再运行
    if os.path.exists(INPUT_CSV):
        generate_hardcore_validation_set(INPUT_CSV, OUTPUT_CSV2, target_col='CSI', top_k=10)
    else:
        print(f"❌ 找不到输入文件: {INPUT_CSV}")

    merge_validation_sets(OUTPUT_CSV1, OUTPUT_CSV2, OUTPUT_CSV3)
