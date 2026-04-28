import pandas as pd
import numpy as np
import os


def extract_scenario_validation_sets(csv_path, output_prefix, target_col='CSI', window=3):
    """
    自动扫描全年数据，提取三种典型场景的连续多天验证集。
    包含核心防御机制：
    🌟 零容忍原则：只要该天 24 小时内出现任何一个 NaN，或者数据行数缺失，直接抛弃该天，重新寻找！
    """
    print(f"📂 正在读取全年数据进行场景扫描: {csv_path} ...")
    df = pd.read_csv(csv_path, parse_dates=True, index_col=0).sort_index()

    # 1. 过滤出白天的数据用于计算波动特征
    if 'Solar_Zenith' in df.columns:
        daylight_df = df[df['Solar_Zenith'] < 85]
    else:
        daylight_df = df.between_time('08:00', '17:00')

    # =========================================================
    # 核心函数 1：计算单日波动率
    # =========================================================
    def calculate_volatility(series):
        clean_series = series.dropna()
        if len(clean_series) < 30:
            return np.nan
        return clean_series.std() + clean_series.diff().abs().mean()

    # =========================================================
    # 核心函数 2：计算单日平均发电水平
    # =========================================================
    def calculate_mean_level(series):
        clean_series = series.dropna()
        if len(clean_series) < 30:
            return np.nan
        return clean_series.mean()

    print("🧮 正在计算每日波动率与平均发电水平...")
    daily_scores = daylight_df.groupby(daylight_df.index.date)[target_col].apply(calculate_volatility)
    daily_mean_csi = daylight_df.groupby(daylight_df.index.date)[target_col].apply(calculate_mean_level)

    # =========================================================
    # 🌟 终极强硬防线：全天（24小时）零容忍扫描！
    # =========================================================
    def is_perfect_day(series):
        # 1. 全天 24 小时绝对不能有任何 NaN
        # 2. 假设 15 分钟一针，一天应该是 96 个点。这里设定至少要有 90 个点才算完整天。
        return (not series.isna().any()) and (len(series) >= 90)

    # 注意：这里是用原始的全天 df 进行 groupby，不仅仅是白天
    daily_perfect_power = df.groupby(df.index.date)['Active_Power'].apply(is_perfect_day)

    # 统一转换索引，保证日历连续性
    daily_scores.index = pd.to_datetime(daily_scores.index)
    daily_mean_csi.index = pd.to_datetime(daily_mean_csi.index)
    daily_perfect_power.index = pd.to_datetime(daily_perfect_power.index)

    daily_scores = daily_scores.asfreq('D')
    daily_mean_csi = daily_mean_csi.asfreq('D')
    daily_perfect_power = daily_perfect_power.asfreq('D', fill_value=False)

    # 🌟 核心制裁：如果这天不是完美的一天（有 NaN 或点数不够），直接将它的得分抹除为 NaN！
    daily_scores = daily_scores.where(daily_perfect_power, np.nan)

    print(f"🪟 正在应用 {window} 天滑动窗口 (遇残缺数据自动跳过)...")
    # min_periods=window 确保：只要这连续几天里混进了一个残缺天，整个窗口立刻作废！
    rolling_mean_vol = daily_scores.rolling(window=window, min_periods=window).mean()
    rolling_std_vol = daily_scores.rolling(window=window, min_periods=window).std()
    rolling_mean_csi = daily_mean_csi.rolling(window=window, min_periods=window).mean()

    # 清理所有无效的滑动窗口数据
    valid_rolling_mean = rolling_mean_vol.dropna()
    valid_rolling_std = rolling_std_vol.dropna()
    valid_rolling_csi = rolling_mean_csi.dropna()

    # ==========================================
    # 🌟 场景 A：连续晴朗
    # ==========================================
    valid_clear_candidates = valid_rolling_mean[valid_rolling_csi > 0.4]

    if valid_clear_candidates.empty:
        print("❌ 找不到符合要求的连续晴朗天气！(可能数据集中完美的连续大晴天太少)")
        clear_dates = []
    else:
        clear_end_date = valid_clear_candidates.idxmin()
        clear_dates = pd.date_range(end=clear_end_date, periods=window)

    # ==========================================
    # 🌟 场景 B：连续突变
    # ==========================================
    if valid_rolling_mean.empty:
        print("❌ 找不到符合要求的连续突变天气！")
        cloudy_dates = []
    else:
        cloudy_end_date = valid_rolling_mean.idxmax()
        cloudy_dates = pd.date_range(end=cloudy_end_date, periods=window)

    # ==========================================
    # 🌟 场景 C：混合天气
    # ==========================================
    if valid_rolling_std.empty:
        print("❌ 找不到符合要求的混合天气！")
        mixed_dates = []
    else:
        mixed_end_date = valid_rolling_std.idxmax()
        mixed_dates = pd.date_range(end=mixed_end_date, periods=window)

    # =============== 提取并分别保存数据 ===============
    scenarios = {
        'clear': clear_dates,
        'ramp': cloudy_dates,
        'mixed': mixed_dates
    }

    for name, dates in scenarios.items():
        if len(dates) == 0:
            continue

        print(f"\n✅ 提取场景: [{name.upper()}]")
        print(f"   📅 入选日期范围: {dates[0].date()} 至 {dates[-1].date()}")

        sampled_dfs = []
        for d in dates:
            date_str = d.strftime('%Y-%m-%d')
            if date_str in df.index:
                # 🌟 直接提取整天数据 (从 00:00 开始)
                # 不再做任何 dropna，因为前面的逻辑已经数学担保了这天绝对没有 NaN！
                sampled_dfs.append(df.loc[date_str])

        if sampled_dfs:
            scenario_df = pd.concat(sampled_dfs)
            out_path = f"{output_prefix}_{name}_weather_{window}days.csv"
            scenario_df.to_csv(out_path)
            print(f"   💾 已保存至: {out_path} (行数: {len(scenario_df)})")

    return


if __name__ == '__main__':
    INPUT_CSV = "../data/val/series_file_test.csv"
    OUTPUT_PREFIX = "../data/val/hardcore"
    DAYS_WINDOW = 2

    if os.path.exists(INPUT_CSV):
        extract_scenario_validation_sets(INPUT_CSV, OUTPUT_PREFIX, target_col='CSI', window=DAYS_WINDOW)
    else:
        print(f"❌ 找不到输入文件，请检查路径: {INPUT_CSV}")