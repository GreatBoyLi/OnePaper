import pandas as pd
import pvlib
import os
from utils.config import load_config


# ===========================================

def process_timeseries(config):
    CSV_PATH = config["file_paths"]["pv_file"]
    OUTPUT_PATH = config["file_paths"]["series_file"]

    START_DATE = config["dates"]["start_date"]
    END_DATE = config["dates"]["end_date"]

    LATITUDE = config["stations"]["lat"]
    LONGITUDE = config["stations"]["lon"]
    ALTITUDE = config["stations"]["altitude"]
    CAPACITY = config["stations"]["capacity"]

    print(f"🚀 开始处理时间序列数据: {CSV_PATH}")

    # 1. 读取数据
    try:
        df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'], index_col='timestamp')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 2. 筛选时间范围
    mask = (df.index >= START_DATE) & (df.index <= f"{END_DATE} 23:59:59")
    df = df.loc[mask]

    if df.empty:
        print("⚠️ 警告：筛选后数据为空，请检查 CSV 中的时间列是否正确！")
        return
    print(f"   筛选后数据量 (5min): {len(df)} 条")

    # ========================================================
    # 🌟 修改点 1：在这里把需要的新列加到列表中
    # ========================================================
    target_columns = [
        'Active_Power',
        'Global_Horizontal_Radiation',
        'Weather_Temperature_Celsius',
        'Weather_Relative_Humidity'
    ]

    # 提取多列并一起进行 15 分钟重采样求平均
    df_15min = df[target_columns].resample('15min').mean()

    # 线性插值填充少量缺失值 (同时作用于这 4 列)
    df_15min = df_15min.interpolate(method='linear', limit=4)
    print(f"   重采样后数据量 (15min): {len(df_15min)} 条")

    # 3. 赋予时区信息
    try:
        if df_15min.index.tz is None:
            df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT',
                                                        nonexistent='shift_forward')
        else:
            df_15min.index = df_15min.index.tz_convert('Australia/Darwin')
    except Exception as e:
        print(f"⚠️ 时区转换警告: {e}")
        df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT')

    print("   ✅ 已修正为爱丽丝泉当地时间 (ACST)")

    # 4. 计算天文学特征 (Zenith & Clear-sky GHI)
    print("   正在计算太阳天顶角和晴空辐照度...")
    location = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE, tz='Australia/Darwin')
    times = df_15min.index

    solpos = location.get_solarposition(times)
    df_15min['Solar_Zenith'] = solpos['zenith'].values

    cs = location.get_clearsky(times, model='ineichen')
    df_15min['Clear_Sky_GHI'] = cs['ghi'].values

    # 5. 保留所有数据 (不单独剔除夜间)
    print(f"   清洗前: {len(df_15min)}")
    df_clean = df_15min

    # 6. 归一化与数据防爆处理
    df_clean['Power_Norm'] = df_clean['Active_Power'] / CAPACITY
    df_clean['Power_Norm'] = df_clean['Power_Norm'].clip(lower=0)
    df_clean['Clear_Sky_GHI'] = df_clean['Clear_Sky_GHI'].clip(lower=0)

    # 🌟 可选：对实测辐射也做一个基础的负值修正（夜间传感器可能出现微小负噪点）
    df_clean['Global_Horizontal_Radiation'] = df_clean['Global_Horizontal_Radiation'].clip(lower=0)

    print(f"   清洗后: {len(df_clean)}")
    print(f"   ✅ 最终特征列: {list(df_clean.columns)}")

    # 7. 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH)
    print(f"💾 处理完成！文件已保存至: {OUTPUT_PATH}")

    # ========================================================
    # 🌟 修改点 2：在打印预览时，展示所有核心列
    # ========================================================
    print("\n数据预览:")
    preview_cols = [
        'Power_Norm',
        'Global_Horizontal_Radiation',
        'Weather_Temperature_Celsius',
        'Weather_Relative_Humidity',
        'Solar_Zenith'
    ]
    print(df_clean[preview_cols].head())


if __name__ == "__main__":
    config = load_config("../config/config.yaml")
    process_timeseries(config)