import pandas as pd
import pvlib
import os
import json  # 🌟 新增：用于保存和加载归一化参数
from utils.config import load_config


# ===========================================

def process_timeseries(config, is_train=True):
    if is_train:
        CSV_PATH = config["train_file_paths"]["pv_file"]
        OUTPUT_PATH = config["train_file_paths"]["series_file"]
        START_DATE = config["train_dates"]["start_date"]
        END_DATE = config["train_dates"]["end_date"]
        LATITUDE = config["train_stations"]["lat"]
        LONGITUDE = config["train_stations"]["lon"]
        ALTITUDE = config["train_stations"]["altitude"]
        CAPACITY = config["train_stations"]["capacity"]
    else:
        CSV_PATH = config["val_file_paths"]["pv_file"]
        OUTPUT_PATH = config["val_file_paths"]["series_file"]
        START_DATE = config["val_dates"]["start_date"]
        END_DATE = config["val_dates"]["end_date"]
        LATITUDE = config["val_stations"]["lat"]
        LONGITUDE = config["val_stations"]["lon"]
        ALTITUDE = config["val_stations"]["altitude"]
        CAPACITY = config["val_stations"]["capacity"]

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
    target_columns = [
        'Active_Power',
        'Global_Horizontal_Radiation',
        'Weather_Temperature_Celsius',
        'Weather_Relative_Humidity'
    ]

    # 提取多列并一起进行 15 分钟重采样求平均
    df_15min = df[target_columns].resample('15min').mean()

    # 线性插值填充少量缺失值
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

    # 5. 保留所有数据
    df_clean = df_15min

    # 6. 归一化与数据防爆处理 (目标值 Power_Norm 独立归一化)
    df_clean['Power_Norm'] = df_clean['Active_Power'] / CAPACITY
    df_clean['Power_Norm'] = df_clean['Power_Norm'].clip(lower=0)
    df_clean['Clear_Sky_GHI'] = df_clean['Clear_Sky_GHI'].clip(lower=0)
    df_clean['Global_Horizontal_Radiation'] = df_clean['Global_Horizontal_Radiation'].clip(lower=0)

    # ========================================================
    # 🌟 核心修改：分离式特征归一化 (防止数据穿越)
    # ========================================================
    columns_to_normalize = [
        'Global_Horizontal_Radiation',
        'Weather_Temperature_Celsius',
        'Weather_Relative_Humidity',
        'Clear_Sky_GHI',
        'Solar_Zenith'
    ]

    # 定义归一化参数 JSON 文件的保存路径 (保存在训练集输出的同级目录)
    # 假设你的配置文件里写了绝对或相对路径，将其保存在 config 指定的地方最安全
    scaler_path = config["scaler_params_paths"]

    if is_train:
        print("   正在提取【训练集】归一化参数并进行归一化...")
        scaler_params = {}
        for col in columns_to_normalize:
            col_min = df_clean[col].min()
            col_max = df_clean[col].max()

            # 将 numpy 数值转换为 Python 原生 float 以便被 json 序列化
            scaler_params[col] = {'min': float(col_min), 'max': float(col_max)}

            if col_max - col_min == 0:
                df_clean[col] = 0.0
            else:
                df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)

        # 保存这把“尺子”
        with open(scaler_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        print(f"   ✅ 归一化参数已保存至: {scaler_path}")

    else:
        print("   正在加载【训练集】归一化参数对【验证/测试集】进行归一化...")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ 找不到归一化参数文件 {scaler_path}，请先以 is_train=True 运行一次以生成尺子！")

        # 读取这把“尺子”
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        for col in columns_to_normalize:
            col_min = scaler_params[col]['min']
            col_max = scaler_params[col]['max']

            if col_max - col_min == 0:
                df_clean[col] = 0.0
            else:
                df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)
        print("   ✅ 验证/测试集归一化完成 (严格遵循训练集尺度，零穿越)")
    # ========================================================

    print(f"   清洗后: {len(df_clean)}")
    print(f"   ✅ 最终特征列: {list(df_clean.columns)}")

    # 7. 保存结果
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH)
    print(f"💾 处理完成！文件已保存至: {OUTPUT_PATH}")

    # 打印预览
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

    # 1. 先跑训练集，生成数据并固化归一化参数 (生成 scaler_params.json)
    print("========== 阶段 1: 处理训练集 ==========")
    process_timeseries(config, is_train=True)

    # 2. 再跑验证/测试集，加载 JSON 参数对新数据进行严格按比例缩放
    print("\n========== 阶段 2: 处理验证/测试集 ==========")
    process_timeseries(config, is_train=False)
