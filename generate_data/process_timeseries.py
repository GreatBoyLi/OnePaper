import pandas as pd
import pvlib
import os
import json
import numpy as np
from utils.config import load_config


def process_timeseries(config, is_train=True):
    """
    处理光伏时间序列数据的核心流水线。
    功能包括：数据重采样、时区对齐、计算物理晴空辐射、防穿越归一化等。
    """
    # ---------------------------------------------------------
    # 1. 根据当前处理的是训练集还是测试集，加载对应的配置路径
    # ---------------------------------------------------------
    if is_train:
        CSV_PATH = config["train_file_paths"]["pv_file"]  # 原始 SCADA 数据路径
        OUTPUT_PATH = config["train_file_paths"]["series_file"]  # 处理后的输出路径
        START_DATE = config["train_dates"]["start_date"]  # 数据截取起始日期
        END_DATE = config["train_dates"]["end_date"]  # 数据截取结束日期
        LATITUDE = config["train_stations"]["lat"]  # 电站纬度
        LONGITUDE = config["train_stations"]["lon"]  # 电站经度
        ALTITUDE = config["train_stations"]["altitude"]  # 电站海拔
        CAPACITY = config["train_stations"]["capacity"]  # 电站装机容量 (用于归一化功率)
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

    # ---------------------------------------------------------
    # 2. 读取原始数据并进行时间段截取
    # ---------------------------------------------------------
    try:
        # 将 timestamp 列解析为时间格式，并设为 DataFrame 的索引
        df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'], index_col='timestamp')
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return

    # 截取实验所需的日期范围 (包含结束日期当天的 23:59:59)
    mask = (df.index >= START_DATE) & (df.index <= f"{END_DATE} 23:59:59")
    df = df.loc[mask]

    if df.empty:
        print("⚠️ 警告：筛选后数据为空，请检查 CSV 中的时间列是否正确！")
        return

    # ---------------------------------------------------------
    # 3. 提取特征并进行降采样 (5分钟 -> 15分钟)
    # ---------------------------------------------------------
    target_columns = [
        'Active_Power',  # 实测功率
        'Global_Horizontal_Radiation',  # 实测总辐射
        'Weather_Temperature_Celsius',  # 环境温度
        'Weather_Relative_Humidity'  # 相对湿度
    ]

    df_15min = df[target_columns].resample('15min').mean()
    df_15min = df_15min.interpolate(method='linear', limit=4)

    # ---------------------------------------------------------
    # 4. 时区强制对齐 (计算天文学特征的先决条件)
    # ---------------------------------------------------------
    try:
        if df_15min.index.tz is None:
            df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT',
                                                        nonexistent='shift_forward')
        else:
            df_15min.index = df_15min.index.tz_convert('Australia/Darwin')
    except Exception as e:
        df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT')

    # ---------------------------------------------------------
    # 5. 引入物理模型 (pvlib) 自动计算天文学特征
    # ---------------------------------------------------------
    location = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE, tz='Australia/Darwin')
    times = df_15min.index

    solpos = location.get_solarposition(times)
    df_15min['Solar_Zenith'] = solpos['zenith'].values

    cs = location.get_clearsky(times, model='ineichen')
    df_15min['Clear_Sky_GHI'] = cs['ghi'].values

    df_clean = df_15min

    # ---------------------------------------------------------
    # 6. 对电站属性相关的数据进行防爆处理与独立归一化
    # ---------------------------------------------------------
    df_clean['Power_Norm'] = df_clean['Active_Power'] / CAPACITY
    df_clean['Power_Norm'] = df_clean['Power_Norm'].clip(lower=0)
    df_clean['Clear_Sky_GHI'] = df_clean['Clear_Sky_GHI'].clip(lower=0)
    df_clean['Global_Horizontal_Radiation'] = df_clean['Global_Horizontal_Radiation'].clip(lower=0)

    # 🌟 核心备份：保存真实的物理角度，用于测试阶段准确切除夜晚数据
    df_clean['Solar_Zenith_Raw'] = df_clean['Solar_Zenith']

    # ---------------------------------------------------------
    # 7. 防止数据穿越的严格归一化流程 (Min-Max Scaling)
    # ---------------------------------------------------------
    columns_to_normalize = [
        'Global_Horizontal_Radiation',
        'Weather_Temperature_Celsius',
        'Weather_Relative_Humidity',
        'Clear_Sky_GHI',  # 👈 注意这里，经过这一步，它将被转化为 0~1 的分布
        'Solar_Zenith'
    ]

    scaler_path = config.get("scaler_params_paths", "../config/scaler_params.json")

    if is_train:
        scaler_params = {}
        for col in columns_to_normalize:
            col_min = df_clean[col].min()
            col_max = df_clean[col].max()
            scaler_params[col] = {'min': float(col_min), 'max': float(col_max)}

            if col_max - col_min == 0:
                df_clean[col] = 0.0
            else:
                df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)

        with open(scaler_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        print(f"   ✅ 归一化参数已保存至: {scaler_path}")

    else:
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ 找不到归一化参数文件 {scaler_path}，请先运行一次训练集生成尺子！")

        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        for col in columns_to_normalize:
            col_min = scaler_params[col]['min']
            col_max = scaler_params[col]['max']

            if col_max - col_min == 0:
                df_clean[col] = 0.0
            else:
                df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)

    # ---------------------------------------------------------
    # 8. 🌟 新增计算 CSI (晴空指数) 和重命名特征
    # ---------------------------------------------------------
    # 此时 Clear_Sky_GHI 已经严格按照训练集标尺被缩放到了 [0, 1] 之间。
    # 限制极小值防止除以0产生无穷大（np.clip下限设为0.01）
    safe_clear_sky = np.clip(df_clean['Clear_Sky_GHI'].values, a_min=0.01, a_max=None)

    # 物理意义计算：CSI = 归一化实测功率 / 归一化晴空功率
    df_clean['CSI'] = df_clean['Power_Norm'] / safe_clear_sky

    # ---------------------------------------------------------
    # 9. 保存最终生成的干净 CSV
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH)
    print(f"💾 处理完成！文件已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    config = load_config("../config/config.yaml")
    # print("========== 阶段 1: 处理训练集 ==========")
    # process_timeseries(config, is_train=True)
    print("\n========== 阶段 2: 处理验证/测试集 ==========")
    process_timeseries(config, is_train=False)