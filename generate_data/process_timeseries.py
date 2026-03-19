import pandas as pd
import pvlib
import os
import json
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
    # 这些是我们真正需要喂给神经网络的数值特征
    target_columns = [
        'Active_Power',  # 实测功率
        'Global_Horizontal_Radiation',  # 实测总辐射
        'Weather_Temperature_Celsius',  # 环境温度
        'Weather_Relative_Humidity'  # 相对湿度
    ]

    # 将 5 分钟的分辨率降为 15 分钟 (求平均值)，以严格对齐卫星图像的时间步长
    df_15min = df[target_columns].resample('15min').mean()
    # 使用线性插值填补极少量的数据缺失 (limit=4 表示最多只连续填补 1 小时)
    df_15min = df_15min.interpolate(method='linear', limit=4)

    # ---------------------------------------------------------
    # 4. 时区强制对齐 (计算天文学特征的先决条件)
    # ---------------------------------------------------------
    try:
        # 如果原始数据没有时区标签，强制打上澳大利亚当地时区 (ACST)
        if df_15min.index.tz is None:
            df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT',
                                                        nonexistent='shift_forward')
        else:
            # 如果原始数据是 UTC 等其他时区，转换为当地时间
            df_15min.index = df_15min.index.tz_convert('Australia/Darwin')
    except Exception as e:
        # 兜底处理：忽略异常直接强转
        df_15min.index = df_15min.index.tz_localize('Australia/Darwin', ambiguous='NaT')

    # ---------------------------------------------------------
    # 5. 引入物理模型 (pvlib) 自动计算天文学特征
    # ---------------------------------------------------------
    location = pvlib.location.Location(LATITUDE, LONGITUDE, altitude=ALTITUDE, tz='Australia/Darwin')
    times = df_15min.index

    # 5.1 计算太阳位置 (我们最需要的是 Solar_Zenith 太阳天顶角，用于判断黑夜)
    solpos = location.get_solarposition(times)
    df_15min['Solar_Zenith'] = solpos['zenith'].values

    # 5.2 计算理论晴空辐射 (Clear Sky GHI，这是光伏预测极强的参考基准)
    cs = location.get_clearsky(times, model='ineichen')
    df_15min['Clear_Sky_GHI'] = cs['ghi'].values

    df_clean = df_15min

    # ---------------------------------------------------------
    # 6. 对电站属性相关的数据进行防爆处理与独立归一化
    # ---------------------------------------------------------
    # 功率归一化公式: 实际功率 / 电站最大装机容量 -> 映射到 [0, 1] 左右
    df_clean['Power_Norm'] = df_clean['Active_Power'] / CAPACITY

    # 将所有的负值噪点（如逆变器夜间待机耗电、传感器微小负偏置）暴力截断为 0
    df_clean['Power_Norm'] = df_clean['Power_Norm'].clip(lower=0)
    df_clean['Clear_Sky_GHI'] = df_clean['Clear_Sky_GHI'].clip(lower=0)
    df_clean['Global_Horizontal_Radiation'] = df_clean['Global_Horizontal_Radiation'].clip(lower=0)

    # 🌟 核心备份：保存真实的物理角度，用于测试阶段准确切除夜晚数据
    df_clean['Solar_Zenith_Raw'] = df_clean['Solar_Zenith']

    # ---------------------------------------------------------
    # 7. 防止数据穿越的严格归一化流程 (Min-Max Scaling)
    # ---------------------------------------------------------
    # 定义需要被缩放为 0~1 的气象特征列表
    columns_to_normalize = [
        'Global_Horizontal_Radiation',
        'Weather_Temperature_Celsius',
        'Weather_Relative_Humidity',
        'Clear_Sky_GHI',
        'Solar_Zenith'
    ]

    # 定义归一化参数字典 (json) 的保存路径
    scaler_path = config["scaler_params_paths"]

    if is_train:
        # 如果是训练集：提取每一列的 Min 和 Max，并将其永久保存，制定"评分标准"
        scaler_params = {}
        for col in columns_to_normalize:
            col_min = df_clean[col].min()
            col_max = df_clean[col].max()

            # 记录标尺
            scaler_params[col] = {'min': float(col_min), 'max': float(col_max)}

            # 执行归一化: (X - Min) / (Max - Min)
            if col_max - col_min == 0:
                df_clean[col] = 0.0
            else:
                df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)

        # 保存字典到本地
        with open(scaler_path, 'w') as f:
            json.dump(scaler_params, f, indent=4)
        print(f"   ✅ 归一化参数已保存至: {scaler_path}")

    else:
        # 如果是验证/测试集：绝对不能重新计算 Min/Max，必须加载训练集制定的"评分标准"
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"❌ 找不到归一化参数文件 {scaler_path}，请先运行一次训练集生成尺子！")

        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)

        for col in columns_to_normalize:
            # 提取训练集的标尺
            col_min = scaler_params[col]['min']
            col_max = scaler_params[col]['max']

            # 用训练集的标尺量测现在的测试集数据
            if col_max - col_min == 0:
                df_clean[col] = 0.0
            else:
                df_clean[col] = (df_clean[col] - col_min) / (col_max - col_min)

    # ---------------------------------------------------------
    # 8. 保存最终生成的干净 CSV
    # ---------------------------------------------------------
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_clean.to_csv(OUTPUT_PATH)
    print(f"💾 处理完成！文件已保存至: {OUTPUT_PATH}")


if __name__ == "__main__":
    config = load_config("../config/config.yaml")
    # 标准的训练与测试数据生成流水线
    print("========== 阶段 1: 处理训练集 ==========")
    process_timeseries(config, is_train=True)
    print("\n========== 阶段 2: 处理验证/测试集 ==========")
    process_timeseries(config, is_train=False)