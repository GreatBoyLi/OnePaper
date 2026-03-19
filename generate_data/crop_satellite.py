import xarray as xr
import numpy as np
import os
import pandas as pd
from utils.config import load_config
from joblib import Parallel, delayed
import multiprocessing


# 🌟 修改 1：参数列表中增加了 target_vars 列表
def process_one_directory(daily_dir, save_dir, target_lat, target_lon, crop_size, target_vars):
    """
    处理一个日期目录下的所有 .nc 文件
    """
    if not os.path.exists(daily_dir):
        print(f"⚠️ 目录不存在，跳过: {daily_dir}")
        return

    # 遍历目录下的文件
    for file in os.listdir(daily_dir):
        if not file.endswith(".nc"):
            continue

        full_file_path = os.path.join(daily_dir, file)

        try:
            ds = xr.open_dataset(full_file_path, decode_timedelta=True, engine='netcdf4')

            lats = ds['latitude'].values
            lons = ds['longitude'].values

            lat_idx = (np.abs(lats - target_lat)).argmin()
            lon_idx = (np.abs(lons - target_lon)).argmin()

            half = crop_size // 2
            lat_start = max(0, lat_idx - half)
            lat_end = min(len(lats), lat_idx + half)
            lon_start = max(0, lon_idx - half)
            lon_end = min(len(lons), lon_idx + half)

            lat_slice = slice(lat_start, lat_end)
            lon_slice = slice(lon_start, lon_end)

            # ==========================================
            # 🌟 修改 2：循环提取多个变量，并堆叠为多通道矩阵
            # ==========================================
            channel_data_list = []
            for var in target_vars:
                # 检查变量是否在文件中
                if var not in ds.variables:
                    print(f"⚠️ 变量 {var} 不在 {file} 中，跳过该文件")
                    ds.close()
                    continue

                # 提取单个变量的裁剪切片
                single_var_data = ds[var].isel(latitude=lat_slice, longitude=lon_slice).values
                channel_data_list.append(single_var_data)

            # 沿着第0个维度堆叠，形成 (Channel, Height, Width) 的 3D 数组
            multi_channel_data = np.stack(channel_data_list, axis=0)

            # 🌟 修改 3：检查多通道数据的形状是否符合预期 (C, H, W)
            expected_shape = (len(target_vars), crop_size, crop_size)
            if multi_channel_data.shape != expected_shape:
                print(f"⚠️ {file} 裁剪尺寸异常 {multi_channel_data.shape}，预期 {expected_shape}，跳过")
                ds.close()
                continue

            # 直接保存为 .npy
            file_name = file.replace(".nc", "_crop.npy")
            save_path = os.path.join(save_dir, file_name)

            np.save(save_path, multi_channel_data.astype(np.float32))

            ds.close()

        except Exception as e:
            print(f"❌ 处理失败 {file}: {e}")


# 🌟 修改 4：透传 target_vars 参数
def process_single_day(current_date, base_read_path, base_save_path, target_lat, target_lon, crop_size, target_vars):
    yyyy = current_date.strftime("%Y")
    mm = current_date.strftime("%m")
    dd = current_date.strftime("%d")
    yyyymm = f"{yyyy}{mm}"

    daily_read_path = os.path.join(base_read_path, yyyymm, dd)
    daily_save_path = os.path.join(base_save_path, yyyymm, dd)

    if not os.path.exists(daily_save_path):
        os.makedirs(daily_save_path, exist_ok=True)

    print(f"🚀 开始多进程任务: {yyyy}-{mm}-{dd}")
    process_one_directory(daily_read_path, daily_save_path, target_lat, target_lon, crop_size, target_vars)
    return f"Done: {yyyy}-{mm}-{dd}"


if __name__ == "__main__":
    config = load_config("../config/config.yaml")

    TARGET_LAT = config["stations"]["lat"]
    TARGET_LON = config["stations"]["lon"]
    CROP_SIZE = config["statellite"]["crop_size"]
    BASE_SATELLITE_PATH = config["file_paths"]["satellite_path"]
    BASE_SAVE_DIR = config["file_paths"]["crop_statellite_path"]

    # ==========================================
    # 🌟 修改 5：在这里定义你需要提取的变量列表
    # (可以根据你的实验需求随时增减)
    # ==========================================
    TARGET_VARS = ['albedo_03', 'tbb_07', 'tbb_13']

    dates = pd.date_range(start=config["dates"]["start_date"],
                          end=config["dates"]["end_date"], freq='D')

    print(f"🛰️ 卫星数据裁剪开始，总日期数: {len(dates)}，提取通道数: {len(TARGET_VARS)}")

    num_cores = multiprocessing.cpu_count() - 5

    Parallel(n_jobs=num_cores, verbose=10)(
        delayed(process_single_day)(
            d, BASE_SATELLITE_PATH, BASE_SAVE_DIR, TARGET_LAT, TARGET_LON, CROP_SIZE, TARGET_VARS
        ) for d in dates
    )

    print("✅ 所有任务已圆满完成！")
