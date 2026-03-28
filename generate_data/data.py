from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from tqdm import tqdm

from generate_data.crop_satellite import process_single_day
from generate_data.align_time import process_alignment_for_day

from utils.config import load_config

if __name__ == "__main__":
    # is_train = False
    is_train = True
    config = load_config("../config/config.yaml")
    # 处理卫星图像
    if is_train:
        TARGET_LAT = config["train_stations"]["lat"]
        TARGET_LON = config["train_stations"]["lon"]
        CROP_SIZE = config["statellite"]["crop_size"]
        BASE_SATELLITE_PATH = config["train_file_paths"]["satellite_path"]
        BASE_SAVE_DIR = config["train_file_paths"]["crop_statellite_path"]
        ALIGNED_DIR = config["train_file_paths"]["aligned_satellite_path"]
        dates = pd.date_range(start=config["train_dates"]["start_date"], end=config["train_dates"]["end_date"],
                              freq='D')
    else:
        TARGET_LAT = config["val_stations"]["lat"]
        TARGET_LON = config["val_stations"]["lon"]
        CROP_SIZE = config["statellite"]["crop_size"]
        BASE_SATELLITE_PATH = config["val_file_paths"]["satellite_path"]
        BASE_SAVE_DIR = config["val_file_paths"]["crop_statellite_path"]
        ALIGNED_DIR = config["val_file_paths"]["aligned_satellite_path"]
        dates = pd.date_range(start=config["val_dates"]["start_date"], end=config["val_dates"]["end_date"],
                              freq='D')

    # ==========================================
    # 🌟 修改 5：在这里定义你需要提取的变量列表
    # (可以根据你的实验需求随时增减)
    # ==========================================
    TARGET_VARS = ['albedo_03', 'tbb_07', 'tbb_13']

    print(f"🛰️ 卫星数据裁剪开始，总日期数: {len(dates)}，提取通道数: {len(TARGET_VARS)}")

    # 留出 10 个核心防止机器卡死
    num_cores = max(1, multiprocessing.cpu_count() - 10)

    Parallel(n_jobs=num_cores, verbose=10)(
        delayed(process_single_day)(
            d, BASE_SATELLITE_PATH, BASE_SAVE_DIR, TARGET_LAT, TARGET_LON, CROP_SIZE, TARGET_VARS
        ) for d in dates
    )

    print("✅ 所有任务已圆满完成！")

    # 对齐卫星图像的时间
    print(f"🚀 开始并行时间对齐 (10min -> 15min，支持多通道)")

    # 执行并行任务
    results = Parallel(n_jobs=num_cores)(
        delayed(process_alignment_for_day)(d, BASE_SAVE_DIR, ALIGNED_DIR, target_channels=len(TARGET_VARS))
        for d in tqdm(dates, desc="对齐进度")
    )

    # 打印结果摘要
    for r in results:
        if "⚠️" in r:
            print(r)

    print("\n🎉 所有日期多通道对齐完成！")
