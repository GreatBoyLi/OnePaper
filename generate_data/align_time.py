import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from joblib import Parallel, delayed
from utils.config import load_config
import multiprocessing


def parse_filename_to_time(filename):
    """从文件名解析时间"""
    try:
        # 根据你之前生成的 _crop.npy 的实际命名规则，这里可能需要调整
        # 假设葵花8号标准命名类似: HS_H08_20260318_0010_..._crop.npy
        parts = filename.split('_')
        date_part = parts[2]  # YYYYMMDD
        time_part = parts[3]  # HHMM
        dt_str = f"{date_part}{time_part}"
        return datetime.strptime(dt_str, "%Y%m%d%H%M")
    except Exception:
        return None


def process_alignment_for_day(current_date, input_root, output_root, target_channels=3):
    """
    处理单日的时间对齐 (支持多通道张量)
    target_channels: 期望的通道数量 (目前是 albedo_03, tbb_07, tbb_13 共 3 个)
    """
    yyyy = current_date.strftime("%Y")
    mm = current_date.strftime("%m")
    dd = current_date.strftime("%d")
    yyyymm = f"{yyyy}{mm}"

    day_input_dir = os.path.join(input_root, yyyymm, dd)
    day_output_dir = os.path.join(output_root, yyyymm, dd)

    if not os.path.exists(day_input_dir):
        return f"⚠️ {yyyy}-{mm}-{dd} 目录不存在，跳过"

    files = [f for f in os.listdir(day_input_dir) if f.endswith(".npy")]
    file_map = {parse_filename_to_time(f): os.path.join(day_input_dir, f) for f in files if parse_filename_to_time(f)}

    if not file_map:
        return f"⚠️ {yyyy}-{mm}-{dd} 无有效的时间戳文件"

    if not os.path.exists(day_output_dir):
        os.makedirs(day_output_dir, exist_ok=True)

    target_times = pd.date_range(start=current_date, periods=24 * 4, freq='15min')
    success_count = 0
    skip_count = 0

    for target_t in target_times:
        save_name = f"sat_15min_{target_t.strftime('%Y%m%d_%H%M')}.npy"
        save_path = os.path.join(day_output_dir, save_name)

        # ==========================================
        # 逻辑 A: 直接匹配 (00分, 30分)
        # ==========================================
        if target_t in file_map:
            img = np.load(file_map[target_t]).astype(np.float32)

            # 🛡️ 维度安全校验：必须是 3D 且通道数为 target_channels
            if len(img.shape) == 3 and img.shape[0] == target_channels:
                np.save(save_path, img)
                success_count += 1
            else:
                skip_count += 1

        # ==========================================
        # 逻辑 B: 多通道线性插值 (15分, 45分)
        # ==========================================
        else:
            minute = target_t.minute
            remain = minute % 10
            prev_t = target_t - timedelta(minutes=remain)
            next_t = prev_t + timedelta(minutes=10)

            # 必须前后 10 分钟的文件都存在，才能进行插值
            if prev_t in file_map and next_t in file_map:
                img_prev = np.load(file_map[prev_t]).astype(np.float32)
                img_next = np.load(file_map[next_t]).astype(np.float32)

                # 🛡️ 维度安全校验
                if (len(img_prev.shape) == 3 and img_prev.shape[0] == target_channels and
                        len(img_next.shape) == 3 and img_next.shape[0] == target_channels):

                    # 💡 这里的加法和除法，Numpy 会自动同时作用于 3 个通道！
                    # 即：albedo 和 albedo 插值，tbb 和 tbb 插值，互不干扰
                    img_interp = (img_prev + img_next) / 2.0
                    np.save(save_path, img_interp)
                    success_count += 1
                else:
                    skip_count += 1

    msg = f"✅ {yyyy}-{mm}-{dd}: 成功对齐 {success_count}/96"
    if skip_count > 0:
        msg += f" (⚠️ {skip_count} 个因通道异常被丢弃)"
    return msg


if __name__ == "__main__":
    config = load_config("../config/config.yaml")

    CROP_DIR = config["file_paths"]["crop_statellite_path"]
    ALIGNED_DIR = config["file_paths"]["aligned_satellite_path"]

    dates = pd.date_range(start=config["dates"]["start_date"],
                          end=config["dates"]["end_date"], freq='D')

    print(f"🚀 开始并行时间对齐 (10min -> 15min，支持多通道)")

    # 留出 10 个核心防止机器卡死
    num_cores = max(1, multiprocessing.cpu_count() - 10)

    # 执行并行任务
    results = Parallel(n_jobs=num_cores)(
        delayed(process_alignment_for_day)(d, CROP_DIR, ALIGNED_DIR, target_channels=3)
        for d in tqdm(dates, desc="对齐进度")
    )

    # 打印结果摘要
    for r in results:
        if "⚠️" in r:
            print(r)

    print("\n🎉 所有日期多通道对齐完成！")