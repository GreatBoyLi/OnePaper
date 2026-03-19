import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from utils.config import load_config


class SatellitePVDataset(Dataset):
    def __init__(self, csv_path, satellite_dir,
                 input_seq_len=16, output_seq_len=4,
                 crop_size=96,
                 mode='train', train_ratio=0.7, val_ratio=0.2):

        self.input_len = input_seq_len
        self.output_len = output_seq_len
        self.satellite_dir = satellite_dir
        self.crop_size = crop_size

        # 1. 读取 CSV
        self.df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        self.df = self.df.sort_index()

        # ==========================================
        # 🌟 核心升级 5：时间序列的正余弦周期性编码
        # ==========================================
        # 提取时间元素
        day_of_year = self.df.index.dayofyear
        minute_of_day = self.df.index.hour * 60 + self.df.index.minute

        # 年周期 (区分春夏秋冬，适应太阳直射点的回归运动)
        self.df['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
        self.df['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)

        # 日周期 (区分早中晚，适应地球自转)
        self.df['sin_min'] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
        self.df['cos_min'] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

        # 2. 划分数据集
        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        if mode == 'train':
            self.data = self.df.iloc[:train_end]
        elif mode == 'val':
            self.data = self.df.iloc[train_end:val_end]
        elif mode == 'test':
            self.data = self.df.iloc[val_end:]
        else:
            raise ValueError(f"不支持的 mode: {mode}")

        # 3. 严格的时间连续性校验
        self.valid_indices = []
        total_len = self.input_len + self.output_len
        expected_time_delta = pd.Timedelta(minutes=15 * (total_len - 1))

        max_possible_idx = len(self.data) - total_len
        for i in range(max_possible_idx + 1):
            start_time = self.data.index[i]
            end_time = self.data.index[i + total_len - 1]

            if end_time - start_time == expected_time_delta:
                self.valid_indices.append(i)

        print(f"[{mode}] 数据集加载完成 | 原始行数: {len(self.data)} | 连续有效样本: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def _get_safe_background(self, timestamp):
        month = timestamp.month
        hour = timestamp.hour

        if month in [11, 12, 1, 2, 3]:
            season = 'summer'
        elif month in [5, 6, 7, 8, 9]:
            season = 'winter'
        else:
            season = 'transition'

        is_day = 7 <= hour <= 18

        safe_albedo = 0.2 if is_day else 0.0

        if season == 'summer':
            safe_ir = 325.0 if is_day else 295.0
        elif season == 'winter':
            safe_ir = 295.0 if is_day else 275.0
        else:
            safe_ir = 310.0 if is_day else 285.0

        return np.array([safe_albedo, safe_ir, safe_ir], dtype=np.float32)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]

        hist_start = real_idx
        hist_end = real_idx + self.input_len
        future_end = hist_end + self.output_len

        # ==========================================
        # 🌟 核心升级 6：将时间特征加入网络输入列表
        # ==========================================
        features = [
            'Power_Norm',
            'Global_Horizontal_Radiation',
            'Weather_Temperature_Celsius',
            'Weather_Relative_Humidity',
            'Clear_Sky_GHI',
            'Solar_Zenith',
            'sin_day', 'cos_day',  # 👈 新增的年周期特征
            'sin_min', 'cos_min'  # 👈 新增的日周期特征
        ]

        x_numeric = self.data.iloc[hist_start:hist_end][features].values
        y_power = self.data.iloc[hist_end:future_end]['Power_Norm'].values
        y_zenith = self.data.iloc[hist_end:future_end]['Solar_Zenith'].values

        # 获取图像数据
        hist_timestamps = self.data.index[hist_start:hist_end]
        images = []
        last_valid_img = None

        for ts in hist_timestamps:
            bg_values = self._get_safe_background(ts)
            current_safe_bg = bg_values[:, None, None] * np.ones((3, self.crop_size, self.crop_size), dtype=np.float32)

            file_name = f"sat_15min_{ts.strftime('%Y%m%d_%H%M')}.npy"
            yyyy, mm, dd = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
            yyyymm = f"{yyyy}{mm}"
            file_path = os.path.join(self.satellite_dir, yyyymm, dd, file_name)

            if os.path.exists(file_path):
                img = np.load(file_path).astype(np.float32)

                if img.shape != (3, self.crop_size, self.crop_size):
                    img = last_valid_img.copy() if last_valid_img is not None else current_safe_bg.copy()
                else:
                    for c in range(3):
                        if np.isnan(img[c]).any() or np.isinf(img[c]).any():
                            valid_mean = np.nanmean(img[c][~np.isinf(img[c])])
                            if np.isnan(valid_mean):
                                img[c] = last_valid_img[c] if last_valid_img is not None else current_safe_bg[c]
                            else:
                                img[c] = np.nan_to_num(img[c], nan=valid_mean, posinf=valid_mean, neginf=valid_mean)
                last_valid_img = img.copy()
            else:
                img = last_valid_img.copy() if last_valid_img is not None else current_safe_bg.copy()

            img[0] = np.clip(img[0], 0.0, 1.0)
            img[1:] = (img[1:] - 180.0) / (345.0 - 180.0)
            img[1:] = np.clip(img[1:], 0.0, 1.0)

            images.append(img)

        x_images = np.stack(images, axis=0)

        return {
            'x_images': torch.from_numpy(x_images).float(),  # Shape: (16, 3, 96, 96)
            'x_numeric': torch.from_numpy(x_numeric).float(),  # Shape: (16, 10) 👈 特征数变为10
            'y': torch.from_numpy(y_power).float(),  # Shape: (4,)
            'y_zenith': torch.from_numpy(y_zenith).float()  # Shape: (4,)
        }


if __name__ == "__main__":
    config = load_config("../config/config.yaml")
    csv_file = config["file_paths"]["series_file"]
    sat_dir = config["file_paths"]["aligned_satellite_path"]

    if os.path.exists(csv_file):
        c_size = config.get("statellite", {}).get("crop_size", 96)

        ds = SatellitePVDataset(csv_file, sat_dir, crop_size=c_size, mode='train')
        if len(ds) > 0:
            sample = ds[0]
            print(f"Input Image Shape : {sample['x_images'].shape}   # 预期: (16, 3, 96, 96)")
            print(f"Input Numeric Shape: {sample['x_numeric'].shape} # 预期: (16, 10) <- 包含了4个时间特征")
            print(f"Target Power Shape : {sample['y'].shape}         # 预期: (4,)")
            print(f"Target Zenith Shape: {sample['y_zenith'].shape}  # 预期: (4,)")
    else:
        print("请先生成对齐后的 CSV 文件")