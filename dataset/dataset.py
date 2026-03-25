import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class SatellitePVDataset(Dataset):
    def __init__(self, csv_path, satellite_dir,
                 input_seq_len=16, output_seq_len=4,
                 crop_size=96, mode="train"):

        self.input_len = input_seq_len
        self.output_len = output_seq_len
        self.satellite_dir = satellite_dir
        self.crop_size = crop_size

        # 1. 读取已经由 process_timeseries.py 预处理好的 CSV
        self.data = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        self.data = self.data.sort_index()

        # ====================================================================
        # 🌟 进阶修改：先插值抢救，抢救无效再剔除 (兼顾连续性与真实性)
        # ====================================================================
        check_cols = [
            'Active_Power', 'Global_Horizontal_Radiation',
            'Weather_Temperature_Celsius', 'Weather_Relative_Humidity',
            'Power_Norm', 'CSI', 'Clear_Sky_GHI'
        ]
        valid_check_cols = [col for col in check_cols if col in self.data.columns]

        len_before = len(self.data)

        # 第一步：时间线性插值抢救
        # limit=4 表示最多允许连续缺失 4 个点 (即 1 小时)。
        # 如果连续缺失 5 个点，中间的空值将保留，留给下一步 dropna 删掉。
        # limit_direction='both' 允许双向插值
        self.data[valid_check_cols] = self.data[valid_check_cols].interpolate(
            method='time',
            limit=4,
            limit_direction='both'
        )

        # 第二步：物理常识兜底 (可选)
        # 插值可能会把原本夜间的 CSI 插成微小的负数或正数，这里截断一下
        if 'CSI' in self.data.columns:
            self.data['CSI'] = self.data['CSI'].clip(lower=0.0)
        if 'Power_Norm' in self.data.columns:
            self.data['Power_Norm'] = self.data['Power_Norm'].clip(lower=0.0)

        # 第三步：清理无药可救的断层
        # 经过 limit=4 的插值后依然是 NaN 的，说明缺失时间太长，只能强行截断
        self.data = self.data.dropna(subset=valid_check_cols)
        len_after = len(self.data)

        if len_before != len_after:
            print(f"[{mode}] 🚑 数据修复与清洗: 尝试插值修复后，仍剔除了 {len_before - len_after} 行严重缺失的记录。")
        else:
            print(f"[{mode}] 🚑 数据修复完毕，时间序列完全连续！")
        # ====================================================================

        # 2. 时间序列的正余弦周期性编码 (捕捉日夜与季节周期规律)
        day_of_year = self.data.index.dayofyear
        minute_of_day = self.data.index.hour * 60 + self.data.index.minute

        self.data['sin_day'] = np.sin(2 * np.pi * day_of_year / 365.25)
        self.data['cos_day'] = np.cos(2 * np.pi * day_of_year / 365.25)
        self.data['sin_min'] = np.sin(2 * np.pi * minute_of_day / (24 * 60))
        self.data['cos_min'] = np.cos(2 * np.pi * minute_of_day / (24 * 60))

        # 3. 严格的时间连续性校验 (黑夜过滤全权交由 train.py 的 mask 控制)
        self.valid_indices = []
        total_len = self.input_len + self.output_len
        expected_time_delta = pd.Timedelta(minutes=15 * (total_len - 1))

        max_possible_idx = len(self.data) - total_len

        for i in range(max_possible_idx + 1):
            start_time = self.data.index[i]
            end_time = self.data.index[i + total_len - 1]

            # 校验时间序列是否连续无断层
            if end_time - start_time == expected_time_delta:
                self.valid_indices.append(i)

        print(f"[{mode}] 数据集加载完成 | 原始总时间步: {len(self.data)}")
        print(f"   -> ✅ 最终连续有效样本: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def _get_safe_background(self, timestamp):
        # 季节性安全背景填充逻辑，保持不变
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

        # 🌟 网络输入数值特征
        # 确保全部是归一化后的数据，保证模型梯度平稳
        features = [
            'Power_Norm',
            'Global_Horizontal_Radiation',
            'Weather_Temperature_Celsius',
            'Weather_Relative_Humidity',
            'Clear_Sky_GHI',  # 👈 直接读取 CSV 中已有的归一化晴空基准
            'Solar_Zenith',
            'sin_day', 'cos_day',
            'sin_min', 'cos_min'
        ]

        x_numeric = self.data.iloc[hist_start:hist_end][features].values

        # 🌟 现在的 y 获取逻辑极其简洁，直接读取 CSV 中对应的列即可
        y_power = self.data.iloc[hist_end:future_end]['Power_Norm'].values
        y_csi = self.data.iloc[hist_end:future_end]['CSI'].values
        y_clear_sky_ghi = self.data.iloc[hist_end:future_end]['Clear_Sky_GHI'].values
        y_zenith_raw = self.data.iloc[hist_end:future_end]['Solar_Zenith_Raw'].values

        # ==========================================
        # 以下获取图像及线性插值的逻辑保持不变
        # ==========================================
        hist_timestamps = self.data.index[hist_start:hist_end]

        # 1. 第一阶段：尝试读取所有存在的图像，缺失的用 None 占位
        raw_images = []
        for ts in hist_timestamps:
            file_name = f"sat_15min_{ts.strftime('%Y%m%d_%H%M')}.npy"
            yyyy, mm, dd = ts.strftime("%Y"), ts.strftime("%m"), ts.strftime("%d")
            yyyymm = f"{yyyy}{mm}"
            file_path = os.path.join(self.satellite_dir, yyyymm, dd, file_name)

            img_valid = False
            if os.path.exists(file_path):
                img = np.load(file_path).astype(np.float32)

                if img.shape == (3, self.crop_size, self.crop_size):
                    # 🌟 计算损坏像素的比例
                    bad_mask = np.isnan(img) | np.isinf(img)
                    bad_ratio = bad_mask.sum() / img.size

                    if bad_ratio == 0:
                        # 完美图像，直接采用
                        raw_images.append(img)
                        img_valid = True
                    elif bad_ratio < 0.10:
                        # 🌟 损坏比例小于 10%，只坏了几个点，我们抢救它！
                        for c in range(3):
                            c_data = img[c]
                            c_bad = bad_mask[c]
                            if c_bad.any():
                                # 取该通道中所有【正常像素】的平均值
                                valid_pixels = c_data[~bad_mask[c]]
                                # 防止该通道全坏导致 valid_mean 报错
                                valid_mean = np.nanmean(valid_pixels) if len(valid_pixels) > 0 else 0.0
                                # 用平均值填补坏点
                                c_data[c_bad] = valid_mean
                            img[c] = c_data

                        raw_images.append(img)
                        img_valid = True
                    else:
                        # 损坏严重 (>10%)，放弃抢救，交给后面的线性插值去处理
                        pass

                        # 如果图片不存在，或者损坏过于严重，填入 None
            if not img_valid:
                raw_images.append(None)

        # 2. 第二阶段：🌟 核心线性插值逻辑 🌟
        for i in range(len(raw_images)):
            if raw_images[i] is None:
                # 向前寻找最近的有效帧
                prev_idx = i - 1
                while prev_idx >= 0 and raw_images[prev_idx] is None:
                    prev_idx -= 1

                # 向后寻找最近的有效帧
                next_idx = i + 1
                while next_idx < len(raw_images) and raw_images[next_idx] is None:
                    next_idx += 1

                # 场景 1：前后都有有效图像 -> 线性插值
                if prev_idx >= 0 and next_idx < len(raw_images):
                    dist_prev = i - prev_idx
                    dist_next = next_idx - i
                    total_dist = dist_prev + dist_next

                    # 距离越近，权重越大
                    w_prev = dist_next / total_dist
                    w_next = dist_prev / total_dist

                    interp_img = raw_images[prev_idx] * w_prev + raw_images[next_idx] * w_next
                    raw_images[i] = interp_img

                # 场景 2：只有前面有有效帧 -> 直接复制前一帧
                elif prev_idx >= 0:
                    raw_images[i] = raw_images[prev_idx].copy()

                # 场景 3：只有后面有有效帧 -> 直接复制后一帧
                elif next_idx < len(raw_images):
                    raw_images[i] = raw_images[next_idx].copy()

                # 场景 4：整段序列 16 步全部缺失！-> 用安全背景兜底
                else:
                    ts = hist_timestamps[i]
                    bg_values = self._get_safe_background(ts)
                    safe_bg = bg_values[:, None, None] * np.ones((3, self.crop_size, self.crop_size), dtype=np.float32)
                    raw_images[i] = safe_bg

        # 3. 第三阶段：统一的归一化处理
        final_images = []
        for img in raw_images:
            img[0] = np.clip(img[0], 0.0, 1.0)
            img[1:] = (img[1:] - 180.0) / (345.0 - 180.0)
            img[1:] = np.clip(img[1:], 0.0, 1.0)
            final_images.append(img)

        x_images = np.stack(final_images, axis=0)

        return {
            'x_images': torch.from_numpy(x_images.copy()).float(),
            'x_numeric': torch.from_numpy(x_numeric.copy()).float(),
            'y_power': torch.from_numpy(y_power.copy()).float(),  # 真实功率 (Loss 评估备用)
            'y_csi': torch.from_numpy(y_csi.copy()).float(),  # 真实CSI (主要预测目标)
            'y_clear_sky_ghi': torch.from_numpy(y_clear_sky_ghi.copy()).float(),  # 未来晴空基准 (用于还原功率)
            'y_zenith': torch.from_numpy(y_zenith_raw.copy()).float()  # 真实物理角度 (用于掩蔽黑夜 Loss)
        }


if __name__ == "__main__":
    from utils.config import load_config

    config = load_config("../config/config.yaml")
    csv_file = config["val_file_paths"]["series_file"]
    sat_dir = config["val_file_paths"]["aligned_satellite_path"]

    if os.path.exists(csv_file):
        c_size = config.get("statellite", {}).get("crop_size", 96)

        ds = SatellitePVDataset(csv_file, sat_dir, crop_size=c_size)
        if len(ds) > 0:
            sample = ds[0]
            print(f"Input Image Shape : {sample['x_images'].shape}   # 预期: (16, 3, 96, 96)")
            print(f"Input Numeric Shape: {sample['x_numeric'].shape} # 预期: (16, 10)")
            print(f"Target CSI Shape   : {sample['y_csi'].shape}       # 预期: (4,)")
            print(f"Target Zenith Shape: {sample['y_zenith'].shape}  # 预期: (4,)")
    else:
        print("请先生成对齐后的 CSV 文件")
