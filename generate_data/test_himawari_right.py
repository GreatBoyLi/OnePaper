import os
import numpy as np

# 设定你下载的那个文件的路径 (请修改这里)
FILE_PATH = "../data/himawari_nc/202001/27/NC_H08_20200127_0000_R21_FLDK.02401_02401.nc"


def inspect_nc_file(file_path):
    print(f"🔍 正在检查文件: {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        print("❌ 错误: 文件路径不存在，请检查路径是否正确。")
        return

    # 尝试使用 xarray (推荐，自动处理数值转换)
    try:
        import xarray as xr
        print("✅ 使用 xarray 库读取...")
        try:
            ds = xr.open_dataset(file_path)
            print("\n📋 文件包含的变量:")
            print(list(ds.keys()))

            # 寻找 Band 13 变量 (通常命名为 'tbb_13' 或类似)
            target_var = None
            for var in ds.keys():
                if "13" in var and ("tbb" in var.lower() or "band" in var.lower() or "temp" in var.lower()):
                    target_var = var
                    break

            if target_var:
                data = ds[target_var].values
                # 过滤无效值 (通常是 NaN 或极小负数)
                valid_data = data[data > 0]

                print(f"\n🎯 发现目标波段变量: {target_var}")
                print(f"   - 最小值: {np.nanmin(valid_data):.2f} K")
                print(f"   - 最大值: {np.nanmax(valid_data):.2f} K")
                print(f"   - 平均值: {np.nanmean(valid_data):.2f} K")

                # 验证逻辑: 论文提到范围是 175K - 340K
                vmin, vmax = np.nanmin(valid_data), np.nanmax(valid_data)
                if 150 < vmin < 250 and 280 < vmax < 350:
                    print("\n✅ [通过] 数值范围符合物理规律 (175K-340K)。文件正确！")
                else:
                    print("\n⚠️ [警告] 数值范围看起来有点奇怪，请检查是否需要手动应用 scale_factor。")
            else:
                print("\n⚠️ 未能自动找到 Band 13 变量，请人工核对上面的变量列表。")

            ds.close()
            return

        except Exception as e:
            print(f"读取出错: {e}")

    except ImportError:
        print("⚠️ 未安装 xarray，尝试使用 netCDF4...")

    # 备选方案: 使用 netCDF4
    try:
        import netCDF4
        print("✅ 使用 netCDF4 库读取...")
        nc = netCDF4.Dataset(file_path)

        print("\n📋 文件包含的变量:")
        print(nc.variables.keys())

        # 寻找 Band 13
        target_var = None
        for var in nc.variables:
            if "13" in var and ("tbb" in var.lower() or "band" in var.lower()):
                target_var = var
                break

        if target_var:
            var_obj = nc.variables[target_var]
            data = var_obj[:]

            # 检查是否需要缩放
            scale = getattr(var_obj, 'scale_factor', 1.0)
            offset = getattr(var_obj, 'add_offset', 0.0)

            # 转换为物理数值
            if scale != 1.0 or offset != 0.0:
                print(f"\n⚙️ 检测到压缩数据，正在应用: value * {scale} + {offset}")
                # 注意：netCDF4 读取时有时会自动应用，有时需要手动，视设置而定
                # 这里简单判断一下量级
                if np.max(data) > 10000:  # 肯定是原始整数
                    data = data * scale + offset

            valid_data = data[(data > 100) & (data < 400)]  # 粗略过滤

            print(f"\n🎯 发现目标波段变量: {target_var}")
            print(f"   - 最小值: {np.min(valid_data):.2f}")
            print(f"   - 最大值: {np.max(valid_data):.2f}")

            if 150 < np.min(valid_data) < 250 and 280 < np.max(valid_data) < 350:
                print("\n✅ [通过] 数值范围符合物理规律。文件正确！")
        else:
            print("\n⚠️ 未找到目标变量。")

        nc.close()

    except ImportError:
        print("❌ 错误: 你的环境里既没有 xarray 也没有 netCDF4。无法读取 .nc 文件。")
        print("请运行: pip install xarray netCDF4")


if __name__ == "__main__":
    inspect_nc_file(FILE_PATH)