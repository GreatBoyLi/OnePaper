import os
import pandas as pd

csv_path = "../../data/val/validate_quan.csv"
save_dir = "weather_split"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

# =========================
# 0. 气象状态划分依据说明
# =========================
weather_basis = {
    "Clear-sky": "CSI >= 0.8",
    "Cloudy": "0.4 <= CSI < 0.8",
    "Rainy": "CSI < 0.4"
}

print("=" * 60)
print("Weather classification basis:")
print("Daytime samples are first selected by Solar_Zenith_Raw <= 85.")
print("Then, weather types are classified according to CSI:")
for weather_type, rule in weather_basis.items():
    print(f"  {weather_type}: {rule}")
print("=" * 60)

# 保存划分依据到 txt 文件，方便后续写论文或查记录
basis_path = os.path.join(save_dir, "weather_classification_basis.txt")
with open(basis_path, "w", encoding="utf-8") as f:
    f.write("Weather classification basis\n")
    f.write("Daytime samples: Solar_Zenith_Raw <= 85\n")
    f.write("Weather types are classified according to CSI:\n")
    for weather_type, rule in weather_basis.items():
        f.write(f"{weather_type}: {rule}\n")

# =========================
# 1. 剔除夜间样本
# =========================
df_day = df[df["Solar_Zenith_Raw"] <= 85].copy()

# 如果 CSI 有缺失值，先删除
df_day = df_day.dropna(subset=["CSI"])

# =========================
# 2. 按 CSI 划分气象状态
# =========================
def classify_weather(csi):
    if csi < 0.4:
        return "Rainy"
    elif csi < 0.8:
        return "Cloudy"
    else:
        return "Clear-sky"

df_day["Weather_Type"] = df_day["CSI"].apply(classify_weather)

# 可选：增加一列划分依据，方便在 CSV 文件中直接看到每个样本为什么被分到该类
df_day["Classification_Basis"] = df_day["Weather_Type"].map(weather_basis)

# =========================
# 3. 统计数量和占比
# =========================
weather_order = ["Clear-sky", "Cloudy", "Rainy"]

weather_count = df_day["Weather_Type"].value_counts().reindex(weather_order, fill_value=0)

weather_ratio = weather_count / len(df_day) * 100

result = pd.DataFrame({
    "Classification_Basis": [weather_basis[w] for w in weather_order],
    "Sample_Number": weather_count,
    "Ratio(%)": weather_ratio.round(2)
}, index=weather_order)

print(result)
print("Total daytime samples:", len(df_day))

statistics_path = os.path.join(save_dir, "weather_type_statistics.csv")
result.to_csv(statistics_path, encoding="utf-8-sig")

# =========================
# 4. 分别提取三类气象样本并保存
# =========================
clear_sky_df = df_day[df_day["Weather_Type"] == "Clear-sky"].copy()
cloudy_df = df_day[df_day["Weather_Type"] == "Cloudy"].copy()
rainy_df = df_day[df_day["Weather_Type"] == "Rainy"].copy()

clear_sky_path = os.path.join(save_dir, "clear_sky_samples.csv")
cloudy_path = os.path.join(save_dir, "cloudy_samples.csv")
rainy_path = os.path.join(save_dir, "rainy_samples.csv")

clear_sky_df.to_csv(clear_sky_path, index=False, encoding="utf-8-sig")
cloudy_df.to_csv(cloudy_path, index=False, encoding="utf-8-sig")
rainy_df.to_csv(rainy_path, index=False, encoding="utf-8-sig")

print("Saved files:")
print("Clear-sky:", clear_sky_path, len(clear_sky_df), "| Basis:", weather_basis["Clear-sky"])
print("Cloudy:", cloudy_path, len(cloudy_df), "| Basis:", weather_basis["Cloudy"])
print("Rainy:", rainy_path, len(rainy_df), "| Basis:", weather_basis["Rainy"])
print("Statistics:", statistics_path)
print("Classification basis:", basis_path)