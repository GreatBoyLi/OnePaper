# import os
# import pandas as pd
#
# csv_path = "../../data/val/validate_quan.csv"
# save_dir = "weather_split"
# os.makedirs(save_dir, exist_ok=True)
#
# df = pd.read_csv(csv_path)
#
# # =========================
# # 1. 剔除夜间样本
# # =========================
# df_day = df[df["Solar_Zenith_Raw"] <= 85].copy()
#
# # 如果 CSI 有缺失值，先删除
# df_day = df_day.dropna(subset=["CSI"])
#
# # =========================
# # 2. 按 CSI 划分气象状态
# # =========================
# def classify_weather(csi):
#     if csi < 0.4:
#         return "Overcast/Shaded"
#     elif csi < 0.8:
#         return "Cloudy"
#     else:
#         return "Clear-sky"
#
# df_day["Weather_Type"] = df_day["CSI"].apply(classify_weather)
#
# # =========================
# # 3. 统计数量和占比
# # =========================
# weather_count = df_day["Weather_Type"].value_counts().reindex(
#     ["Clear-sky", "Cloudy", "Overcast/Shaded"]
# )
#
# weather_ratio = weather_count / len(df_day) * 100
#
# result = pd.DataFrame({
#     "Sample_Number": weather_count,
#     "Ratio(%)": weather_ratio.round(2)
# })
#
# print(result)
# print("Total daytime samples:", len(df_day))
#
# statistics_path = os.path.join(save_dir, "weather_type_statistics.csv")
# result.to_csv(statistics_path, encoding="utf-8-sig")
#
# # =========================
# # 4. 分别提取三类气象样本并保存
# # =========================
# clear_sky_df = df_day[df_day["Weather_Type"] == "Clear-sky"].copy()
# cloudy_df = df_day[df_day["Weather_Type"] == "Cloudy"].copy()
# overcast_df = df_day[df_day["Weather_Type"] == "Overcast/Shaded"].copy()
#
# clear_sky_path = os.path.join(save_dir, "clear_sky_samples.csv")
# cloudy_path = os.path.join(save_dir, "cloudy_samples.csv")
# overcast_path = os.path.join(save_dir, "overcast_shaded_samples.csv")
#
# clear_sky_df.to_csv(clear_sky_path, index=False, encoding="utf-8-sig")
# cloudy_df.to_csv(cloudy_path, index=False, encoding="utf-8-sig")
# overcast_df.to_csv(overcast_path, index=False, encoding="utf-8-sig")
#
# print("Saved files:")
# print("Clear-sky:", clear_sky_path, len(clear_sky_df))
# print("Cloudy:", cloudy_path, len(cloudy_df))
# print("Overcast/Shaded:", overcast_path, len(overcast_df))
# print("Statistics:", statistics_path)
#
a = 1785500
b = a / 1.03
c = b-a
print(b)
print(c)

d = a * 0.3
print(d)