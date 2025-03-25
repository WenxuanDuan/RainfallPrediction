import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# 创建图表保存目录
output_dir = 'eda_outputs'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv('data/hongkong.csv')

# ===== Step 1: 缺失值统计 =====
missing_values = df.isnull().sum()
missing_values.to_csv(f"{output_dir}/missing_values.csv")
print("✔ 缺失值统计已保存：missing_values.csv")

# ===== Step 2: 查看 object 类型列的唯一值 =====
object_summary = {}
for col in df.select_dtypes(include='object').columns:
    unique_vals = df[col].unique()
    object_summary[col] = unique_vals

# 保存为 txt
with open(f"{output_dir}/object_column_values.txt", "w") as f:
    for col, vals in object_summary.items():
        f.write(f"Unique values in '{col}':\n{vals}\n\n")
print("✔ Object 列的唯一值已保存：object_column_values.txt")

# ===== Step 3: 处理 rainfall 列为数值型 + 直方图 =====
rainfall_raw = df['rainfall'].copy()
rainfall_numeric = rainfall_raw.replace({'Minor': 0.1, '-': 0.0}).astype(float)
df['rainfall_numeric'] = rainfall_numeric

# 保存 rainfall 描述性统计信息
rainfall_numeric.describe().to_csv(f"{output_dir}/rainfall_stats.csv")
print("✔ Rainfall 描述性统计已保存：rainfall_stats.csv")

# 画图保存 rainfall 分布图
plt.figure(figsize=(10, 6))
plt.hist(rainfall_numeric, bins=50, color='skyblue', edgecolor='black')
plt.title('Rainfall Distribution')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/rainfall_distribution.png")
plt.close()
print("✔ Rainfall 分布图已保存：rainfall_distribution.png")

# ===== Step 4: 数值特征统计与相关性热力图 =====
numeric_df = df.select_dtypes(include=['int64', 'float64']).copy()
numeric_df['rainfall'] = rainfall_numeric

# 保存所有数值列的描述性统计
numeric_df.describe().to_csv(f"{output_dir}/numeric_summary.csv")
print("✔ 数值特征统计已保存：numeric_summary.csv")

# 热力图
plt.figure(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_heatmap.png")
plt.close()
print("✔ 特征相关性热力图已保存：correlation_heatmap.png")


# 月份 vs 降雨量
# 确保 rainfall_numeric 存在
df['rainfall_numeric'] = df['rainfall_numeric'].astype(float)

# 按月统计平均降雨量
monthly_rain = df.groupby('month')['rainfall_numeric'].mean()

plt.figure(figsize=(10, 6))
monthly_rain.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title('Average Rainfall by Month')
plt.xlabel('Month')
plt.ylabel('Average Rainfall (mm)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(f"{output_dir}/rainfall_by_month.png")
plt.show()

# 云量 vs 降雨量
plt.figure(figsize=(12, 6))
sns.boxplot(x=pd.cut(df['cloud'], bins=5), y='rainfall_numeric', data=df)
plt.title('Rainfall Distribution by Cloud Level')
plt.xlabel('Cloud Level (binned)')
plt.ylabel('Rainfall (mm)')
plt.tight_layout()
plt.savefig(f"{output_dir}/rainfall_by_cloud.png")
plt.show()

# 湿度 vs 降雨量
plt.figure(figsize=(10, 6))
sns.scatterplot(x='humidity', y='rainfall_numeric', data=df, alpha=0.5)
plt.title('Rainfall vs Humidity')
plt.xlabel('Humidity (%)')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{output_dir}/rainfall_vs_humidity.png")
plt.show()


# 降雨等级分布图柱状图
# 自定义降雨等级
def rainfall_level(mm):
    if mm == 0:
        return 'No Rain'
    elif mm <= 2:
        return 'Drizzle'
    elif mm <= 10:
        return 'Light Rain'
    elif mm <= 25:
        return 'Moderate Rain'
    else:
        return 'Heavy Rain'

df['rain_level'] = df['rainfall_numeric'].apply(rainfall_level)

# 画柱状图
plt.figure(figsize=(10, 6))
sns.countplot(x='rain_level', data=df, order=['No Rain', 'Drizzle', 'Light Rain', 'Moderate Rain', 'Heavy Rain'], palette='Blues')
plt.title('Rainfall Level Distribution')
plt.xlabel('Rainfall Level')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(f"{output_dir}/rainfall_level_distribution.png")
plt.show()



