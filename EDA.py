import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def plot_feature_distribution(df, output_dir='eda_outputs'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Plot Temperature Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['temparature'], kde=True, color='blue')
    plt.title('Temperature Distribution')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/temperature_distribution.png")
    plt.close()

    # Plot Pressure Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['pressure'], kde=True, color='red')
    plt.title('Pressure Distribution')
    plt.xlabel('Pressure (hPa)')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/pressure_distribution.png")
    plt.close()

    # Plot Cloud Coverage Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['cloud'], kde=True, color='purple')
    plt.title('Cloud Coverage Distribution')
    plt.xlabel('Cloud Coverage (%)')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/cloud_coverage_distribution.png")
    plt.close()

    # Plot Humidity Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['humidity'], kde=True, color='green')
    plt.title('Humidity Distribution')
    plt.xlabel('Humidity (%)')
    plt.ylabel('Frequency')
    plt.savefig(f"{output_dir}/humidity_distribution.png")
    plt.close()



# 创建图表保存目录
output_dir = 'eda_outputs'
os.makedirs(output_dir, exist_ok=True)

# 读取数据
df = pd.read_csv('data/hongkong.csv')
plot_feature_distribution(df)

# ===== Step 1: 缺失值统计 =====
missing_values = df.isnull().sum()
missing_values.to_csv(f"{output_dir}/missing_values.csv")
print("✔ 缺失值统计已保存：missing_values.csv")

# 创建一个函数来统计每列中 'Minor'、'-' 和空缺的数量
def count_special_values(df):
    special_values = ['Minor', '-']  # 需要检查的特殊值
    count_dict = {}

    for col in df.columns:
        col_counts = {value: (df[col] == value).sum() for value in special_values}
        col_counts['NaN'] = df[col].isna().sum()  # 统计NaN值
        count_dict[col] = col_counts

    return count_dict

# 统计每一列的 'Minor'、'-' 和空缺值
special_value_counts = count_special_values(df)

# 将结果转换为 DataFrame 方便查看
special_value_df = pd.DataFrame(special_value_counts).T
print(special_value_df)


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


# feature distribution
# 读取数据（假设已经处理好并保存）
X = np.load("data/processed_X_binary.npy")
y = np.load("data/processed_y_binary.npy")

# 将 X 转换为 DataFrame 以便于列名的使用
# 根据之前的列命名规则，将特征列命名为 feature_0, feature_1, ...
columns = [f"feature_{i}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=columns)

# 设置绘图风格
sns.set(style="whitegrid")

# 创建输出目录
output_dir = 'eda_outputs'
os.makedirs(output_dir, exist_ok=True)


# 4. Rainfall Level (Binary Classification)
plt.figure(figsize=(7, 5))
sns.countplot(x=y, palette='coolwarm')
plt.title('Rainfall Level Distribution')
plt.xlabel('Rainfall')
plt.ylabel('Count')
plt.xticks([0, 1], ['No Rain', 'Rain'])
plt.savefig(os.path.join(output_dir, 'rainfall_level_distribution.png'))
plt.close()
