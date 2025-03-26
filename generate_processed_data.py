import pandas as pd
import numpy as np
from data_preprocessing import preprocess_weather_data_binary

# 原始数据读取
df = pd.read_csv("data/hongkong.csv")
X, y = preprocess_weather_data_binary(df)

print("X shape:", X.shape)
print("y class counts:", np.unique(y, return_counts=True))

# 路径
X_path = "data/processed_X_binary.csv"
Y_path = "data/processed_y_binary.csv"

# 读取
X = pd.read_csv(X_path)
Y = pd.read_csv(Y_path)

# ======= 基本信息 =======
print("\n🔎 X shape:", X.shape)
print("🔎 Y shape:", Y.shape)
print("\n📌 Columns in Y:", Y.columns.tolist())

# ======= 缺失值检查 =======
print("\n❓ Missing values in X:", X.isnull().sum().sum())
print("❓ Missing values in Y:", Y.isnull().sum().sum())

# ======= 标签分布检查 =======
for col in Y.columns:
    print(f"\n🎯 Label distribution for {col}:")
    print(Y[col].value_counts().sort_index())

# ======= 维度一致性 =======
if len(X) != len(Y):
    print("❌ Mismatch: X and Y have different number of rows!")
else:
    print("✅ Row count match between X and Y")

# ======= 示例样本 =======
print("\n🧪 First row of X:")
print(X.iloc[0, :10])  # 只显示前10个特征

print("\n🧪 Corresponding label:")
print(Y.iloc[0])