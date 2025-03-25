import pandas as pd
import numpy as np
from data_preprocessing import preprocess_weather_data

df = pd.read_csv("data/hongkong.csv")
X, y8, y9, y10 = preprocess_weather_data(df)

print("X shape:", X.shape)
print("y8 class counts:", np.unique(y8, return_counts=True))
print("y9 class counts:", np.unique(y9, return_counts=True))
print("y10 class counts:", np.unique(y10, return_counts=True))

# 路径
X_path = "data/processed_X.csv"
Y_path = "data/processed_Y.csv"

# 读取
X = pd.read_csv(X_path)
Y = pd.read_csv(Y_path)

# ======= 基本信息 =======
print("🔎 X shape:", X.shape)
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

print("\n🧪 Corresponding labels:")
print(Y.iloc[0])
