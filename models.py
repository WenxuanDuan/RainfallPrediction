import pandas as pd
import numpy as np
from cross_validation import run_cross_validation

# 读取数据
X = pd.read_csv("data/processed_X_binary.csv").values
y = pd.read_csv("data/processed_y_binary.csv")["y_binary"].values

# 要测试的模型列表
models = ["XGB", "LOGREG", "RF", "LGBM", "CATBOOST", "MLP"]

# 循环测试
for model_name in models:
    print(f"\n🔍 Testing model: {model_name}")
    run_cross_validation(
        model_name=model_name,
        X=X,
        y=y,
        output_dir="cv_results",
    )
