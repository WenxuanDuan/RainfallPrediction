import os
import numpy as np
import pandas as pd
from feature_engineering import preprocess_weather_data_with_features
from cross_validation import run_cross_validation
from model_factory import build_model

# 1. Load the data and preprocess with feature engineering
print("Loading and preprocessing data...")
df = pd.read_csv("data/hongkong.csv")

# Preprocess the weather data with feature engineering
X, y = preprocess_weather_data_with_features(df)

print(f"Preprocessed data: X shape={X.shape}, y shape={y.shape}")

# 2. Define the models to test
models = ["XGB", "LOGREG", "RF", "CATBOOST", "MLP"]

# 3. Run cross-validation for each model
# for model_name in models:
#     print(f"\nRunning cross-validation for {model_name}...")
#     run_cross_validation(model_name, X, y, output_dir="cv_results")

# 运行Stacking模型的交叉验证
run_cross_validation("STACKING", X, y, output_dir="cv_results", n_splits=5)
