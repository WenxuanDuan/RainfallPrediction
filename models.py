import pandas as pd
import numpy as np
from data_preprocessing import preprocess_weather_data_binary
from cross_validation import run_cross_validation
from feature_engineering import preprocess_weather_data_with_features

df = pd.read_csv("data/hongkong.csv")
X, y = preprocess_weather_data_with_features(df)
print(f"X shape: {X.shape}, y shape: {y.shape}")

models = ["XGB", "LOGREG", "RF", "LGBM", "CATBOOST", "MLP"]
for model_name in models:
    run_cross_validation(model_name, X, y)