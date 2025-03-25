import pandas as pd
import numpy as np
from data_preprocessing import preprocess_weather_data

from cross_validation import evaluate_model_cv
from sklearn.ensemble import RandomForestClassifier

label_names = ['No Rain', 'Drizzle', 'Light', 'Moderate', 'Heavy']

df = pd.read_csv("data/hongkong.csv")
X, y8, y9, y10 = preprocess_weather_data(df)

results = evaluate_model_cv(
    X, y8,
    model=RandomForestClassifier(random_state=42),
    n_splits=3,
    label_names=label_names,
    output_dir="cv_rf_y8"
)
