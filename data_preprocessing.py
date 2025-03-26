import pandas as pd
import numpy as np

def preprocess_weather_data_binary(df, save_to_disk=True):
    """
    Clean and transform raw weather DataFrame into supervised learning format for binary classification.
    Labels: 0 = No Rain, 1 = Rain (Drizzle and above)
    Returns:
        X - 2D numpy array of shape (n_samples, 7 * n_features)
        y_binary - Binary label array for day 8
    """
    # 1. Merge date and sort
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Clean rainfall column
    df['rainfall_numeric'] = df['rainfall'].replace({'Minor': 0.1, '-': 0.0}).astype(float)

    # 3. Clean sunshine column
    df['sunshine'] = df['sunshine'].replace('-', 0.0).astype(float)

    # 4. radiation column cleanup
    df['radiation'] = df['radiation'].astype(str).str.extract(r'([\d\.]+)', expand=False)
    df['radiation'] = pd.to_numeric(df['radiation'], errors='coerce')
    df['radiation'] = df['radiation'].fillna(df['radiation'].median())

    # 5. evaporation column cleanup
    df['evaporation'] = df['evaporation'].replace('N.A.', pd.NA)
    df['evaporation'] = pd.to_numeric(df['evaporation'], errors='coerce')
    df['evaporation'] = df['evaporation'].fillna(df['evaporation'].median())

    # 6. Fill missing numeric values
    df['low visibility hour'] = df['low visibility hour'].fillna(df['low visibility hour'].median())
    df['windspeed'] = df['windspeed'].fillna(df['windspeed'].median())

    # 7. Create binary rainfall label
    def binary_label(mm):
        return 0 if mm == 0.0 else 1

    df['rain_binary'] = df['rainfall_numeric'].apply(binary_label)

    # 8. Prepare features and labels (7-day sliding window)
    feature_cols = [
        col for col in df.columns
        if col not in ['year', 'month', 'day', 'date', 'rainfall', 'rainfall_numeric', 'rain_binary']
    ]

    X = []
    y_binary = []

    for i in range(len(df) - 7):
        past_7_days = df.loc[i:i+6, feature_cols].values.flatten()
        label_8 = df.loc[i+7, 'rain_binary']
        X.append(past_7_days)
        y_binary.append(label_8)

    X = np.array(X)
    y_binary = np.array(y_binary)

    if save_to_disk:
        pd.DataFrame(X).to_csv("data/processed_X_binary.csv", index=False)
        pd.DataFrame({'y_binary': y_binary}).to_csv("data/processed_y_binary.csv", index=False)
        np.save("data/processed_X_binary.npy", X)
        np.save("data/processed_y_binary.npy", y_binary)
        print("✅ 已保存处理后的二分类数据为 CSV 和 NPY 文件")

    return X, y_binary