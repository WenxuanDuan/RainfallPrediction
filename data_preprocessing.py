import pandas as pd
import numpy as np

def preprocess_weather_data(df, save_to_disk=True):
    """
    Clean and transform raw weather DataFrame into supervised learning format.
    Inputs:
        df - Raw DataFrame with weather features and rainfall info.
    Returns:
        X  - 2D numpy array of shape (n_samples, 7 * n_features)
        y8 - Label array for day 8 rainfall level
        y9 - Label array for day 9 rainfall level
        y10 - Label array for day 10 rainfall level
    """
    # 1. Merge date and sort
    df['date'] = pd.to_datetime(df[['year', 'month', 'day']])
    df = df.sort_values('date').reset_index(drop=True)

    # 2. Clean rainfall column: "Minor" → 0.1, "-" → 0.0
    df['rainfall_numeric'] = df['rainfall'].replace({'Minor': 0.1, '-': 0.0}).astype(float)

    # 3. Clean sunshine
    df['sunshine'] = df['sunshine'].replace('-', 0.0).astype(float)

    # 4. radiation to float (non-numeric removal + conversion)
    df['radiation'] = df['radiation'].astype(str).str.extract(r'([\d\.]+)', expand=False)
    df['radiation'] = pd.to_numeric(df['radiation'], errors='coerce')
    df['radiation'] = df['radiation'].fillna(df['radiation'].median())

    # 5. evaporation: "N.A." → NaN → float → fillna
    df['evaporation'] = df['evaporation'].replace('N.A.', pd.NA)
    df['evaporation'] = pd.to_numeric(df['evaporation'], errors='coerce')
    df['evaporation'] = df['evaporation'].fillna(df['evaporation'].median())

    # 6. Fill missing values for numeric cols
    df['low visibility hour'] = df['low visibility hour'].fillna(df['low visibility hour'].median())
    df['windspeed'] = df['windspeed'].fillna(df['windspeed'].median())

    # 7. Rainfall level categories
    def rainfall_level(mm):
        if mm == 0:
            return 0  # No Rain
        elif mm <= 2:
            return 1  # Drizzle
        elif mm <= 10:
            return 2  # Light Rain
        elif mm <= 25:
            return 3  # Moderate Rain
        else:
            return 4  # Heavy Rain

    df['rain_level'] = df['rainfall_numeric'].apply(rainfall_level)

    # 8. Construct sliding window features
    feature_cols = [
        col for col in df.columns
        if col not in ['year', 'month', 'day', 'date', 'rainfall', 'rainfall_numeric', 'rain_level']
    ]

    X = []
    y8, y9, y10 = [], [], []

    for i in range(len(df) - 9):
        past_7_days = df.loc[i:i+6, feature_cols].values.flatten()
        label_8 = df.loc[i+7, 'rain_level']
        label_9 = df.loc[i+8, 'rain_level']
        label_10 = df.loc[i+9, 'rain_level']

        X.append(past_7_days)
        y8.append(label_8)
        y9.append(label_9)
        y10.append(label_10)


    X = np.array(X)
    y8 = np.array(y8)
    y9 = np.array(y9)
    y10 = np.array(y10)

    # === 选项：保存为 .csv 文件（查看方便）
    if save_to_disk:
        pd.DataFrame(X).to_csv(f"data/processed_X.csv", index=False)
        pd.DataFrame({'y8': y8, 'y9': y9, 'y10': y10}).to_csv(f"data/processed_Y.csv", index=False)

        # 可选保存为 .npy（加载更快）
        np.save(f"data/processed_X.npy", X)
        np.save(f"data/processed_y8.npy", y8)
        np.save(f"data/processed_y9.npy", y9)
        np.save(f"data/processed_y10.npy", y10)

        print(f"✅ 数据已保存为 data/processed_*.csv 和 *.npy 文件")

    return np.array(X), np.array(y8), np.array(y9), np.array(y10)
