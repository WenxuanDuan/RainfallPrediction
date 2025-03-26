import pandas as pd
import numpy as np

def preprocess_weather_data_with_features(df, save_to_disk=True):
    """
    Clean and transform raw weather DataFrame into supervised learning format,
    with feature engineering.
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

    # 7. Rainfall level categories (for binary classification: Rain vs No Rain)
    def rainfall_level(mm):
        return 1 if mm > 0 else 0  # 1 = Rain, 0 = No Rain

    df['y_binary'] = df['rainfall_numeric'].apply(rainfall_level)

    # 8. Create new features
    # 8.1. Daily change features (for temperature, pressure, dewpoint)
    df['temp_range'] = df['maxtemp'] - df['mintemp']
    df['pressure_diff'] = df['pressure'].diff().fillna(0)
    df['dewpoint_diff'] = df['dewpoint'].diff().fillna(0)

    # 8.2. Cumulative/Mean features (for past days)
    df['mean_temp_past3'] = df[['temparature']].rolling(window=3).mean().shift(-2)  # Adjusted for past 3 days

    df['cum_sunshine_past7'] = df['sunshine'].rolling(window=7).sum().shift(-6)  # Adjusted for past 7 days
    df['cloud_mean'] = df[['cloud']].rolling(window=7).mean().shift(-6)  # Adjusted for past 7 days

    # 8.3. Boolean features (for continuous days of rain or clouds)
    df['three_day_cloudy'] = (df['cloud'] > 85).rolling(window=3).apply(lambda x: np.all(x), raw=True).shift(-2)
    df['low_evaporation_streak'] = (df['evaporation'] < 0.5).rolling(window=3).apply(lambda x: np.all(x), raw=True).shift(-2)
    df['no_sun_3days'] = (df['sunshine'] == 0).rolling(window=3).apply(lambda x: np.all(x), raw=True).shift(-2)

    # 8.4. Wind-related features
    df['wind_dir_change'] = (df['winddirection'].diff().abs() > 90).fillna(False)
    df['mean_wind_speed_past7'] = df['windspeed'].rolling(window=7).mean().shift(-6)
    df['windgust_event'] = (df['windspeed'] > 30).shift(-1)  # For high gust events

    # 8.5. Interaction features (for combined interactions)
    df['humidity_x_temp'] = df['humidity'] * df['temparature']
    df['cloud_x_sunshine'] = df['cloud'] * (12 - df['sunshine'])
    df['low_visibility_x_humidity'] = df['low visibility hour'] * df['humidity']

    # 9. Select features for X and target for y
    feature_cols = [
        'pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud',
        'low visibility hour', 'sunshine', 'radiation', 'windspeed', 'temp_range', 'pressure_diff',
        'dewpoint_diff', 'mean_temp_past3', 'cum_sunshine_past7', 'cloud_mean', 'three_day_cloudy',
        'low_evaporation_streak', 'no_sun_3days', 'wind_dir_change', 'mean_wind_speed_past7',
        'windgust_event', 'humidity_x_temp', 'cloud_x_sunshine', 'low_visibility_x_humidity'
    ]

    X = df[feature_cols].values
    y = df['y_binary'].values

    # 确保 X 是浮动类型的 NumPy 数组
    X = np.array(X, dtype=np.float64)

    # 检查 X 中是否有 NaN 值
    nan_indices = np.isnan(X).any(axis=1)  # 检查 X 中是否有 NaN 值
    if np.any(nan_indices):  # 如果存在 NaN 值
        print("NaN values found at the following indices in X:")
        print(np.where(nan_indices)[0])  # 打印出包含 NaN 的行索引

        # 选择处理方式：删除包含 NaN 的行
        X = X[~nan_indices]
        y = y[:X.shape[0]]  # 确保 y 的大小与 X 匹配

        # 或者选择填充 NaN（如使用均值填充）
        # X = np.nan_to_num(X, nan=np.nanmean(X))  # 用均值填充 NaN


    # Optionally save to disk
    if save_to_disk:
        pd.DataFrame(X).to_csv("data/processed_X_with_features.csv", index=False)
        pd.DataFrame({'y_binary': y}).to_csv("data/processed_y_binary.csv", index=False)

        # Save as numpy files for faster loading
        np.save("data/processed_X_with_features.npy", X)
        np.save("data/processed_y_binary.npy", y)

    return X, y

# Example usage:
df = pd.read_csv("data/hongkong.csv")
X, y = preprocess_weather_data_with_features(df)
print(f"X shape: {X.shape}, y shape: {y.shape}")
