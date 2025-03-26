from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import numpy as np
import pandas as pd
from feature_engineering import preprocess_weather_data_with_features
import joblib

# Stacking模型的实现
def build_stacking_model():
    # 定义基础模型
    base_learners = [
        ('xgb', XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0, eval_metric='logloss')),
        ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5,
                                      min_samples_leaf=2, random_state=42)),
        ('catboost', CatBoostClassifier(depth=3, iterations=200, learning_rate=0.1, verbose=0)),
        ('logreg', LogisticRegression(C=1, solver='liblinear', max_iter=1000))
    ]

    # 定义元模型（可以是Logistic Regression，ElasticNet等）
    meta_model = LogisticRegression(max_iter=1000)

    # 创建StackingClassifier
    stacking_model = StackingClassifier(estimators=base_learners, final_estimator=meta_model)

    return stacking_model


# 进行Cross-Validation
def run_stacking_cross_validation(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    all_metrics = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"🔁 Fold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # 构建Stacking模型
        model = build_stacking_model()
        model.fit(X_train, y_train)

        joblib.dump(model, 'pkl/stacking_model.pkl')

        # 预测
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]  # 预测概率用于AUC计算

        # 计算指标
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        all_metrics.append((acc, prec, rec, f1, auc))

        print(f"  Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

    # 计算平均指标
    metrics_avg = np.mean(all_metrics, axis=0)
    print("✅ Average Metrics Across All Folds:")
    print(f"Accuracy: {metrics_avg[0]:.4f}")
    print(f"Precision: {metrics_avg[1]:.4f}")
    print(f"Recall: {metrics_avg[2]:.4f}")
    print(f"F1: {metrics_avg[3]:.4f}")
    print(f"AUC: {metrics_avg[4]:.4f}")


print("Loading and preprocessing data...")
df = pd.read_csv("data/hongkong.csv")

# 使用之前处理过的数据
X, y = preprocess_weather_data_with_features(df)  # 确保你使用的X, y已经准备好
run_stacking_cross_validation(X, y)



