import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from feature_selection import select_features

# 加载数据
X = pd.read_csv("data/processed_X.csv").values
y = pd.read_csv("data/processed_Y.csv")["y8"].values

# 可选：列名用于输出可读特征名
# feature_names = pd.read_csv("data/processed_X.csv").columns.tolist()

# 选择的特征选择方法及其参数
methods = {
    "TopK Importance": {"method": "topk_importance", "top_k": 20},
    "Variance Threshold": {"method": "variance_threshold", "threshold": 0.01},
    "Model Based": {"method": "model_based"},
    "RFE": {"method": "rfe", "n_features": 20}
}

# 用于评估的模型
model = XGBClassifier(n_jobs=-1, verbosity=0, random_state=42)

# 遍历每种特征选择方法
for method_name, params in methods.items():
    print(f"\n📌 Testing feature selection method: {method_name}")

    # 特征选择
    X_sel, sel_names = select_features(X, y, **params)
    print(f"Selected features: {len(sel_names)}")

    # 5折交叉验证评估
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accs, precs, recs, f1s = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_sel, y), 1):
        X_train, X_val = X_sel[train_idx], X_sel[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        accs.append(accuracy_score(y_val, y_pred))
        precs.append(precision_score(y_val, y_pred, average="macro"))
        recs.append(recall_score(y_val, y_pred, average="macro"))
        f1s.append(f1_score(y_val, y_pred, average="macro"))

        print(f"  Fold {fold}: Acc={accs[-1]:.4f}, Prec={precs[-1]:.4f}, Rec={recs[-1]:.4f}, F1={f1s[-1]:.4f}")

    print(f"✅ Avg Accuracy: {np.mean(accs):.4f}")
    print(f"✅ Avg Precision: {np.mean(precs):.4f}")
    print(f"✅ Avg Recall: {np.mean(recs):.4f}")
    print(f"✅ Avg F1: {np.mean(f1s):.4f}")
