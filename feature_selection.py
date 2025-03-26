import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


# ========== 各种特征选择方法 ==========

def topk_feature_importance(X, y, top_k=20):
    """基于XGBoost特征重要性选择Top K特征"""
    model = XGBClassifier(random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    X_selected = X[:, indices]
    selected_names = [f"f{i}" for i in indices]
    return X_selected, selected_names


def variance_threshold_selection(X, threshold=0.01):
    """移除低方差特征"""
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_names = [f"f{i}" for i, keep in enumerate(selector.get_support()) if keep]
    return X_selected, selected_names


def model_based_selection(X, y):
    """使用模型（如随机森林）进行基于重要性的特征选择"""
    model = RandomForestClassifier(random_state=42, n_jobs=-1)
    model.fit(X, y)
    importances = model.feature_importances_
    mask = importances > np.median(importances)
    X_selected = X[:, mask]
    selected_names = [f"f{i}" for i, keep in enumerate(mask) if keep]
    return X_selected, selected_names


def recursive_feature_elimination(X, y, n_features=20):
    """递归特征消除（RFE）"""
    estimator = LogisticRegression(max_iter=1000)
    selector = RFE(estimator, n_features_to_select=n_features)
    X_selected = selector.fit_transform(X, y)
    selected_names = [f"f{i}" for i, keep in enumerate(selector.support_) if keep]
    return X_selected, selected_names


# ========== 封装统一接口 ==========

def select_features(X, y, method="topk_importance", **kwargs):
    """
    参数：
        method: str, 支持 "topk_importance", "variance_threshold", "model_based", "rfe"
        kwargs: 根据不同方法传入，如 top_k=20, threshold=0.01, n_features=30
    返回：
        X_selected, selected_names
    """
    if method == "topk_importance":
        top_k = kwargs.get("top_k", 20)
        return topk_feature_importance(X, y, top_k=top_k)

    elif method == "variance_threshold":
        threshold = kwargs.get("threshold", 0.01)
        return variance_threshold_selection(X, threshold=threshold)

    elif method == "model_based":
        return model_based_selection(X, y)

    elif method == "rfe":
        n_features_to_select = kwargs.get("n_features", 20)
        return recursive_feature_elimination(X, y, n_features=n_features_to_select)

    else:
        raise ValueError(f"Unsupported method: {method}")
