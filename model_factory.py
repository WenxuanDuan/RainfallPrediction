# model_factory.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def build_model(model_name):
    if model_name == "LOGREG":
        return LogisticRegression(max_iter=1000)
    elif model_name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "XGB":
        return XGBClassifier(eval_metric='logloss')
    elif model_name == "LGBM":
        return LGBMClassifier()
    elif model_name == "CATBOOST":
        return CatBoostClassifier(verbose=0)
    elif model_name == "MLP":
        return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
