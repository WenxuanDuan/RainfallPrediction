# model_factory.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier

def build_model(model_name):
    if model_name == "LOGREG":
        return LogisticRegression(C=1, solver='liblinear', max_iter=1000)
    elif model_name == "RF":
        return RandomForestClassifier(
            n_estimators=50, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, random_state=42)
    elif model_name == "XGB":
        return XGBClassifier(
            learning_rate=0.1, max_depth=3, n_estimators=100,
            subsample=1.0, eval_metric='logloss')
    elif model_name == "LGBM":
        return LGBMClassifier(
            learning_rate=0.1, max_depth=3, n_estimators=100, num_leaves=31)
    elif model_name == "CATBOOST":
        return CatBoostClassifier(
            depth=3, iterations=200, learning_rate=0.1, verbose=0)
    elif model_name == "MLP":
        return MLPClassifier(
            activation='tanh', alpha=0.001, hidden_layer_sizes=(50,), max_iter=500, random_state=42)
    elif model_name == "STACKING":
        base_learners = [
            ('xgb', XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0, eval_metric='logloss')),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5,
                                          min_samples_leaf=2, random_state=42)),
            ('catboost', CatBoostClassifier(depth=3, iterations=200, learning_rate=0.1, verbose=0)),
            ('logreg', LogisticRegression(C=1, solver='liblinear', max_iter=1000))
        ]
        meta_model = LogisticRegression(max_iter=1000)
        return StackingClassifier(estimators=base_learners, final_estimator=meta_model)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
