# model_factory.py

from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
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
            ('ada', AdaBoostClassifier(n_estimators=200, learning_rate=1.0)),
            ('logreg', LogisticRegression(C=1, solver='liblinear', max_iter=1000)),
            ('nb', GaussianNB(var_smoothing=1e-07))
        ]
        meta_model = LogisticRegression(C=1, solver='liblinear', max_iter=1000)
        return StackingClassifier(estimators=base_learners, final_estimator=meta_model)

    elif model_name == "GBM":
        return GradientBoostingClassifier(n_estimators=50, learning_rate=0.2, max_depth=3)

    elif model_name == "ADA":
        return AdaBoostClassifier(n_estimators=200, learning_rate=1.0)

    elif model_name == "SVM":
        return SVC(C=0.1, gamma='scale', kernel='linear')  # Support Vector Machine (SVM)

    elif model_name == "KNN":
        return KNeighborsClassifier(n_neighbors=7, weights='uniform', metric='manhattan')

    elif model_name == "NB":
        return GaussianNB(var_smoothing=1e-07)

    elif model_name == "ElasticNet":
        return ElasticNet(alpha=0.1, l1_ratio=0.1)

    elif model_name == "VOTING":
        # Voting Classifier with Logistic Regression, Random Forest, and XGBoost
        models = [
            ('logreg', LogisticRegression(C=1, solver='liblinear', max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5,
                                          min_samples_leaf=2, random_state=42)),
            ('xgb', XGBClassifier(learning_rate=0.1, max_depth=3, n_estimators=100, subsample=1.0, eval_metric='logloss'))
        ]
        return VotingClassifier(estimators=models, voting='soft')

    else:
        raise ValueError(f"Unknown model name: {model_name}")
