import pandas as pd
import numpy as np
from data_preprocessing import preprocess_weather_data_binary
from cross_validation import run_cross_validation
from feature_engineering import preprocess_weather_data_with_features
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, roc_auc_score
from model_factory import build_model

df = pd.read_csv("data/hongkong.csv")
X, y = preprocess_weather_data_with_features(df)
print(f"X shape: {X.shape}, y shape: {y.shape}")



# Define hyperparameters for different models
param_grid = {
    'XGB': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    },
    'LOGREG': {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    'RF': {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'LGBM': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'num_leaves': [31, 50, 100]
    },
    'CATBOOST': {
        'iterations': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'depth': [3, 5, 7]
    },
    'MLP': {
        'hidden_layer_sizes': [(50,), (64, 32), (128, 64)],
        'max_iter': [500, 1000],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001]
    }
}


# GridSearchCV function to tune hyperparameters
def hyperparameter_tuning(model_name, X, y, param_grid, n_splits=5):
    model = build_model(model_name)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid[model_name],
                               cv=n_splits, scoring=make_scorer(roc_auc_score),
                               verbose=1, n_jobs=-1)

    grid_search.fit(X, y)

    print(f"Best parameters for {model_name}: {grid_search.best_params_}")
    print(f"Best ROC AUC Score for {model_name}: {grid_search.best_score_:.4f}")

    # Get the best model and evaluate on the entire dataset
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X)
    y_prob = best_model.predict_proba(X)[:, 1]

    print(f"Best {model_name} Accuracy: {accuracy_score(y, y_pred):.4f}")
    print(f"Best {model_name} AUC: {roc_auc_score(y, y_prob):.4f}")


# Example usage:
# Assuming X and y are loaded data and labels
# Run hyperparameter tuning for different models
for model_name in ['XGB', 'LOGREG', 'RF']:
    hyperparameter_tuning(model_name, X, y, param_grid)
