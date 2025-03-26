import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from model_factory import build_model
import warnings

warnings.filterwarnings("ignore")


def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Average Confusion Matrix")
    plt.savefig(output_path)
    plt.close()


def plot_roc_curve(fpr, tpr, auc, model_name, output_path):
    plt.figure()
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()


def run_cross_validation(model_name, X, y, output_dir="cv_results", n_splits=5):
    os.makedirs(output_dir, exist_ok=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_y_true = []
    all_y_pred = []
    all_y_prob = []
    all_metrics = []
    cms = []

    print(f"\n🔍 Testing model: {model_name}")

    for fold, (train_index, val_index) in enumerate(skf.split(X, y), 1):
        print(f"🔁 Fold {fold}")
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model = build_model(model_name)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else y_pred

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)

        print(f"  Fold {fold}: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}, AUC={auc:.4f}")

        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)
        all_metrics.append((acc, prec, rec, f1, auc))
        cms.append(confusion_matrix(y_val, y_pred))

    # === 平均指标 ===
    metrics_avg = np.mean(all_metrics, axis=0)
    print("✅ Average Metrics Across All Folds:")
    print(f"Accuracy: {metrics_avg[0]:.4f}")
    print(f"Precision: {metrics_avg[1]:.4f}")
    print(f"Recall: {metrics_avg[2]:.4f}")
    print(f"F1: {metrics_avg[3]:.4f}")
    print(f"AUC: {metrics_avg[4]:.4f}")

    # === 混淆矩阵 ===
    cm_avg = np.mean(cms, axis=0).astype(int)
    class_names = ["No Rain", "Rain"]
    cm_path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plot_confusion_matrix(cm_avg, class_names, cm_path)

    # === ROC Curve ===
    fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
    roc_path = os.path.join(output_dir, f"roc_curve_{model_name}.png")
    plot_roc_curve(fpr, tpr, metrics_avg[4], model_name, roc_path)
