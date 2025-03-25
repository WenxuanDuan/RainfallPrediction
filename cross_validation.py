import numpy as np
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model_cv(X, y, model=None, n_splits=3, label_names=None, output_dir="cv_results", verbose=True):
    os.makedirs(output_dir, exist_ok=True)

    fold_size = len(X) // (n_splits + 1)

    all_metrics = []
    all_cm = np.zeros((len(np.unique(y)), len(np.unique(y))), dtype=int)

    if model is None:
        model = RandomForestClassifier(random_state=42)

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end = fold_size * (i + 2)

        X_train, X_val = X[:train_end], X[train_end:val_end]
        y_train, y_val = y[:train_end], y[train_end:val_end]

        clf = clone(model)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        fold_cm = confusion_matrix(y_val, y_pred, labels=np.unique(y))
        all_cm += fold_cm

        metrics = {
            'fold': i + 1,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_macro': f1
        }
        all_metrics.append(metrics)

        # Save confusion matrix plot
        cm_path = os.path.join(output_dir, f"confusion_matrix_fold_{i+1}.png")
        plot_confusion_matrix(fold_cm, label_names, f"Fold {i+1} Confusion Matrix", cm_path)

        if verbose:
            print(f"\n📦 Fold {i+1} Results:")
            print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            print(classification_report(y_val, y_pred, target_names=label_names, digits=3))

    # === 平均指标 ===
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in all_metrics]),
        'precision': np.mean([m['precision'] for m in all_metrics]),
        'recall': np.mean([m['recall'] for m in all_metrics]),
        'f1_macro': np.mean([m['f1_macro'] for m in all_metrics])
    }

    print("\n✅ Average Metrics Across All Folds:")
    for k, v in avg_metrics.items():
        print(f"{k.capitalize()}: {v:.4f}")

    # === 平均 confusion matrix 可视化 ===
    cm_avg_path = os.path.join(output_dir, f"confusion_matrix_avg.png")
    plot_confusion_matrix(all_cm, label_names, "Average Confusion Matrix", cm_avg_path)

    return {
        'fold_metrics': all_metrics,
        'avg_metrics': avg_metrics,
        'confusion_matrix': all_cm
    }
