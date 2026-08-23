import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)

RANDOM_SEED = 42


def split_and_scale(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_logistic_regression(X_train_scaled, y_train):
    """Logistic Regression with hyperparameter tuning via GridSearchCV."""
    param_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["lbfgs"],
        "class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
        param_grid, scoring="accuracy", cv=cv, n_jobs=-1,
    )
    grid.fit(X_train_scaled, y_train)
    print(f"Best Logistic Regression params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def train_random_forest(X_train, y_train):
    """Random Forest with hyperparameter tuning via GridSearchCV."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, None],
        "min_samples_leaf": [2],
        "class_weight": [None, "balanced"],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    grid = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=1),
        param_grid, scoring="accuracy", cv=cv, n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    print(f"Best Random Forest params: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.4f}")
    return grid.best_estimator_


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print(f"\n--- {model_name} Evaluation ---")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k.capitalize():>10}: {v:.4f}")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Default", "Default"],
                yticklabels=["No Default", "Default"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = f"confusion_matrix_{model_name.replace(' ', '_').lower()}.png"
    plt.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close()


def plot_roc_curves(results: list, output_dir: str):
    plt.figure(figsize=(6, 5))
    for name, y_test, y_proba in results:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve_comparison.png"), dpi=150)
    plt.close()


def plot_feature_importance(model, feature_names, output_dir, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(7, 6))
    sns.barplot(x=importances.values, y=importances.index, color="#4C72B0")
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_importance.png"), dpi=150)
    plt.close()


def run_training_pipeline(X: pd.DataFrame, y: pd.Series, output_dir: str, model_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = split_and_scale(X, y)

    # --- Logistic Regression (uses scaled features) ---
    log_reg = train_logistic_regression(X_train_scaled, y_train)
    lr_metrics, lr_pred, lr_proba = evaluate_model(log_reg, X_test_scaled, y_test, "Logistic Regression")
    plot_confusion_matrix(y_test, lr_pred, "Logistic Regression", output_dir)

    # --- Random Forest (tree models don't need scaling) ---
    rf = train_random_forest(X_train, y_train)
    rf_metrics, rf_pred, rf_proba = evaluate_model(rf, X_test, y_test, "Random Forest")
    plot_confusion_matrix(y_test, rf_pred, "Random Forest", output_dir)
    plot_feature_importance(rf, X.columns, output_dir)

    plot_roc_curves(
        [("Logistic Regression", y_test, lr_proba), ("Random Forest", y_test, rf_proba)],
        output_dir,
    )

    results_df = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
    print("\n--- Model Comparison ---")
    print(results_df.round(4))

    # Pick the best model by accuracy and persist it
    best_name = results_df["accuracy"].idxmax()
    best_model = log_reg if best_name == "Logistic Regression" else rf
    best_bundle = {
        "model": best_model,
        "scaler": scaler if best_name == "Logistic Regression" else None,
        "feature_names": list(X.columns),
        "metrics": results_df.loc[best_name].to_dict(),
    }
    joblib.dump(best_bundle, os.path.join(model_dir, "best_model.pkl"))
    print(f"\nBest model: {best_name} (accuracy={results_df.loc[best_name, 'accuracy']:.4f})")
    print(f"Saved to: {os.path.join(model_dir, 'best_model.pkl')}")

    return results_df
