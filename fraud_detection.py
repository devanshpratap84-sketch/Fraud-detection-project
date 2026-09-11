"""
Fraud Detection System
Major Project - Credit Card Fraud Detection

Dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place the downloaded file here:
    data/creditcard.csv

Run:
    python fraud_detection.py

The program:
1. Loads and validates the dataset
2. Performs exploratory analysis
3. Splits data into train/validation/test sets
4. Scales features without data leakage
5. Handles class imbalance with class-weighted models
6. Trains Logistic Regression and Random Forest classifiers
7. Tunes the fraud probability threshold using the validation set
8. Evaluates models using Precision, Recall, F1, ROC-AUC and PR-AUC
9. Runs Isolation Forest for anomaly detection
10. Saves plots, metrics, predictions and the best model
"""

from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = Path("data/creditcard.csv")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

# -----------------------------
# Helper functions
# -----------------------------
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Download creditcard.csv from Kaggle and place it in data/."
        )

    df = pd.read_csv(path)

    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not set(df["Class"].dropna().unique()).issubset({0, 1}):
        raise ValueError("Class column must contain only 0 (legitimate) and 1 (fraud).")

    return df


def save_class_distribution(df: pd.DataFrame):
    counts = df["Class"].value_counts().sort_index()
    labels = ["Legitimate", "Fraud"]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Transaction Class Distribution")
    plt.ylabel("Number of Transactions")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "class_distribution.png", dpi=200)
    plt.close()


def save_amount_distribution(df: pd.DataFrame):
    plt.figure(figsize=(8, 5))
    plt.hist(df.loc[df["Class"] == 0, "Amount"], bins=60, alpha=0.65, label="Legitimate")
    plt.hist(df.loc[df["Class"] == 1, "Amount"], bins=60, alpha=0.65, label="Fraud")
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Amount")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "amount_distribution.png", dpi=200)
    plt.close()


def build_models():
    # Standardization is especially useful for Logistic Regression.
    # class_weight='balanced' increases the importance of the minority class.
    logistic = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="liblinear",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    random_forest = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=150,
                    class_weight="balanced_subsample",
                    max_depth=None,
                    min_samples_leaf=2,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Logistic Regression": logistic,
        "Random Forest": random_forest,
    }


def find_best_threshold(y_true, probabilities):
    """
    Select a threshold on VALIDATION data only.
    We maximize F1 so that both precision and recall matter.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.50
    best_f1 = -1

    for threshold in thresholds:
        preds = (probabilities >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)

        if score > best_f1:
            best_f1 = score
            best_threshold = float(threshold)

    return best_threshold, best_f1


def evaluate_model(name, model, X_test, y_test, threshold):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model": name,
        "threshold": threshold,
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
    }

    cm = confusion_matrix(y_test, predictions)
    report = classification_report(
        y_test, predictions, target_names=["Legitimate", "Fraud"], zero_division=0
    )

    print(f"\n{'=' * 65}")
    print(name)
    print(f"{'=' * 65}")
    print(pd.Series(metrics))
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.yticks([0, 1], ["Legitimate", "Fraud"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"{safe_name}_confusion_matrix.png", dpi=200)
    plt.close()

    return metrics, probabilities, predictions


def save_roc_pr_curves(results):
    plt.figure(figsize=(7, 5))

    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result["y_test"], result["probabilities"])
        plt.plot(fpr, tpr, label=f"{name} (AUC={result['metrics']['roc_auc']:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title("ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))

    for name, result in results.items():
        precision, recall, _ = precision_recall_curve(
            result["y_test"], result["probabilities"]
        )
        plt.plot(
            recall,
            precision,
            label=f"{name} (AP={result['metrics']['pr_auc']:.3f})",
        )

    plt.title("Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "precision_recall_curves.png", dpi=200)
    plt.close()


def run_isolation_forest(X_train, X_test, y_test):
    """
    Unsupervised anomaly detection.
    Isolation Forest marks unusual transactions as anomalies.
    The anomaly label is converted to:
        1 = anomaly/fraud alert
        0 = normal
    """
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    fraud_rate = max(float(y_test.mean()), 0.001)

    iso = IsolationForest(
        n_estimators=100,
        contamination=fraud_rate,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train_scaled)

    anomaly_predictions = (iso.predict(X_test_scaled) == -1).astype(int)

    metrics = {
        "model": "Isolation Forest",
        "precision": precision_score(y_test, anomaly_predictions, zero_division=0),
        "recall": recall_score(y_test, anomaly_predictions, zero_division=0),
        "f1": f1_score(y_test, anomaly_predictions, zero_division=0),
    }

    print(f"\n{'=' * 65}")
    print("Isolation Forest (Anomaly Detection)")
    print(f"{'=' * 65}")
    print(pd.Series(metrics))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, anomaly_predictions))

    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "model": iso},
        MODEL_DIR / "isolation_forest.joblib",
    )

    return metrics


def main():
    print("Loading dataset...")
    df = load_data(DATA_PATH)

    print("\nDataset shape:", df.shape)
    print("\nFirst five rows:")
    print(df.head())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head())

    print("\nClass counts:")
    print(df["Class"].value_counts())

    fraud_rate = df["Class"].mean() * 100
    print(f"\nFraud percentage: {fraud_rate:.4f}%")

    # EDA plots
    save_class_distribution(df)
    save_amount_distribution(df)

    # Features and target
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)

    # Train/test split, stratified to preserve the rare fraud class.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Validation split is used only for threshold selection.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    print("\nSplit sizes:")
    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)

    models = build_models()
    results = {}
    thresholds = {}

    # Train supervised models
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        val_probabilities = model.predict_proba(X_val)[:, 1]
        threshold, val_f1 = find_best_threshold(y_val, val_probabilities)

        thresholds[name] = {
            "threshold": threshold,
            "validation_f1": val_f1,
        }

        metrics, test_probabilities, test_predictions = evaluate_model(
            name, model, X_test, y_test, threshold
        )

        results[name] = {
            "model": model,
            "metrics": metrics,
            "probabilities": test_probabilities,
            "predictions": test_predictions,
            "y_test": y_test.to_numpy(),
        }

    # Curves for supervised models
    save_roc_pr_curves(results)

    # Select best supervised model by PR-AUC.
    best_name = max(results, key=lambda k: results[k]["metrics"]["pr_auc"])
    best_model = results[best_name]["model"]
    best_threshold = thresholds[best_name]["threshold"]

    joblib.dump(
        {
            "model": best_model,
            "threshold": best_threshold,
            "feature_columns": list(X.columns),
        },
        MODEL_DIR / "best_fraud_model.joblib",
    )

    # Save test predictions and risk levels.
    best_prob = results[best_name]["probabilities"]
    risk = pd.cut(
        best_prob,
        bins=[-np.inf, 0.20, 0.50, 0.80, np.inf],
        labels=["Low", "Medium", "High", "Critical"],
    )

    prediction_output = X_test.copy()
    prediction_output["Actual_Class"] = y_test.to_numpy()
    prediction_output["Fraud_Probability"] = best_prob
    prediction_output["Risk_Level"] = risk.astype(str)
    prediction_output["Predicted_Class"] = (
        best_prob >= best_threshold
    ).astype(int)

    prediction_output.to_csv(
        OUTPUT_DIR / "fraud_predictions.csv",
        index=False,
    )

    # Unsupervised anomaly detection
    isolation_metrics = run_isolation_forest(X_train, X_test, y_test)

    # Save metrics
    all_metrics = {
        "dataset_rows": int(len(df)),
        "dataset_columns": int(df.shape[1]),
        "fraud_transactions": int(y.sum()),
        "legitimate_transactions": int((y == 0).sum()),
        "fraud_percentage": float(fraud_rate),
        "best_supervised_model": best_name,
        "supervised_models": {
            name: result["metrics"] for name, result in results.items()
        },
        "threshold_selection": thresholds,
        "isolation_forest": isolation_metrics,
    }

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"\nBest supervised model: {best_name}")
    print(f"Best threshold: {best_threshold:.2f}")
    print("\nFiles saved in the outputs/ and models/ folders.")
    print("Project completed successfully.")


if __name__ == "__main__":
    main()
