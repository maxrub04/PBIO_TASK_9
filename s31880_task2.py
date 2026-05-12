"""
Bioinformatics exercise 10 — Task 2: Wisconsin breast tumor classification.

Requires: pandas, seaborn, matplotlib, scikit-learn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


def eval_model(
    name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    png_path: Path,
) -> None:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Accuracy:  {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"Recall:    {recall_score(y_test, pred):.4f}")
    print(f"F1-score:  {f1_score(y_test, pred):.4f}")

    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(name)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {png_path.name}")


def main() -> None:
    base = Path(__file__).resolve().parent

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="diagnosis")

    print("=== WISCONSIN BREAST CANCER — CLASSIFICATION ===")
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {X.shape[1]}")
    print("Class distribution:")
    print(f"  Benign (1): {(y == 1).sum()} ({100 * (y == 1).mean():.1f}%)")
    print(f"  Malignant (0): {(y == 0).sum()} ({100 * (y == 0).mean():.1f}%)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\nData split: {len(X_train)} training, {len(X_test)} test")
    print("Class distribution in training set:")
    print(f"  Benign: {100 * (y_train == 1).mean():.1f}%")
    print(f"  Malignant: {100 * (y_train == 0).mean():.1f}%")
    print("Class distribution in test set:")
    print(f"  Benign: {100 * (y_test == 1).mean():.1f}%")
    print(f"  Malignant: {100 * (y_test == 0).mean():.1f}%")

    eval_model(
        "LOGISTIC REGRESSION",
        LogisticRegression(max_iter=10000),
        X_train,
        X_test,
        y_train,
        y_test,
        base / "confusion_matrix_lr.png",
    )
    eval_model(
        "RANDOM FOREST",
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_train,
        X_test,
        y_train,
        y_test,
        base / "confusion_matrix_rf.png",
    )


if __name__ == "__main__":
    main()
