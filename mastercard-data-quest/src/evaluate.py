"""Model evaluation, metrics, plots, and prediction exports."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import CLASSIFICATION_THRESHOLD


def predict_with_threshold(
    model,
    x: np.ndarray,
    threshold: float = CLASSIFICATION_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Return class predictions and positive-class probabilities."""
    y_proba = positive_class_probability(model, x)
    y_pred = (y_proba >= threshold).astype(int)
    return y_pred, y_proba


def positive_class_probability(model, x: np.ndarray) -> np.ndarray:
    """Extract P(label=1) from a fitted sklearn model or pipeline."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model must support predict_proba for ranking consumers.")

    probabilities = model.predict_proba(x)
    classes = list(getattr(model, "classes_", [0, 1]))
    positive_index = classes.index(1) if 1 in classes else probabilities.shape[1] - 1
    return probabilities[:, positive_index]


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, object]:
    """Calculate classification metrics required by the judging rubric."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["consumer", "business_like"],
            output_dict=True,
            zero_division=0,
        ),
    }

    metrics["roc_auc"] = (
        roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else None
    )
    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """Save a confusion matrix figure for the presentation."""
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["Consumer", "Business-like"])
    ax.set_yticks([0, 1], labels=["Consumer", "Business-like"])

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="black",
                fontsize=12,
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_predictions(
    card_numbers: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    output_path: Path,
) -> None:
    """Save test predictions for ranking and business follow-up."""
    predictions = pl.DataFrame(
        {
            "card_number": card_numbers.astype(str),
            "y_true": y_true,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    ).sort("y_proba", descending=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_csv(output_path)

