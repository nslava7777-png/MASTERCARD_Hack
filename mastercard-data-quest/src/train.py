"""Train and compare baseline supervised models."""

from __future__ import annotations

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import CARD_ID_COLUMN, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE


EXCLUDED_FEATURE_COLUMNS = {CARD_ID_COLUMN, TARGET_COLUMN}


def get_feature_columns(card_features: pl.DataFrame) -> list[str]:
    """Return model feature columns, explicitly excluding ID and target."""
    return [
        column
        for column in card_features.columns
        if column not in EXCLUDED_FEATURE_COLUMNS
    ]


def split_train_test(
    card_features: pl.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, object]:
    """Create a stratified card-level train/test split."""
    feature_columns = get_feature_columns(card_features)
    if not feature_columns:
        raise ValueError("No model feature columns were found.")

    x = card_features.select(feature_columns).to_numpy()
    y = card_features.get_column(TARGET_COLUMN).to_numpy().astype(int)
    card_numbers = card_features.get_column(CARD_ID_COLUMN).cast(pl.Utf8).to_numpy()

    unique_classes, class_counts = np.unique(y, return_counts=True)
    can_stratify = len(unique_classes) > 1 and class_counts.min() >= 2
    stratify = y if can_stratify else None

    (
        x_train,
        x_test,
        y_train,
        y_test,
        card_numbers_train,
        card_numbers_test,
    ) = train_test_split(
        x,
        y,
        card_numbers,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return {
        "X_train": x_train,
        "X_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "card_numbers_train": card_numbers_train,
        "card_numbers_test": card_numbers_test,
        "feature_columns": feature_columns,
    }


def build_model_candidates(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Define baseline and stronger classical ML candidates."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_models(
    x_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Fit all candidate models and return trained estimators."""
    models = build_model_candidates(random_state=random_state)
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def select_best_model_name(
    model_metrics: dict[str, dict[str, object]],
    primary_metric: str = "roc_auc",
    fallback_metric: str = "f1",
) -> str:
    """Select the best model by ROC AUC, falling back to F1 if needed."""

    def score(metrics: dict[str, object]) -> float:
        value = metrics.get(primary_metric)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            value = metrics.get(fallback_metric)
        if value is None:
            return float("-inf")
        return float(value)

    return max(model_metrics, key=lambda model_name: score(model_metrics[model_name]))

