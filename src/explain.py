"""Simple model explainability outputs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def _unwrap_estimator(model):
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        return model.named_steps["model"]
    return model


def get_feature_importance(model, feature_columns: list[str]) -> pl.DataFrame:
    """Return feature importance for tree models or absolute coefficients."""
    estimator = _unwrap_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        importance = estimator.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(estimator, "coef_"):
        importance = np.abs(estimator.coef_).ravel()
        importance_type = "absolute_coefficient"
    else:
        importance = np.zeros(len(feature_columns))
        importance_type = "not_available"

    return (
        pl.DataFrame(
            {
                "feature": feature_columns,
                "importance": importance,
                "importance_type": [importance_type] * len(feature_columns),
            }
        )
        .sort("importance", descending=True)
        .with_row_index("rank", offset=1)
    )


def plot_feature_importance(
    feature_importance: pl.DataFrame,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Save a top feature importance plot for explainability slides."""
    top_features = feature_importance.head(top_n).sort("importance")

    fig, ax = plt.subplots(figsize=(8, 6))
    if top_features.get_column("importance").sum() == 0:
        ax.text(
            0.5,
            0.5,
            "Built-in feature importance is not available for this model.",
            ha="center",
            va="center",
            wrap=True,
        )
        ax.set_axis_off()
    else:
        ax.barh(
            top_features.get_column("feature").to_list(),
            top_features.get_column("importance").to_list(),
            color="#2f6fed",
        )
        ax.set_title("Top Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

