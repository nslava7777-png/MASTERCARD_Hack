"""Orchestrate the Mastercard Data Quest card-level ML pipeline."""

from __future__ import annotations

import joblib

from src.config import (
    CLASSIFICATION_THRESHOLD,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    predict_with_threshold,
    save_predictions,
)
from src.explain import get_feature_importance, plot_feature_importance
from src.feature_engineering import build_card_level_features
from src.load_data import load_raw_data
from src.preprocessing import prepare_transactions
from src.train import select_best_model_name, split_train_test, train_models
from src.utils import ensure_directories, save_json


def run_pipeline() -> None:
    """Run data loading, feature engineering, training, evaluation, and exports."""
    ensure_directories()

    print("Loading raw parquet files...")
    raw_data = load_raw_data()

    print("Preparing transactions and joining merchant reference...")
    prepared_transactions = prepare_transactions(
        business_cards=raw_data["business_cards"],
        consumer_cards=raw_data["consumer_cards"],
        merchants_reference=raw_data["merchants_reference"],
    )

    print("Building card-level behavioral features...")
    card_features = build_card_level_features(prepared_transactions)
    card_features_path = PROCESSED_DATA_DIR / "card_level_features.parquet"
    card_features.write_parquet(card_features_path)

    print("Splitting train/test data...")
    split = split_train_test(
        card_features,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    print("Training candidate models...")
    models = train_models(
        x_train=split["X_train"],
        y_train=split["y_train"],
        random_state=RANDOM_STATE,
    )

    model_metrics = {}
    model_predictions = {}
    print("Evaluating models...")
    for model_name, model in models.items():
        y_pred, y_proba = predict_with_threshold(
            model,
            split["X_test"],
            threshold=CLASSIFICATION_THRESHOLD,
        )
        model_metrics[model_name] = calculate_metrics(
            split["y_test"],
            y_pred,
            y_proba,
        )
        model_predictions[model_name] = {
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    best_model_name = select_best_model_name(model_metrics)
    best_model = models[best_model_name]
    best_predictions = model_predictions[best_model_name]

    print(f"Best model: {best_model_name}")
    feature_importance = get_feature_importance(
        best_model,
        split["feature_columns"],
    )

    metrics_payload = {
        "best_model": best_model_name,
        "selection_metric": "roc_auc",
        "threshold": CLASSIFICATION_THRESHOLD,
        "test_size": TEST_SIZE,
        "train_cards": len(split["y_train"]),
        "test_cards": len(split["y_test"]),
        "feature_count": len(split["feature_columns"]),
        "feature_columns": split["feature_columns"],
        "models": model_metrics,
        "top_features": feature_importance.head(15).to_dicts(),
    }

    print("Saving metrics, plots, predictions, and model artifact...")
    save_json(metrics_payload, METRICS_DIR / "metrics.json")
    plot_confusion_matrix(
        split["y_test"],
        best_predictions["y_pred"],
        FIGURES_DIR / "confusion_matrix.png",
    )
    plot_feature_importance(
        feature_importance,
        FIGURES_DIR / "feature_importance.png",
    )
    save_predictions(
        card_numbers=split["card_numbers_test"],
        y_true=split["y_test"],
        y_pred=best_predictions["y_pred"],
        y_proba=best_predictions["y_proba"],
        output_path=PREDICTIONS_DIR / "test_predictions.csv",
    )
    joblib.dump(best_model, MODELS_DIR / "model.pkl")

    print("Pipeline completed successfully.")


if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)

