"""Orchestrate the Mastercard Data Quest card-level ML pipeline."""

from __future__ import annotations

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split as sk_split

from src.config import (
    CARD_ID_COLUMN,
    CLASSIFICATION_THRESHOLD,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    predict_with_threshold,
    save_predictions,
)
from src.explain import get_feature_importance, plot_feature_importance
from src.feature_engineering import build_features_safe
from src.load_data import load_raw_data
from src.preprocessing import prepare_transactions
from src.train import select_best_model_name, train_models
from src.utils import ensure_directories, save_json


# ─── helpers ──────────────────────────────────────────────────────────────────

def _print_section(title: str) -> None:
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")


def _print_feature_list(feature_columns: list[str]) -> None:
    _print_section(f"Feature List  ({len(feature_columns)} features)")
    col_width = max(len(f) for f in feature_columns) + 4
    cols_per_row = max(1, 80 // col_width)
    for i, feat in enumerate(feature_columns, start=1):
        end = "\n" if i % cols_per_row == 0 else ""
        print(f"  {i:>3}. {feat:<{col_width}}", end=end)
    print()


def _print_metrics_table(
    model_metrics: dict[str, dict[str, object]],
    best_model_name: str,
) -> None:
    _print_section("Model Comparison")

    metrics_order = ["roc_auc", "f1", "precision", "recall", "accuracy"]
    header_map = {
        "roc_auc":   "ROC-AUC",
        "f1":        "F1",
        "precision": "Precision",
        "recall":    "Recall",
        "accuracy":  "Accuracy",
    }

    name_w = max(len(n) for n in model_metrics) + 2
    col_w  = 11
    headers = ["Model"] + [header_map[m] for m in metrics_order]
    widths  = [name_w] + [col_w] * len(metrics_order)

    header_line = "  " + "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print(header_line)
    print("  " + "  ".join("─" * w for w in widths))

    for model_name, metrics in model_metrics.items():
        marker = " ★" if model_name == best_model_name else "  "
        row = [f"{model_name}{marker}"]
        for key in metrics_order:
            val = metrics.get(key)
            row.append("N/A" if val is None else f"{float(val):.4f}")
        print("  " + "  ".join(v.ljust(w) for v, w in zip(row, widths)))

    print(f"\n  ★ = best model (selected by ROC-AUC)\n")


def _print_correlation_diagnostics(card_features: pl.DataFrame) -> None:
    """Выводит корреляции признаков с target и предупреждает об утечках."""
    correlations = []
    for col in card_features.columns:
        if col in (CARD_ID_COLUMN, TARGET_COLUMN):
            continue
        corr = card_features.select(pl.corr(col, TARGET_COLUMN)).item()
        correlations.append(
            (col, abs(corr) if (corr is not None and corr == corr) else 0.0)
        )
    correlations.sort(key=lambda x: x[1], reverse=True)

    print("\nКорреляция признаков с label:")
    for feat, corr in correlations[:10]:
        bar = "█" * int(corr * 30)
        flag = "  ⚠️  УТЕЧКА?" if corr > 0.95 else ""
        print(f"  {feat:<35} {corr:.4f}  {bar}{flag}")
    print()


def _prepare_xy(
    train_features: pl.DataFrame,
    test_features: pl.DataFrame,
) -> dict[str, object]:
    """Конвертирует готовые фичи в numpy/pandas для sklearn-моделей."""
    # Все колонки кроме id и target — это признаки
    exclude = {CARD_ID_COLUMN, TARGET_COLUMN}
    feature_columns = [c for c in train_features.columns if c not in exclude]

    if not feature_columns:
        raise ValueError("No feature columns found after excluding id and target.")

    x_train = train_features.select(feature_columns).to_pandas()
    x_test  = test_features.select(feature_columns).to_pandas()

    y_train = train_features.get_column(TARGET_COLUMN).to_numpy().astype(int)
    y_test  = (
        test_features.get_column(TARGET_COLUMN).to_numpy().astype(int)
        if TARGET_COLUMN in test_features.columns
        else np.full(len(test_features), -1, dtype=int)
    )

    card_numbers_train = (
        train_features.get_column(CARD_ID_COLUMN).cast(pl.Utf8).to_numpy()
    )
    card_numbers_test = (
        test_features.get_column(CARD_ID_COLUMN).cast(pl.Utf8).to_numpy()
    )

    return {
        "X_train":            x_train.reset_index(drop=True),
        "X_test":             x_test.reset_index(drop=True),
        "y_train":            y_train,
        "y_test":             y_test,
        "card_numbers_train": card_numbers_train,
        "card_numbers_test":  card_numbers_test,
        "feature_columns":    feature_columns,
    }


# ─── pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    """Run data loading, feature engineering, training, evaluation, and exports."""
    ensure_directories()

    # ── Step 1: Load ─────────────────────────────────────────────
    _print_section("Step 1 — Loading raw parquet files")
    raw_data = load_raw_data()

    # ── Step 2: Prepare transactions ─────────────────────────────
    _print_section("Step 2 — Preparing transactions")
    prepared_transactions = prepare_transactions(
        business_cards=raw_data["business_cards"],
        consumer_cards=raw_data["consumer_cards"],
        merchants_reference=raw_data["merchants_reference"],
    )

    # ── Step 3: Leakage-free train/test split ────────────────────
    # ВАЖНО: split по card_id ДО feature engineering,
    # чтобы mcc_b2b_weights считались только по train-картам
    _print_section("Step 3 — Splitting cards before feature engineering")
    all_card_ids = (
        prepared_transactions
        .get_column(CARD_ID_COLUMN)
        .unique()
        .to_numpy()
    )
    train_ids, test_ids = sk_split(
        all_card_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_ids_list = train_ids.tolist()
    test_ids_list = test_ids.tolist()

    df_train_trans = prepared_transactions.filter(
        pl.col(CARD_ID_COLUMN).is_in(train_ids_list)
    )
    df_test_trans = prepared_transactions.filter(
        pl.col(CARD_ID_COLUMN).is_in(test_ids_list)
    )

    CRASH_TEST_MODE = False  # ← переключи в False когда не нужно

    if CRASH_TEST_MODE and TARGET_COLUMN in df_test_trans.columns:
        df_test_trans = df_test_trans.drop(TARGET_COLUMN)
        print(f"[CRASH TEST] '{TARGET_COLUMN}' удалён из тестовых данных ✓")

    print(f"  Train transactions : {len(df_train_trans):,}")
    print(f"  Test  transactions : {len(df_test_trans):,}")

    # ── Step 4: Feature engineering (no leakage) ─────────────────
    _print_section("Step 4 — Building card-level behavioral features")
    train_features, test_features = build_features_safe(df_train_trans, df_test_trans)

    # Сохраняем для воспроизводимости
    train_features.write_parquet(PROCESSED_DATA_DIR / "train_card_features.parquet")
    test_features.write_parquet(PROCESSED_DATA_DIR  / "test_card_features.parquet")

    # Диагностика корреляций (только по train, чтобы не трогать test)
    _print_correlation_diagnostics(train_features)

    # ── Step 5: Prepare X/y arrays ───────────────────────────────
    _print_section("Step 5 — Preparing model inputs")
    split = _prepare_xy(train_features, test_features)

    _print_feature_list(split["feature_columns"])
    print(f"  Train cards : {len(split['y_train']):,}")
    print(f"  Test  cards : {len(split['y_test']):,}")
    print(f"  Positives   : {split['y_train'].sum():,} train  /  {split['y_test'].sum():,} test")

    # ── Step 6: Train ─────────────────────────────────────────────
    _print_section("Step 6 — Training candidate models")
    models = train_models(
        x_train=split["X_train"],
        y_train=split["y_train"],
        random_state=RANDOM_STATE,
    )
    vals, counts = np.unique(split["y_test"], return_counts=True)
    print(dict(zip(vals, counts)))

    # ── Step 7: Evaluate ──────────────────────────────────────────
    _print_section("Step 7 — Evaluating models")
    model_metrics: dict[str, dict[str, object]] = {}
    model_predictions: dict[str, dict] = {}

    for model_name, model in models.items():
        print(f"  Evaluating {model_name}...")
        y_pred, y_proba = predict_with_threshold(
            model,
            split["X_test"],
            threshold=CLASSIFICATION_THRESHOLD,
        )
        model_metrics[model_name] = calculate_metrics(
            split["y_test"], y_pred, y_proba,
        )
        model_predictions[model_name] = {"y_pred": y_pred, "y_proba": y_proba}

        feature_importance = get_feature_importance(model, split["feature_columns"])
        plot_feature_importance(
            feature_importance,
            FIGURES_DIR / f"feature_importance_{model_name}.png",
        )

    best_model_name = select_best_model_name(model_metrics)
    _print_metrics_table(model_metrics, best_model_name)

    best_model         = models[best_model_name]
    best_predictions   = model_predictions[best_model_name]
    feature_importance = get_feature_importance(best_model, split["feature_columns"])

    _print_section(f"Top-10 Features  [{best_model_name}]")
    for row in feature_importance.head(10).iter_rows(named=True):
        bar = "█" * int(row["importance"] * 40)
        print(f"  {row['feature']:<35} {row['importance']:.4f}  {bar}")

    # ── Step 8: Save artifacts ────────────────────────────────────
    _print_section("Step 8 — Saving artifacts")
    metrics_payload = {
        "best_model":       best_model_name,
        "selection_metric": "roc_auc",
        "threshold":        CLASSIFICATION_THRESHOLD,
        "test_size":        TEST_SIZE,
        "train_cards":      len(split["y_train"]),
        "test_cards":       len(split["y_test"]),
        "feature_count":    len(split["feature_columns"]),
        "feature_columns":  split["feature_columns"],
        "models":           model_metrics,
        "top_features":     feature_importance.head(15).to_dicts(),
    }

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

    _print_section("Pipeline completed successfully ✓")
    print(f"  Best model  : {best_model_name}")
    print(f"  ROC-AUC     : {model_metrics[best_model_name].get('roc_auc', 'N/A'):.4f}")
    print(f"  F1-score    : {model_metrics[best_model_name].get('f1', 'N/A'):.4f}")
    print(f"  Threshold   : {CLASSIFICATION_THRESHOLD}\n")


if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)