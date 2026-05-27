"""Orchestrate the Mastercard Data Quest card-level PU-Learning pipeline."""

from __future__ import annotations

import joblib
import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from src.config import (
    CARD_ID_COLUMN,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
)
from src.evaluate import calculate_metrics, save_predictions
from src.explain import get_feature_importance, plot_feature_importance
from src.feature_engineering import build_pu_features
from src.load_data import load_raw_data
from src.preprocessing import prepare_transactions
from src.train import train_models
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

def _prepare_xy(train_features: pl.DataFrame) -> dict[str, object]:
    exclude = {CARD_ID_COLUMN, TARGET_COLUMN}
    feature_columns = [c for c in train_features.columns if c not in exclude]

    X = train_features.select(feature_columns).to_pandas()
    y = train_features.get_column(TARGET_COLUMN).to_numpy().astype(int)
    card_numbers = train_features.get_column(CARD_ID_COLUMN).cast(pl.Utf8).to_numpy()

    return {"X": X, "y": y, "card_numbers": card_numbers, "feature_columns": feature_columns}

# ─── pipeline ─────────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    ensure_directories()

    # ── Step 1: Load & Prepare ───────────────────────────────────
    _print_section("Step 1 — Loading & Preparing raw parquet files")
    raw_data = load_raw_data()
    merchants = raw_data["merchants_reference"]

    df_biz_trans = prepare_transactions(raw_data["business_cards"], merchants, "business")
    df_cons_trans = prepare_transactions(raw_data["consumer_cards"], merchants, "consumer")

    print(f"  Business transactions : {len(df_biz_trans):,}")
    print(f"  Consumer transactions : {len(df_cons_trans):,}")

    # ── Step 2: Feature engineering (PU-Learning) ────────────────
    _print_section("Step 2 — Building PU features & MCC weights")
    df_biz_cards, df_cons_cards = build_pu_features(df_biz_trans, df_cons_trans)

    df_biz_cards.write_parquet(PROCESSED_DATA_DIR / "biz_card_features.parquet")
    df_cons_cards.write_parquet(PROCESSED_DATA_DIR / "cons_card_features.parquet")

    # ── Step 3: Combine for Model ────────────────────────────────
    _print_section("Step 3 — Preparing X/y arrays")
    # Обучаем на всём. Бизнес (1), Потребители (0)
    df_train_full = pl.concat([df_biz_cards, df_cons_cards], how="vertical")

    split = _prepare_xy(df_train_full)
    X, y = split["X"], split["y"]

    _print_feature_list(split["feature_columns"])
    print(f"  Total cards for training: {len(y):,}")
    print(f"  Target distribution: 1 (Biz) = {y.sum():,}, 0 (Cons) = {len(y) - y.sum():,}")

    # ── Step 4: Evaluate via Cross-Validation ────────────────────
    _print_section("Step 4 — Cross-Validation (Evaluating performance)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        models_dict = train_models(X_train, y_train, RANDOM_STATE)
        # Берем первую обученную модель (например, LightGBM)
        model_name, model = next(iter(models_dict.items()))

        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]

    cv_roc_auc = roc_auc_score(y, oof_preds)
    print(f"\n  OOF ROC-AUC: {cv_roc_auc:.4f}")

    # ── Step 5: Final Model & Inference ──────────────────────────
    _print_section("Step 5 — Final Training & Scoring Consumer Data")

    # Обучаем финальную модель на ВСЕХ данных
    final_models = train_models(X, y, RANDOM_STATE)
    best_model_name, best_model = next(iter(final_models.items()))

    # Скорим только потребителей (Dataset Y), чтобы найти скрытый бизнес
    X_cons = df_cons_cards.select(split["feature_columns"]).to_pandas()
    y_proba = best_model.predict_proba(X_cons)[:, 1]

    # Сохраняем финальные результаты
    submission_df = df_cons_cards.select([CARD_ID_COLUMN]).with_columns(pl.Series("score", y_proba))
    submission_df = submission_df.sort("score", descending=True)

    submission_path = PREDICTIONS_DIR / "final_submission.csv"
    submission_df.write_csv(submission_path)

    # Сохраняем артефакты
    feature_importance = get_feature_importance(best_model, split["feature_columns"])
    plot_feature_importance(feature_importance, FIGURES_DIR / "feature_importance.png")
    joblib.dump(best_model, MODELS_DIR / "model.pkl")

    save_json({"cv_roc_auc": cv_roc_auc, "best_model": best_model_name}, METRICS_DIR / "metrics.json")

    _print_section("Pipeline completed successfully ✓")
    print(f"  Submission saved to: {submission_path}")
    print(f"  Top-5 potential hidden businesses:")
    print(submission_df.head(15))

if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}")
        raise SystemExit(1)