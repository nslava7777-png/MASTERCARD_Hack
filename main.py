from __future__ import annotations
"""
Mastercard Hidden Business Detection — v6
==========================================
PIPELINE:
  1.  Load raw data (3 parquet files)
  2.  Preprocess + add transaction flags
  3.  Aggregate 25 card-level features (leakage-free)
  4.  Hyperparameter tuning — RandomizedSearchCV (n_iter=6, cv=3)
  5.  OOF cross-validation — StratifiedKFold (5 folds)
  6.  Model comparison table + select best model (ROC-AUC > F1)
  7.  Fit final model on all training data
  8.  Ensemble score on consumer_cards (dist 65% + iso 35%)
  9.  Grey zone confident-sampling refiner (subtle features only)
      Method: consumers(score<0.30) + business(OOF>0.80) as training data
      Features: 10 subtle behavioural signals (timing, commercial, payment)
      Blend: 0.45*ensemble + 0.55*grey_model
  10. 8 diagnostic plots
  11. final_submission.csv with risk_tier + metrics.json

DATA LAYOUT:
  data/raw/business_cards_MDQ.parquet
  data/raw/consumer_cards_MDQ.parquet
  data/raw/merchants_reference.parquet
"""
import joblib
import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    CARD_ID_COLUMN, EVAL_CV_SPLITS, FIGURES_DIR,
    GREY_ZONE_FEATURES, GREY_ZONE_HIGH, GREY_ZONE_LOW,
    METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR,
    PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN,
    TUNE_CV_SPLITS, TUNE_N_ITER,
)
from src.evaluate import (
    make_curve_payloads,
    plot_confusion_matrix,
    plot_consumer_score_breakdown,
    plot_feature_importance,
    plot_grey_zone_analysis,
    plot_precision_recall_curves,
    plot_roc_curves,
    plot_score_distribution,
    plot_top_suspicious_cards,
    save_predictions,
)
from src.feature_engineering import (
    build_dataset_features,
    compute_biz_distance_score,
    compute_isolation_score,
)
from src.load_data import load_raw_data
from src.preprocessing import add_txn_flags, prepare_transactions
from src.train import (
    apply_grey_zone_refiner,
    evaluate_tuned_models_oof,
    fit_final_model,
    select_best_model_name,
    tune_all_models,
)
from src.utils import ensure_directories, save_json

SEP = "=" * 80


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def print_metrics_table(metrics_by_model):
    section("Model comparison  (sorted: ROC-AUC > F1)")
    hdr = (
        f"  {'Model':<24} {'ROC-AUC':>9} {'F1':>9} "
        f"{'Avg-Prec':>10} {'Precision':>11} {'Recall':>9} {'Accuracy':>10}"
    )
    print(hdr)
    print("  " + "-" * 86)
    ordered = sorted(
        metrics_by_model.items(),
        key=lambda kv: (
            round(kv[1].get("roc_auc") or -1, 4),
            round(kv[1].get("f1")      or -1, 4),
        ),
        reverse=True,
    )
    for name, m in ordered:
        auc = f"{m['roc_auc']:.4f}"           if m.get("roc_auc")           is not None else "N/A"
        ap  = f"{m['average_precision']:.4f}" if m.get("average_precision") is not None else "N/A"
        print(
            f"  {name:<24} {auc:>9} {m.get('f1', 0):>9.4f} {ap:>10} "
            f"{m.get('precision', 0):>11.4f} {m.get('recall', 0):>9.4f} "
            f"{m.get('accuracy', 0):>10.4f}"
        )


def score_diagnostics(tag, scores, y=None):
    p = lambda q: float(np.percentile(scores, q))
    print(f"  [{tag}]")
    print(
        f"    n={len(scores):,}  min={scores.min():.4f}  max={scores.max():.4f}  "
        f"mean={scores.mean():.4f}  p90={p(90):.4f}  p95={p(95):.4f}  p99={p(99):.4f}"
    )
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print("    " + "  ".join(f">={t:.1f}:{int((scores >= t).sum()):>6}"
                              for t in thresholds))
    if y is not None:
        bs = scores[y == 1]
        cs = scores[y == 0]
        if len(bs):
            print(f"    BIZ  mean={bs.mean():.4f} "
                  f"p10={float(np.percentile(bs, 10)):.4f} "
                  f"p50={float(np.median(bs)):.4f} "
                  f"p90={float(np.percentile(bs, 90)):.4f}")
        if len(cs):
            print(f"    CONS mean={cs.mean():.4f} "
                  f"p90={float(np.percentile(cs, 90)):.4f} "
                  f"p95={float(np.percentile(cs, 95)):.4f} "
                  f"p99={float(np.percentile(cs, 99)):.4f}")


def prepare_xy(df):
    features = [c for c in df.columns
                if c not in {CARD_ID_COLUMN, TARGET_COLUMN}]
    X   = df.select(features).to_pandas().astype("float32")
    y   = df.get_column(TARGET_COLUMN).to_numpy().astype(int)
    ids = df.get_column(CARD_ID_COLUMN).cast(pl.Utf8).to_numpy()
    return X, y, ids, features


def feature_importance_df(model, feature_names):
    inner = model.named_steps.get("model", model)
    if hasattr(inner, "feature_importances_"):
        imp = inner.feature_importances_
    elif hasattr(inner, "coef_"):
        imp = np.abs(np.ravel(inner.coef_))
    else:
        imp = np.zeros(len(feature_names))
    return pd.DataFrame({"feature": feature_names, "importance": imp})


def run_pipeline():
    ensure_directories()

    # ── Step 1: Load ──────────────────────────────────────────────────────────
    section("Step 1 — Load raw data")
    raw     = load_raw_data()
    biz_tx  = add_txn_flags(prepare_transactions(
        raw["business_cards"], raw["merchants_reference"], "business"))
    cons_tx = add_txn_flags(prepare_transactions(
        raw["consumer_cards"], raw["merchants_reference"], "consumer"))
    print(f"  Business transactions : {biz_tx.height:,}")
    print(f"  Consumer transactions : {cons_tx.height:,}")

    # ── Step 2: Feature engineering ───────────────────────────────────────────
    section("Step 2 — Feature engineering  (25 card-level features)")
    biz_cards, cons_cards = build_dataset_features(biz_tx, cons_tx)
    biz_cards.write_parquet(PROCESSED_DATA_DIR  / "biz_card_features.parquet")
    cons_cards.write_parquet(PROCESSED_DATA_DIR / "cons_card_features.parquet")

    train_df = pl.concat([biz_cards, cons_cards], how="vertical")
    X, y, _, feature_names = prepare_xy(train_df)
    n_biz = int(y.sum())
    n_con = int((y == 0).sum())
    print(f"  Total: {len(y):,}  |  Biz: {n_biz:,}  |  Consumer: {n_con:,}  "
          f"|  Imbalance 1:{n_con // max(n_biz, 1)}")
    print(f"  Features ({len(feature_names)}): {feature_names}")

    # ── Step 3: Hyperparameter tuning ─────────────────────────────────────────
    section(f"Step 3 — Hyperparameter tuning  "
            f"(RandomizedSearchCV: n_iter={TUNE_N_ITER}, cv={TUNE_CV_SPLITS})")
    tuned_models, tuning_summary = tune_all_models(X, y, RANDOM_STATE)

    # ── Step 4: OOF cross-validation ──────────────────────────────────────────
    section(f"Step 4 — OOF cross-validation  ({EVAL_CV_SPLITS} folds)")
    print("  WHY MULTIPLE TRAINING CYCLES?")
    print(f"  {EVAL_CV_SPLITS} folds x {len(tuned_models)} models = "
          f"{EVAL_CV_SPLITS * len(tuned_models)} total training runs.")
    print("  Each sample is validated exactly once — honest, leakage-free evaluation.\n")
    metrics_by_model, oof_proba = evaluate_tuned_models_oof(
        tuned_models, X, y, RANDOM_STATE)
    print_metrics_table(metrics_by_model)

    # ── Step 5: Select best model ─────────────────────────────────────────────
    best_name = select_best_model_name(metrics_by_model, "roc_auc", "f1")
    bm = metrics_by_model[best_name]
    section(
        f"Step 5 — Best model: {best_name}  "
        f"(ROC-AUC={bm['roc_auc']:.4f}  F1={bm['f1']:.4f})"
    )
    print("  Selection rule: highest ROC-AUC; tie-break by F1")
    score_diagnostics(f"{best_name} OOF on TRAIN", oof_proba[best_name], y)

    # ── Step 6: Fit final model ───────────────────────────────────────────────
    section("Step 6 — Fit final model on ALL training data")
    final_model = fit_final_model(best_name, tuned_models, X, y)
    joblib.dump(final_model, MODELS_DIR / "best_base_model.pkl")
    print(f"  Saved: {MODELS_DIR / 'best_base_model.pkl'}")

    # ── Step 7: Ensemble score on consumer_cards ──────────────────────────────
    section("Step 7 — Ensemble score on consumer_cards  (dist 65% + iso 35%)")
    feat_cols    = [c for c in feature_names if c in cons_cards.columns]
    X_cons       = cons_cards.select(feat_cols).to_pandas().astype("float32")
    cons_ids     = cons_cards.get_column(CARD_ID_COLUMN).cast(pl.Utf8).to_numpy()

    X_cons_named = pd.DataFrame(X_cons.values, columns=feat_cols)
    base_scores  = final_model.predict_proba(X_cons_named)[:, 1]
    print("  A) Base model P(biz) [diagnostic only — not in ensemble]:")
    score_diagnostics("base_scores", base_scores)

    dist_scores = compute_biz_distance_score(X, y, X_cons)
    print("\n  B) Distance score (centroid)  [weight=0.65]:")
    score_diagnostics("dist_scores", dist_scores)

    iso_scores = compute_isolation_score(X_cons, X_cons, RANDOM_STATE)
    print("\n  C) Isolation Forest anomaly   [weight=0.35]:")
    score_diagnostics("iso_scores", iso_scores)

    ensemble_scores = np.clip(0.65 * dist_scores + 0.35 * iso_scores, 0.0, 1.0)
    print("\n  D) Ensemble raw (0.65*dist + 0.35*iso):")
    score_diagnostics("ensemble_raw", ensemble_scores)

    # ── Step 8: Grey zone confident-sampling refiner ───────────────────────────
    section("Step 8 — Grey zone refiner  (confident-sampling + subtle features)")
    print(f"  Grey zone range  : [{GREY_ZONE_LOW}, {GREY_ZONE_HIGH}]")
    print(f"  Training data    : consumers(score<{GREY_ZONE_LOW}) + "
          f"business(OOF>0.80)")
    print(f"  Subtle features  : {GREY_ZONE_FEATURES}")
    print(f"  Blend formula    : 0.45*ensemble + 0.55*grey_model")
    print(f"  Rationale        : grey model gets majority weight because it was")
    print(f"                     trained specifically for the ambiguous region\n")

    X_cons_ri = X_cons.reset_index(drop=True)

    final_scores, grey_model = apply_grey_zone_refiner(
        ensemble_scores  = ensemble_scores,
        X_cons           = X_cons_ri,
        X_train          = X.reset_index(drop=True),
        y_train          = y,
        oof_scores_train = oof_proba[best_name],
        grey_features    = GREY_ZONE_FEATURES,
        grey_low         = GREY_ZONE_LOW,
        grey_high        = GREY_ZONE_HIGH,
        blend_weight     = 0.55,
        random_state     = RANDOM_STATE,
    )

    print("\n  Final scores after grey zone refiner:")
    score_diagnostics("final_scores", final_scores)

    high   = int((final_scores >= 0.70).sum())
    medium = int(((final_scores >= 0.30) & (final_scores < 0.70)).sum())
    low    = int((final_scores < 0.30).sum())
    print(f"\n  Risk tier breakdown:")
    print(f"    HIGH   (score >= 0.70) : {high:>7,} cards  <- very likely hidden business")
    print(f"    MEDIUM (0.30 – 0.70)   : {medium:>7,} cards  <- grey zone, needs investigation")
    print(f"    LOW    (score  < 0.30) : {low:>7,} cards  <- likely genuine consumer")

    save_predictions(
        cons_ids, final_scores,
        PREDICTIONS_DIR / "final_submission.csv",
        grey_low=0.30, high_thresh=0.70,
    )

    top30 = (pd.DataFrame({"card_number": cons_ids, "score": final_scores})
               .sort_values("score", ascending=False).head(30))
    print(f"\n  Top-30 suspected hidden businesses:")
    print(top30.to_string(index=False))

    # ── Step 9: Diagnostic plots ───────────────────────────────────────────────
    section("Step 9 — Diagnostic plots  (8 charts)")
    best_oof  = oof_proba[best_name]
    best_pred = (best_oof >= 0.5).astype(int)

    plot_confusion_matrix(
        y, best_pred,
        FIGURES_DIR / "01_confusion_matrix.png",
        f"Confusion Matrix — {best_name} OOF",
    )
    plot_score_distribution(
        {"business (OOF)": best_oof[y == 1], "consumer (OOF)": best_oof[y == 0]},
        FIGURES_DIR / "02_oof_score_dist.png",
        f"OOF Score Distribution — {best_name}",
    )
    roc_data, pr_data = make_curve_payloads(y, oof_proba)
    plot_roc_curves(roc_data, FIGURES_DIR / "03_roc_curves.png")
    plot_precision_recall_curves(pr_data, FIGURES_DIR / "04_pr_curves.png")
    fi = feature_importance_df(final_model, feature_names)
    plot_feature_importance(fi, FIGURES_DIR / "05_feature_importance.png", top_n=25)
    plot_grey_zone_analysis(
        ensemble_scores, final_scores,
        GREY_ZONE_LOW, GREY_ZONE_HIGH,
        FIGURES_DIR / "06_grey_zone_analysis.png",
    )
    plot_consumer_score_breakdown(
        base_scores, dist_scores, iso_scores, final_scores,
        FIGURES_DIR / "07_consumer_score_breakdown.png",
    )
    plot_top_suspicious_cards(
        cons_ids, final_scores,
        FIGURES_DIR / "08_top50_suspicious.png", top_n=50,
    )
    print(f"  01_confusion_matrix.png        — OOF confusion matrix")
    print(f"  02_oof_score_dist.png          — biz vs consumer score separation")
    print(f"  03_roc_curves.png              — all models ROC with AUC")
    print(f"  04_pr_curves.png               — all models Precision-Recall")
    print(f"  05_feature_importance.png      — top-25 features")
    print(f"  06_grey_zone_analysis.png      — before/after/delta grey zone")
    print(f"  07_consumer_score_breakdown.png — 4 score components")
    print(f"  08_top50_suspicious.png        — top-50 suspicious cards bar chart")
    print(f"  Saved to: {FIGURES_DIR}")

    # ── Step 10: Save metrics JSON ─────────────────────────────────────────────
    section("Step 10 — Save metrics JSON")
    save_json({
        "best_model":       best_name,
        "metrics_by_model": metrics_by_model,
        "tuning_summary":   tuning_summary,
        "ensemble":  {"weights": {"dist": 0.65, "iso": 0.35}},
        "grey_zone": {
            "method":                   "confident_sampling_refiner_v6",
            "range":                    [GREY_ZONE_LOW, GREY_ZONE_HIGH],
            "conf_biz_threshold":       0.80,
            "conf_cons_threshold":      GREY_ZONE_LOW,
            "blend_weight_grey_model":  0.55,
            "blend_weight_ensemble":    0.45,
            "subtle_features":          GREY_ZONE_FEATURES,
            "why_subtle_features": (
                "Primary signals (token_ratio, online_ratio, susp_mcc_ratio, "
                "amt_mean, n_txns) are already exploited by the base ensemble. "
                "Grey zone model focuses on timing (hour_mean, burst_cv, weekend_ratio), "
                "commercial structure (same_merchant_ratio, txns_per_merchant, "
                "merchant_diversity, mcc_diversity), and payment regularity "
                "(recur_ratio, max_same_amt_count, amt_cv)."
            ),
        },
        "features": feature_names,
        "n_features": len(feature_names),
        "n_biz": n_biz,
        "n_con": n_con,
    }, METRICS_DIR / "metrics.json")
    print(f"  Saved: {METRICS_DIR / 'metrics.json'}")

    # ── Summary ───────────────────────────────────────────────────────────────
    section("Pipeline complete")
    print(f"  Best model      : {best_name}")
    print(f"  ROC-AUC OOF     : {bm['roc_auc']:.4f}  |  F1: {bm['f1']:.4f}")
    print(f"  Features used   : {len(feature_names)}")
    print(f"  Submission      : {PREDICTIONS_DIR / 'final_submission.csv'}")
    print(f"  HIGH   (>=0.70) : {high:,} cards")
    print(f"  MEDIUM (0.30-0.70): {medium:,} cards")
    print(f"  LOW    (<0.30)  : {low:,} cards")


if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("  Make sure all 3 parquet files are in data/raw/")
        raise SystemExit(1)
