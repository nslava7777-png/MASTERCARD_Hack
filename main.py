from __future__ import annotations
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
"""
Mastercard Hidden Business Detection — v6_fixed
===============================================
PIPELINE (LEAKAGE-FREE):
  1.  Load raw data (3 parquet files)
  2.  Preprocess + add transaction flags (stateless row-by-row)
  3.  Aggregate 25 card-level features (including commercial_hours_ratio)
  4.  Initialize fixed baseline model candidates via build_candidates
  5.  Leakage-Free Full Pipeline OOF CV — StratifiedKFold (5 folds)
      -> Dynamically computes distance, anomaly, and grey-zone refining per fold.
  6.  Model comparison table for composite pipelines
  7.  Fit final composite pipeline on 100% of data for Inference
  8.  Risk tier breakdown & final_submission.csv generation
  9.  8 diagnostic plots based on honest OOF distributions
  10. metrics.json export
"""
import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

from src.config import (
    CARD_ID_COLUMN, EVAL_CV_SPLITS, FIGURES_DIR,
    GREY_ZONE_FEATURES, GREY_ZONE_HIGH, GREY_ZONE_LOW,
    CONF_BIZ_THRESH, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR,
    PROCESSED_DATA_DIR, RANDOM_STATE, TARGET_COLUMN,
)
from src.evaluate import (
    calculate_metrics,
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
    fit_biz_distance_predictor,
    compute_biz_distance_score,
    compute_isolation_score,
)
from src.load_data import load_raw_data
from src.preprocessing import add_txn_flags, prepare_transactions
from src.train import (
    build_candidates,
    select_best_model_name,
    train_grey_zone_model,
)
from src.utils import ensure_directories, save_json

SEP = "=" * 80


def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def print_metrics_table(metrics_by_model):
    section("Model comparison (sorted: ROC-AUC > F1)")
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
    p = lambda q: float(np.percentile(scores, q)) if len(scores) > 0 else 0.0
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


# ─────────────────────────────────────────────────────────────────────────────
# ОРКЕСТРАТОР ПОЛНОЙ КРОСС-ВАЛИДАЦИИ ВСЕГО ПАЙПЛАЙНА БЕЗ УТЕЧЕК
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_composite_pipeline_oof(best_base_model, X, y, grey_features, random_state):
    """
    Честный расчет Out-of-Fold предсказаний для всей сквозной системы.
    Расстояния, Isolation Forest и Серая модель обучаются строго изолированно по фолдам.
    """
    cv = StratifiedKFold(n_splits=EVAL_CV_SPLITS, shuffle=True, random_state=random_state)
    final_oof_scores = np.zeros(len(X))
    
    meta_diagnostics = {
        "base_scores": np.zeros(len(X)),
        "dist_scores": np.zeros(len(X)),
        "iso_scores": np.zeros(len(X)),
        "ensemble_scores": np.zeros(len(X))
    }
    
    avail_features = [f for f in grey_features]
    
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx].copy(), y[tr_idx]
        X_va, y_va = X.iloc[va_idx].copy(), y[va_idx]
        
        # 1. Вычисляем центроиды расстояний строго на обучающем фолде
        scaler, biz_c, con_c = fit_biz_distance_predictor(X_tr, y_tr)
        dist_tr = compute_biz_distance_score(scaler, biz_c, con_c, X_tr)
        dist_va = compute_biz_distance_score(scaler, biz_c, con_c, X_va)
        
        # 2. Обучаем Isolation Forest строго на чистых консьюмерах обучающего фолда
        X_tr_cons = X_tr[y_tr == 0]
        iso_tr = compute_isolation_score(X_tr_cons, X_tr, random_state)
        iso_va = compute_isolation_score(X_tr_cons, X_va, random_state)
        
        # 3. Расчет промежуточного ансамбля фолда (65% расстояние + 35% изоляция)
        ens_tr = np.clip(0.65 * dist_tr + 0.35 * iso_tr, 0.0, 1.0)
        ens_va = np.clip(0.65 * dist_va + 0.35 * iso_va, 0.0, 1.0)
        
        # 4. Обучаем базовый алгоритм (для поиска уверенного бизнеса)
        fold_base = clone(best_base_model)
        fold_base.fit(X_tr, y_tr)
        base_scores_tr = fold_base.predict_proba(X_tr)[:, 1]
        base_scores_va = fold_base.predict_proba(X_va)[:, 1]
        
        # Сохраняем промежуточные фолдовые скоры для диагностики
        meta_diagnostics["base_scores"][va_idx] = base_scores_va
        meta_diagnostics["dist_scores"][va_idx] = dist_va
        meta_diagnostics["iso_scores"][va_idx] = iso_va
        meta_diagnostics["ensemble_scores"][va_idx] = ens_va
        
        # 5. Обогащаем матрицы мета-фичей "biz_distance_score" (она требуется в GREY_ZONE_FEATURES)
        X_tr_enriched = X_tr.copy()
        X_tr_enriched["biz_distance_score"] = dist_tr
        X_va_enriched = X_va.copy()
        X_va_enriched["biz_distance_score"] = dist_va
        
        fold_avail = [f for f in avail_features if f in X_tr_enriched.columns]
        
        # 6. Confident-Sampling: сборка обучающего сета серой зоны текущего фолда
        mask_conf_cons = ens_tr < GREY_ZONE_LOW
        mask_conf_biz  = (y_tr == 1) & (base_scores_tr > CONF_BIZ_THRESH)
        
        X_grey_cons = X_tr_enriched.loc[mask_conf_cons, fold_avail]
        y_grey_cons = np.zeros(len(X_grey_cons), dtype=int)
        
        X_grey_biz = X_tr_enriched.loc[mask_conf_biz, fold_avail]
        y_grey_biz = np.ones(len(X_grey_biz), dtype=int)
        
        X_grey_fold = pd.concat([X_grey_cons, X_grey_biz], ignore_index=True)
        y_grey_fold = np.hstack([y_grey_cons, y_grey_biz])
        
        # Сглаживание серой зоны для валидационной части
        fold_final_scores = ens_va.copy()
        if len(X_grey_fold) > 0 and y_grey_fold.sum() > 0 and (y_grey_fold == 0).sum() > 0:
            fold_grey_model = train_grey_zone_model(X_grey_fold, y_grey_fold, random_state)
            
            mask_grey_va = (ens_va >= GREY_ZONE_LOW) & (ens_va <= GREY_ZONE_HIGH)
            if mask_grey_va.sum() > 0:
                X_grey_pred_va = X_va_enriched.loc[mask_grey_va, fold_avail]
                grey_proba_va = fold_grey_model.predict_proba(X_grey_pred_va)[:, 1]
                fold_final_scores[mask_grey_va] = (0.45 * ens_va[mask_grey_va]) + (0.55 * grey_proba_va)
                
        final_oof_scores[va_idx] = fold_final_scores
        
    return final_oof_scores, meta_diagnostics


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
    section("Step 2 — Feature engineering  (25 card-level features + new rhythm features)")
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

    feat_cols = [c for c in feature_names if c in cons_cards.columns]
    X_cons    = cons_cards.select(feat_cols).to_pandas().astype("float32")
    cons_ids  = cons_cards.get_column(CARD_ID_COLUMN).cast(pl.Utf8).to_numpy()

    # ── Step 3 & 4: Model Candidate Evaluation ────────────────────────────────
    section("Step 3 & 4 — Evaluating Full Composite Pipelines Out-of-Fold")
    
    neg_pos_ratio = float(n_con / max(n_biz, 1))
    candidates = build_candidates(random_state=RANDOM_STATE, spw=neg_pos_ratio)
    tuning_summary = {"info": "Preset optimized architectures loaded from build_candidates"}

    metrics_by_model = {}
    oof_scores_by_model = {}
    meta_oof_by_model = {}

    for name, model_candidate in candidates.items():
        print(f"\n>>> Running full cross-validation loop for: {name}")
        final_oof_scores, meta_oof = evaluate_composite_pipeline_oof(
            model_candidate, X, y, GREY_ZONE_FEATURES, RANDOM_STATE
        )
        oof_scores_by_model[name] = final_oof_scores
        meta_oof_by_model[name] = meta_oof
        metrics_by_model[name] = calculate_metrics(y, (final_oof_scores >= 0.5).astype(int), final_oof_scores)

    print_metrics_table(metrics_by_model)

    # ── Step 5 & 6: Select Best Pipeline ──────────────────────────────────────
    best_name = select_best_model_name(metrics_by_model, "roc_auc", "f1")
    best_base_pipeline = candidates[best_name]
    
    final_oof_scores = oof_scores_by_model[best_name]
    meta_oof = meta_oof_by_model[best_name]
    pipe_metrics = metrics_by_model[best_name]

    section(f"Step 5 & 6 — Best Base Model selected for Composite Pipeline: {best_name}")
    print(f"  [Pipeline OOF Result] ROC-AUC: {pipe_metrics.get('roc_auc', 0):.6f} | F1: {pipe_metrics.get('f1', 0):.4f}")
    score_diagnostics("Composite Pipeline Honest OOF Scores", final_oof_scores, y)

    # ── Step 7: Final Inference Stage (100% Data) ─────────────────────────────
    section("Step 7 — Fitting Final Production Models on ALL training data")
    
    # А) Обучаем финальную базовую модель
    final_base_model = clone(best_base_pipeline)
    final_base_model.fit(X, y)
    joblib.dump(final_base_model, MODELS_DIR / "best_base_model.pkl")
    
    base_scores_all  = final_base_model.predict_proba(X)[:, 1]
    base_scores_cons = final_base_model.predict_proba(X_cons)[:, 1]
    
    # Б) Считаем финальные расстояния и аномалии по всей выборке без деления
    scaler, biz_centroid, con_centroid = fit_biz_distance_predictor(X, y)
    dist_scores_all  = compute_biz_distance_score(scaler, biz_centroid, con_centroid, X)
    dist_scores_cons = compute_biz_distance_score(scaler, biz_centroid, con_centroid, X_cons)
    
    iso_scores_all   = compute_isolation_score(X[y == 0], X, RANDOM_STATE)
    iso_scores_cons  = compute_isolation_score(X[y == 0], X_cons, RANDOM_STATE)
    
    ensemble_scores_all  = np.clip(0.65 * dist_scores_all + 0.35 * iso_scores_all, 0.0, 1.0)
    ensemble_scores_cons = np.clip(0.65 * dist_scores_cons + 0.35 * iso_scores_cons, 0.0, 1.0)
    
    # В) Строим финальную серой зону на основе скоров полной выборки
    X_enriched = X.copy()
    X_enriched["biz_distance_score"] = dist_scores_all
    X_cons_enriched = X_cons.copy()
    X_cons_enriched["biz_distance_score"] = dist_scores_cons
    
    avail_final = [f for f in GREY_ZONE_FEATURES if f in X_enriched.columns]
    
    mask_conf_cons = ensemble_scores_all < GREY_ZONE_LOW
    mask_conf_biz  = (y == 1) & (base_scores_all > CONF_BIZ_THRESH)
    
    X_grey_cons_f = X_enriched.loc[mask_conf_cons, avail_final]
    y_grey_cons_f = np.zeros(len(X_grey_cons_f), dtype=int)
    X_grey_biz_f  = X_enriched.loc[mask_conf_biz, avail_final]
    y_grey_biz_f  = np.ones(len(X_grey_biz_f), dtype=int)
    
    X_grey_final = pd.concat([X_grey_cons_f, X_grey_biz_f], ignore_index=True)
    y_grey_final = np.hstack([y_grey_cons_f, y_grey_biz_f])
    
    final_scores_cons = ensemble_scores_cons.copy()
    
    if len(X_grey_final) > 0 and y_grey_final.sum() > 0 and (y_grey_final == 0).sum() > 0:
        final_grey_model = train_grey_zone_model(X_grey_final, y_grey_final, RANDOM_STATE)
        joblib.dump(final_grey_model, MODELS_DIR / "final_grey_refiner.pkl")
        
        mask_grey_cons = (ensemble_scores_cons >= GREY_ZONE_LOW) & (ensemble_scores_cons <= GREY_ZONE_HIGH)
        if mask_grey_cons.sum() > 0:
            X_grey_pred_cons = X_cons_enriched.loc[mask_grey_cons, avail_final]
            grey_proba_cons = final_grey_model.predict_proba(X_grey_pred_cons)[:, 1]
            final_scores_cons[mask_grey_cons] = (0.45 * ensemble_scores_cons[mask_grey_cons]) + (0.55 * grey_proba_cons)

    # ── Step 8: Risk Tiering ──────────────────────────────────────────────────
    section("Step 8 — Final Consumer Scores & Risk Tiering")
    score_diagnostics("Final Submission Scores (Consumer Pool)", final_scores_cons)

    high   = int((final_scores_cons >= 0.70).sum())
    medium = int(((final_scores_cons >= 0.30) & (final_scores_cons < 0.70)).sum())
    low    = int((final_scores_cons < 0.30).sum())
    print(f"\n  Risk tier breakdown:")
    print(f"    HIGH   (score >= 0.70) : {high:>7,} cards  <- very likely hidden business")
    print(f"    MEDIUM (0.30 – 0.70)   : {medium:>7,} cards  <- grey zone, needs investigation")
    print(f"    LOW    (score  < 0.30) : {low:>7,} cards  <- likely genuine consumer")

    save_predictions(
        cons_ids, final_scores_cons,
        PREDICTIONS_DIR / "final_submission.csv",
        grey_low=0.30, high_thresh=0.70,
    )

    top30 = (pd.DataFrame({"card_number": cons_ids, "score": final_scores_cons})
               .sort_values("score", ascending=False).head(30))
    print(f"\n  Top-30 suspected hidden businesses:")
    print(top30.to_string(index=False))

    # ── Step 9: Diagnostic plots ───────────────────────────────────────────────
    section("Step 9 — Diagnostic plots  (8 charts generated from clean OOF)")
    best_pred = (final_oof_scores >= 0.5).astype(int)

    plot_confusion_matrix(
        y, best_pred,
        FIGURES_DIR / "01_confusion_matrix.png",
        f"Confusion Matrix — Composite Pipeline OOF",
    )
    plot_score_distribution(
        {"business (OOF)": final_oof_scores[y == 1], "consumer (OOF)": final_oof_scores[y == 0]},
        FIGURES_DIR / "02_oof_score_dist.png",
        f"Honest OOF Score Distribution — Composite Pipeline",
    )
    
    custom_oof_payload = {best_name: final_oof_scores}
    roc_data, pr_data = make_curve_payloads(y, custom_oof_payload)
    plot_roc_curves(roc_data, FIGURES_DIR / "03_roc_curves.png")
    plot_precision_recall_curves(pr_data, FIGURES_DIR / "04_pr_curves.png")
    
    fi = feature_importance_df(final_base_model, feature_names)
    plot_feature_importance(fi, FIGURES_DIR / "05_feature_importance.png", top_n=25)
    
    plot_grey_zone_analysis(
        meta_oof["ensemble_scores"], final_oof_scores,
        GREY_ZONE_LOW, GREY_ZONE_HIGH,
        FIGURES_DIR / "06_grey_zone_analysis.png",
    )
    plot_consumer_score_breakdown(
        meta_oof["base_scores"], meta_oof["dist_scores"], meta_oof["iso_scores"], final_oof_scores,
        FIGURES_DIR / "07_consumer_score_breakdown.png",
    )
    plot_top_suspicious_cards(
        cons_ids, final_scores_cons,
        FIGURES_DIR / "08_top50_suspicious.png", top_n=50,
    )
    print(f"  Plots saved successfully to: {FIGURES_DIR}")

    # ── Step 10: Save metrics JSON ─────────────────────────────────────────────
    section("Step 10 — Save metrics JSON")
    save_json({
        "best_model":       best_name,
        "metrics_by_model": metrics_by_model,
        "pipeline_oof":     pipe_metrics,
        "tuning_summary":   tuning_summary,
        "ensemble":   {"weights": {"dist": 0.65, "iso": 0.35}},
        "grey_zone": {
            "method":                  "confident_sampling_refiner_v6_fixed",
            "range":                   [GREY_ZONE_LOW, GREY_ZONE_HIGH],
            "conf_biz_threshold":      CONF_BIZ_THRESH,
            "conf_cons_threshold":     GREY_ZONE_LOW,
            "blend_weight_grey_model": 0.55,
            "blend_weight_ensemble":   0.45,
            "subtle_features":         GREY_ZONE_FEATURES,
        },
        "features": feature_names,
        "n_features": len(feature_names),
        "n_biz": n_biz,
        "n_con": n_con,
    }, METRICS_DIR / "metrics.json")
    print(f"  Saved: {METRICS_DIR / 'metrics.json'}")

    # ── Step 11: MCC Analysis (Ваша вставка) ───────────────────────────────────
    section("Step 11 — Export detailed MCC analysis for High Risk")
    
    # Читаем файл, который мы только что сохранили в Step 8
    # 1. Читаем файл
    high_risk_cards = pd.read_csv(PREDICTIONS_DIR / "final_submission.csv")
    
    # ПРИВЕДЕНИЕ К STRING: конвертируем номера карт в строки
    high_risk_list = high_risk_cards[high_risk_cards["score"] >= 0.7]["card_number"].astype(str).tolist()
    
    if high_risk_list:
        # Теперь Polars сравнит String с String
        risk_details = cons_tx.filter(pl.col("card_number").is_in(high_risk_list))
        
        mcc_report = (
            risk_details.group_by(["card_number", "mcc"])
            .agg(
                pl.len().alias("txn_count"), 
                pl.col("transaction_amount_kzt").sum().alias("total_spent")
            )
            .sort(["card_number", "txn_count"], descending=[False, True])
        )
        
        mcc_report.write_csv(PREDICTIONS_DIR / "high_risk_mcc_details.csv")
        print(f"  [SUCCESS] Report saved: {PREDICTIONS_DIR}/high_risk_mcc_details.csv")
    else:
        print("  [WARNING] No cards found with score >= 0.7, skipping MCC report.")

    section("Pipeline complete successfully")


if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        print("  Make sure all 3 parquet files are in data/raw/")
        raise SystemExit(1)