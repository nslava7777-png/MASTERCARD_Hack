from __future__ import annotations
"""
train.py — v6
=============
Grey Zone Refiner (v6) — Confident-Sampling approach
------------------------------------------------------
PROBLEM with v5:
  Linear rescaling only changed score values, NOT the ranking.
  Card at 0.50 became 0.67 — still ambiguous. No real separation.

SOLUTION (v6):
  Build training data from boundary examples on BOTH sides of grey zone:

    Label 0 (confident consumers):
      Consumer cards with ensemble_score < 0.30
      Base ensemble is very certain: these are real consumers.

    Label 1 (confident businesses):
      Business train cards where OOF score > 0.80
      These are the clearest business behavioural patterns.

  Model features: 10 SUBTLE behavioural signals that the base ensemble
  has NOT fully exploited (timing, commercial pattern, payment structure).
  Excludes primary signals: token_ratio, online_ratio, susp_mcc_ratio,
  amt_mean, n_txns — already used up by base ensemble.

  Applied only to grey zone [0.30, 0.70]:
    final = 0.45 * ensemble + 0.55 * grey_score
    Grey model gets majority weight — it was trained specifically
    to discriminate within the ambiguous region.

  Expected outcome:
    Cards bunched at 0.45-0.65 now spread to 0.30-0.80 depending on
    whether subtle patterns say "consumer" or "business".
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import (
    CONF_BIZ_THRESH, EVAL_CV_SPLITS, GREY_ZONE_FEATURES,
    GREY_ZONE_HIGH, GREY_ZONE_LOW, RANDOM_STATE,
    TUNE_CV_SPLITS, TUNE_N_ITER,
)
from src.evaluate import calculate_metrics


class PassThrough(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X): return X


def class_weight_ratio(y):
    neg = int((y == 0).sum())
    pos = int((y == 1).sum())
    return float(neg / pos) if pos > 0 else 1.0


def build_candidates(random_state, spw):
    return {
        "lightgbm": Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("sc",    PassThrough()),
            ("model", LGBMClassifier(
                n_estimators=600, learning_rate=0.03, num_leaves=31,
                min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0, scale_pos_weight=spw,
                random_state=random_state, n_jobs=-1, verbosity=-1,
            )),
        ]),
        "catboost": Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("sc",    PassThrough()),
            ("model", CatBoostClassifier(
                iterations=400, depth=6, learning_rate=0.05,
                l2_leaf_reg=5.0, min_data_in_leaf=10,
                scale_pos_weight=spw,
                random_seed=random_state, verbose=0,
            )),
        ]),
        "logistic_regression": Pipeline([
            ("imp",   SimpleImputer(strategy="median")),
            ("sc",    RobustScaler()),
            ("model", LogisticRegression(
                max_iter=3000, class_weight="balanced",
                C=0.5, random_state=random_state,
            )),
        ]),
    }


def param_grids():
    return {
        "lightgbm": {
            "model__n_estimators":      [400, 600, 800],
            "model__learning_rate":     [0.02, 0.05, 0.08],
            "model__num_leaves":        [15, 31, 63],
            "model__min_child_samples": [10, 20, 40],
            "model__reg_alpha":         [0.0, 0.1, 1.0],
            "model__reg_lambda":        [0.1, 1.0, 5.0],
            "model__colsample_bytree":  [0.6, 0.8, 1.0],
        },
        "catboost": {
            "model__iterations":       [300, 500, 700],
            "model__depth":            [4, 6, 8],
            "model__learning_rate":    [0.02, 0.05, 0.08],
            "model__l2_leaf_reg":      [1, 5, 9],
            "model__min_data_in_leaf": [5, 10, 20],
        },
        "logistic_regression": {
            "model__C":      np.logspace(-3, 1, 8).tolist(),
            "model__solver": ["lbfgs", "liblinear"],
        },
    }


def select_best_model_name(metrics_by_model, primary="roc_auc", fallback="f1"):
    """Select best model by primary metric, tie-break with fallback."""
    return max(
        metrics_by_model,
        key=lambda n: (
            round(metrics_by_model[n].get(primary) or -1, 4),
            round(metrics_by_model[n].get(fallback) or -1, 4),
        ),
    )


def tune_all_models(X, y, random_state=RANDOM_STATE):
    spw = class_weight_ratio(y)
    print(f"  Class ratio (neg/pos) = {spw:.2f}")
    candidates = build_candidates(random_state, spw)
    grids = param_grids()
    cv = StratifiedKFold(n_splits=TUNE_CV_SPLITS, shuffle=True,
                         random_state=random_state)
    tuned, summary = {}, {}
    for name, pipe in candidates.items():
        print(f"    [{name}] RandomizedSearchCV "
              f"(n_iter={TUNE_N_ITER}, cv={TUNE_CV_SPLITS})...")
        search = RandomizedSearchCV(
            pipe, grids[name], n_iter=TUNE_N_ITER, scoring="roc_auc",
            cv=cv, n_jobs=-1, random_state=random_state,
            refit=True, verbose=0,
        )
        search.fit(X, y)
        tuned[name]   = search.best_estimator_
        summary[name] = {
            "best_cv_roc_auc": float(search.best_score_),
            "best_params": search.best_params_,
        }
        print(f"      cv_roc_auc = {search.best_score_:.4f}")
    return tuned, summary


def evaluate_tuned_models_oof(tuned_models, X, y, random_state=RANDOM_STATE):
    """
    5-Fold OOF evaluation for all tuned models.

    WHY MULTIPLE TRAINING CYCLES?
    Each fold trains the model on 4/5 of the data and validates on 1/5.
    With 5 folds x 3 models = 15 training runs total.
    This gives an honest OOF score — every sample is validated exactly
    once, preventing data leakage and providing stable metric estimates.
    """
    cv = StratifiedKFold(n_splits=EVAL_CV_SPLITS, shuffle=True,
                         random_state=random_state)
    oof_proba = {n: np.zeros(len(X)) for n in tuned_models}

    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X, y), 1):
        print(f"  Fold {fold_i}/{EVAL_CV_SPLITS}...")
        for name, tuned in tuned_models.items():
            m = clone(tuned)
            m.fit(X.iloc[tr_idx], y[tr_idx])
            oof_proba[name][va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]

    metrics = {}
    for name, probs in oof_proba.items():
        metrics[name] = calculate_metrics(y, (probs >= 0.5).astype(int), probs)
    return metrics, oof_proba


def fit_final_model(best_name, tuned_models, X, y):
    m = clone(tuned_models[best_name])
    m.fit(X, y)
    return m


# ─────────────────────────────────────────────────────────────────────────────
# GREY ZONE v6 — Confident-Sampling Refiner
# ─────────────────────────────────────────────────────────────────────────────

def build_grey_zone_training_data(
    X_train, y_train, oof_scores_train,
    X_cons, ensemble_scores_cons,
    grey_features,
    conf_biz_thresh=CONF_BIZ_THRESH,
    grey_low=GREY_ZONE_LOW,
):
    """
    Build confident-sampling training data for the grey zone refiner.

    Label 0: Consumer cards with ensemble_score < grey_low (0.30)
      The base ensemble is very certain about these — pure consumers.

    Label 1: Business train cards with OOF score > conf_biz_thresh (0.80)
      These are the clearest business behavioural patterns.

    Only GREY_ZONE_FEATURES (subtle signals) are used.
    """
    avail   = [f for f in grey_features if f in X_cons.columns]
    missing = [f for f in grey_features if f not in X_cons.columns]
    if missing:
        print(f"    [grey zone] WARNING: missing features skipped: {missing}")

    mask_conf_cons = ensemble_scores_cons < grey_low
    X_conf_cons    = X_cons.loc[mask_conf_cons, avail].copy()
    y_conf_cons    = np.zeros(int(mask_conf_cons.sum()), dtype=int)

    biz_mask      = y_train == 1
    biz_oof       = oof_scores_train[biz_mask]
    mask_conf_biz = biz_oof > conf_biz_thresh
    X_conf_biz    = X_train.loc[biz_mask, avail].iloc[np.where(mask_conf_biz)[0]].copy()
    y_conf_biz    = np.ones(int(mask_conf_biz.sum()), dtype=int)

    X_grey = pd.concat([X_conf_cons, X_conf_biz], ignore_index=True)
    y_grey = np.hstack([y_conf_cons, y_conf_biz])

    n_cons = int(mask_conf_cons.sum())
    n_biz  = int(mask_conf_biz.sum())
    ratio  = float(n_cons / max(n_biz, 1))
    print(f"    [grey zone train] confident_consumers={n_cons:,}  "
          f"confident_biz={n_biz:,}  ratio={ratio:.1f}:1")
    print(f"    [grey zone train] using {len(avail)} subtle features: {avail}")
    return X_grey, y_grey, avail


def train_grey_zone_model(X_grey, y_grey, random_state=RANDOM_STATE):
    """
    Train a shallow LightGBM specifically for the grey zone.

    Hyperparameter choices explained:
      num_leaves=15       — shallow trees: subtle signals, not brute-force
      min_child_samples=30 — needs solid evidence per leaf (avoids noise)
      reg_lambda=5.0      — strong regularisation (small training set)
      learning_rate=0.02  — slow, careful learning
      colsample_bytree=0.7 — not all subtle features every tree
      scale_pos_weight    — handles imbalance (many consumers, few biz)
    """
    spw = class_weight_ratio(y_grey)
    print(f"    [grey zone model] scale_pos_weight={spw:.2f}")
    pipe = Pipeline([
        ("imp",   SimpleImputer(strategy="median")),
        ("sc",    RobustScaler()),
        ("model", LGBMClassifier(
            n_estimators=500,
            learning_rate=0.02,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=5.0,
            scale_pos_weight=spw,
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
        )),
    ])
    pipe.fit(X_grey, y_grey)
    return pipe


def apply_grey_zone_refiner(
    ensemble_scores,
    X_cons,
    X_train,
    y_train,
    oof_scores_train,
    grey_features,
    grey_low=GREY_ZONE_LOW,
    grey_high=GREY_ZONE_HIGH,
    blend_weight=0.55,
    random_state=RANDOM_STATE,
):
    """
    Full grey zone confident-sampling refiner pipeline.

    STEPS:
      1. Build training data from confident boundary examples
      2. Train shallow LGBM on 10 subtle features
      3. Predict on grey zone cards [grey_low, grey_high]
      4. Blend: final = (1-w)*ensemble + w*grey_score
         w=0.55 — grey model has majority weight because it was trained
         specifically to discriminate within the ambiguous region

    EXPECTED RESULT:
      Cards bunched at 0.45-0.65 spread out to 0.30-0.80 depending on
      whether subtle behavioural patterns say "consumer" or "business".
    """
    X_grey_train, y_grey_train, avail = build_grey_zone_training_data(
        X_train, y_train, oof_scores_train,
        X_cons, ensemble_scores,
        grey_features, CONF_BIZ_THRESH, grey_low,
    )

    if len(avail) == 0 or y_grey_train.sum() == 0:
        print("    [grey zone] Insufficient data — skipping refiner")
        return ensemble_scores.copy(), None

    grey_model = train_grey_zone_model(X_grey_train, y_grey_train, random_state)

    mask_grey = (ensemble_scores >= grey_low) & (ensemble_scores <= grey_high)
    n_grey    = int(mask_grey.sum())
    print(f"    [grey zone] Refining {n_grey:,} grey zone cards "
          f"[{grey_low}, {grey_high}]...")

    final_scores = ensemble_scores.copy().astype(np.float64)

    if n_grey > 0:
        X_grey_pred = X_cons.loc[mask_grey, avail]
        grey_proba  = grey_model.predict_proba(X_grey_pred)[:, 1]

        blended = ((1 - blend_weight) * ensemble_scores[mask_grey]
                   + blend_weight * grey_proba)
        final_scores[mask_grey] = blended

        before = ensemble_scores[mask_grey]
        after  = blended
        print(f"    [grey zone] Before: mean={before.mean():.3f}  "
              f"std={before.std():.3f}  min={before.min():.3f}  max={before.max():.3f}")
        print(f"    [grey zone] After:  mean={after.mean():.3f}  "
              f"std={after.std():.3f}  min={after.min():.3f}  max={after.max():.3f}")

        delta = after - before
        moved_up   = int((delta >  0.05).sum())
        moved_down = int((delta < -0.05).sum())
        print(f"    [grey zone] Pushed UP   (+0.05): {moved_up:,} cards")
        print(f"    [grey zone] Pushed DOWN (-0.05): {moved_down:,} cards")

    return np.clip(final_scores, 0.0, 1.0).astype(np.float32), grey_model
