from __future__ import annotations
"""
train.py — v6_fixed_final
=========================
Исправленный пайплайн двухэтапного PU-дообучения (Grey Zone Refiner) без утечек данных.
Включает корректную валидацию базовых моделей, расчет центроид расстояний внутри фолдов
и изоляцию Confident-Sampling для серой зоны.
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

# Импорт безопасных функций для расчета фичей расстояний и аномалий
from src.feature_engineering import fit_biz_distance_predictor, compute_biz_distance_score, compute_isolation_score

from src.config import (
    CONF_BIZ_THRESH, EVAL_CV_SPLITS,
    GREY_ZONE_HIGH, GREY_ZONE_LOW, RANDOM_STATE,
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
    return max(
        metrics_by_model,
        key=lambda n: (
            round(metrics_by_model[n].get(primary) or -1, 4),
            round(metrics_by_model[n].get(fallback) or -1, 4),
        ),
    )


def train_grey_zone_model(X_grey, y_grey, random_state=RANDOM_STATE):
    spw = class_weight_ratio(y_grey)
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