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


def build_grey_zone_training_data(
    X_train_fold, y_train_fold, base_scores_train_fold,
    X_cons, base_scores_cons_fold,
    grey_features,
    conf_biz_thresh=CONF_BIZ_THRESH,
    grey_low=GREY_ZONE_LOW,
):
    """
    БЕЗ УТЕЧЕК: Сборка обучающего сета для рефайнера внутри конкретного фолда.
    Использует только тонкие поведенческие признаки (включая commercial_hours_ratio).
    """
    avail = [f for f in grey_features if f in X_cons.columns]
    
    # Label 0: Надежные потребители из неразмеченных данных по оценке текущего фолда базовой модели
    mask_conf_cons = base_scores_cons_fold < grey_low
    X_conf_cons    = X_cons.loc[mask_conf_cons, avail].copy()
    y_conf_cons    = np.zeros(int(mask_conf_cons.sum()), dtype=int)

    # Label 1: Надежные бизнес-карты из текущего обучающего фолда
    biz_mask      = y_train_fold == 1
    biz_scores    = base_scores_train_fold[biz_mask]
    mask_conf_biz = biz_scores > conf_biz_thresh
    X_conf_biz    = X_train_fold.loc[biz_mask, avail].iloc[np.where(mask_conf_biz)[0]].copy()
    y_conf_biz    = np.ones(int(mask_conf_biz.sum()), dtype=int)

    X_grey = pd.concat([X_conf_cons, X_conf_biz], ignore_index=True)
    y_grey = np.hstack([y_conf_cons, y_conf_biz])

    return X_grey, y_grey, avail


# ─────────────────────────────────────────────────────────────────────────────
# ПОЛНЫЙ ЦИКЛ ОЦЕНКИ КАЧЕСТВА (OOF) БЕЗ УТЕЧЕК
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_full_pipeline_oof(base_model_candidate, X_train, y_train, X_cons, grey_features, random_state=RANDOM_STATE):
    """
    Честная кросс-валидация всей системы. 
    Все динамические фичи (расстояния) и дообучение серой зоны изолированы внутри фолдов.
    """
    cv = StratifiedKFold(n_splits=EVAL_CV_SPLITS, shuffle=True, random_state=random_state)
    final_oof_scores = np.zeros(len(X_train))
    
    print("\n[OOF Validation] Запуск кросс-валидации пайплайна...")
    
    for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
        print(f"  --- Фолд {fold_i}/{EVAL_CV_SPLITS} ---")
        
        # Выделяем чистый трейн и чистую валидацию для размеченных карт
        X_tr_f, y_tr_f = X_train.iloc[tr_idx].copy(), y_train[tr_idx]
        X_va_f, y_va_f = X_train.iloc[va_idx].copy(), y_train[va_idx]
        
        # ШАГ 1: Извлекаем фичи расстояний до центроид БЕЗ утечки
        # Обучаем скейлер и вычисляем центроиды СТРОГО на тренировочной части фолда
        scaler, biz_centroid, con_centroid = fit_biz_distance_predictor(X_tr_f, y_tr_f)
        
        # Генерируем мета-признак расстояния для трейна, валидации и неразмеченных данных
        X_tr_f["biz_distance_score"] = compute_biz_distance_score(scaler, biz_centroid, con_centroid, X_tr_f)
        X_va_f["biz_distance_score"] = compute_biz_distance_score(scaler, biz_centroid, con_centroid, X_va_f)
        
        X_cons_fold = X_cons.copy()
        X_cons_fold["biz_distance_score"] = compute_biz_distance_score(scaler, biz_centroid, con_centroid, X_cons_fold)
        
        # ШАГ 2: Обучаем базовую модель на обогащенных признаках тренировочного фолда
        fold_base_model = clone(base_model_candidate)
        fold_base_model.fit(X_tr_f, y_tr_f)
        
        # ШАГ 3: Скоринг для отбора уверенных кандидатов
        base_scores_train_fold = fold_base_model.predict_proba(X_tr_f)[:, 1]
        base_scores_cons_fold  = fold_base_model.predict_proba(X_cons_fold)[:, 1]
        
        # ШАГ 4: Сборка датасета для серой зоны на основе предсказаний ЭТОГО фолда
        X_grey_train, y_grey_train, avail = build_grey_zone_training_data(
            X_train_fold=X_tr_f,
            y_train_fold=y_tr_f,
            base_scores_train_fold=base_scores_train_fold,
            X_cons=X_cons_fold,
            base_scores_cons_fold=base_scores_cons_fold,
            grey_features=grey_features,
            conf_biz_thresh=CONF_BIZ_THRESH,
            grey_low=GREY_ZONE_LOW,
        )
        
        base_scores_val = fold_base_model.predict_proba(X_va_f)[:, 1]
        fold_final_val_scores = base_scores_val.copy()
        
        # ШАГ 5: Обучение серой модели и блендинг (только если выборка сформировалась корректно)
        if len(avail) > 0 and y_grey_train.sum() > 0 and (y_grey_train == 0).sum() > 0:
            fold_grey_model = train_grey_zone_model(X_grey_train, y_grey_train, random_state)
            
            mask_grey_val = (base_scores_val >= GREY_ZONE_LOW) & (base_scores_val <= GREY_ZONE_HIGH)
            if mask_grey_val.sum() > 0:
                X_grey_pred_val = X_va_f.loc[mask_grey_val, avail]
                grey_proba_val = fold_grey_model.predict_proba(X_grey_pred_val)[:, 1]
                
                # Блендинг для пограничных карт на валидации
                fold_final_val_scores[mask_grey_val] = (0.45 * base_scores_val[mask_grey_val]) + (0.55 * grey_proba_val)
        
        final_oof_scores[va_idx] = fold_final_val_scores

    final_metrics = calculate_metrics(y_train, (final_oof_scores >= 0.5).astype(int), final_oof_scores)
    print(f"\n[Итог контроля] Честный стабильный OOF AUC: {final_metrics.get('roc_auc', 0):.6f}")
    return final_metrics, final_oof_scores


# ─────────────────────────────────────────────────────────────────────────────
# СТАДИЯ СУПЕР-ФИНАЛА ДЛЯ ИНФЕРЕНСА (НА ВСЕХ ДАННЫХ)
# ─────────────────────────────────────────────────────────────────────────────

def train_and_apply_final_pipeline(best_base_model, X_train, y_train, X_cons, grey_features, random_state=RANDOM_STATE):
    """
    Финальный расчет предсказаний для неразмеченных данных (X_cons) перед отправкой сабмишена.
    Обучается один раз на полном объеме данных.
    """
    print("\n[Final Inference] Сборка итогового решения на 100% данных...")
    
    X_train_final = X_train.copy()
    X_cons_final  = X_cons.copy()
    
    # 1. Расчет эталонных центроид по всей обучающей выборке
    scaler, biz_centroid, con_centroid = fit_biz_distance_predictor(X_train_final, y_train)
    X_train_final["biz_distance_score"] = compute_biz_distance_score(scaler, biz_centroid, con_centroid, X_train_final)
    X_cons_final["biz_distance_score"]  = compute_biz_distance_score(scaler, biz_centroid, con_centroid, X_cons_final)
    
    # 2. Финальное обучение базовой модели
    final_base_model = clone(best_base_model)
    final_base_model.fit(X_train_final, y_train)
    
    base_scores_train = final_base_model.predict_proba(X_train_final)[:, 1]
    base_scores_cons  = final_base_model.predict_proba(X_cons_final)[:, 1]
    
    # 3. Финальная сборка обучающего сета серой зоны
    X_grey_train, y_grey_train, avail = build_grey_zone_training_data(
        X_train_fold=X_train_final,
        y_train_fold=y_train,
        base_scores_train_fold=base_scores_train,
        X_cons=X_cons_final,
        base_scores_cons_fold=base_scores_cons,
        grey_features=grey_features,
        conf_biz_thresh=CONF_BIZ_THRESH,
        grey_low=GREY_ZONE_LOW,
    )
    
    final_scores_cons = base_scores_cons.copy().astype(np.float64)
    final_grey_model = None
    
    # 4. Применение рефайнера к финальной серой зоне
    if len(avail) > 0 and y_grey_train.sum() > 0 and (y_grey_train == 0).sum() > 0:
        final_grey_model = train_grey_zone_model(X_grey_train, y_grey_train, random_state)
        
        mask_grey = (base_scores_cons >= GREY_ZONE_LOW) & (base_scores_cons <= GREY_ZONE_HIGH)
        if mask_grey.sum() > 0:
            X_grey_pred = X_cons_final.loc[mask_grey, avail]
            grey_proba = final_grey_model.predict_proba(X_grey_pred)[:, 1]
            
            final_scores_cons[mask_grey] = (0.45 * base_scores_cons[mask_grey]) + (0.55 * grey_proba)
            
    return np.clip(final_scores_cons, 0.0, 1.0).astype(np.float32), final_grey_model