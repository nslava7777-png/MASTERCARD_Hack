from __future__ import annotations
import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from src.config import CARD_ID_COLUMN, TARGET_COLUMN


def _safe_ratio(num, den, alias):
    return pl.when(den > 0).then(num / den).otherwise(0.0).alias(alias)


def aggregate_card_features(tx, label):
    drop_cols = [c for c in ["source_name", "dt", "ts_raw"] if c in tx.columns]
    if drop_cols:
        tx = tx.drop(drop_cols)

    # 1. Самый частый мерчант
    top_merch = (
        tx.group_by([CARD_ID_COLUMN, "merchant_id"]).agg(pl.len().alias("cnt"))
          .sort([CARD_ID_COLUMN, "cnt"], descending=[False, True])
          .group_by(CARD_ID_COLUMN).agg(pl.first("cnt").alias("top1_cnt"))
    )
    
    # 2. Максимальное повторение одной и той же суммы трат
    per_amt = (
        tx.with_columns(pl.col("amount").round(0).alias("amt_r"))
          .group_by([CARD_ID_COLUMN, "amt_r"]).agg(pl.len().alias("cnt"))
    )
    rep = per_amt.group_by(CARD_ID_COLUMN).agg(
        pl.col("cnt").max().fill_null(1).alias("max_same_amt_count")
    )
    
    # 3. Всплески активности по дням (коэффициент вариации)
    daily = tx.group_by([CARD_ID_COLUMN, "date"]).agg(pl.len().alias("d_cnt"))
    burst = (
        daily.group_by(CARD_ID_COLUMN)
             .agg([
                 pl.col("d_cnt").mean().fill_null(1).alias("dmean"),
                 pl.col("d_cnt").std().fill_null(0).alias("dstd"),
             ])
             .with_columns(_safe_ratio(pl.col("dstd"), pl.col("dmean"), "burst_cv"))
             .drop(["dmean", "dstd"])
    )
    
    # 4. Квантиль 95% для сумм операций
    amt_q95 = tx.group_by(CARD_ID_COLUMN).agg(
        pl.col("amt_abs").quantile(0.95).alias("amt_q95")
    )

    # Дополнительный стабильный признак бизнес-ритма: будни с 9 до 18
    tx = tx.with_columns(
        ((pl.col("dow") < 6) & pl.col("hour").is_between(9, 18)).cast(pl.Int8).alias("f_commercial_hours")
    )

    # 5. Базовые агрегаты по картам
    card = tx.group_by(CARD_ID_COLUMN).agg([
        pl.len().alias("n_txns"),
        pl.col("amt_abs").mean().alias("amt_mean"),
        pl.col("amt_abs").std().fill_null(0).alias("amt_std"),
        pl.col("amt_abs").median().alias("amt_median"),
        pl.col("amt_abs").min().alias("amt_min"),
        pl.col("log_amt").std().fill_null(0).alias("log_amt_std"),
        pl.col("merchant_id").n_unique().alias("n_merchants"),
        pl.col("mcc").n_unique().alias("mcc_diversity"),
        pl.col("country").n_unique().alias("n_countries_raw"),
        pl.col("f_online").mean().alias("online_ratio"),
        pl.col("f_token").mean().alias("token_ratio"),
        pl.col("f_recur").mean().alias("recur_ratio"),
        pl.col("f_night").mean().alias("night_ratio"),
        pl.col("f_weekend").mean().alias("weekend_ratio"),
        pl.col("f_susp_mcc").mean().alias("susp_mcc_ratio"),
        pl.col("f_premium").mean().alias("premium_ratio"),
        pl.col("f_online_night").mean().alias("online_night_ratio"),
        pl.col("hour").mean().alias("hour_mean"),
        pl.col("f_commercial_hours").mean().alias("commercial_hours_ratio"), # Новый признак
    ])
    
    # 6. Расчет относительных коэффициентов (Diversity / Ratio)
    card = card.with_columns([
        _safe_ratio(pl.col("n_merchants"), pl.col("n_txns"),           "merchant_diversity"),
        _safe_ratio(pl.col("n_countries_raw") - 1, pl.col("n_txns"),  "foreign_ratio"),
        _safe_ratio(pl.col("amt_std"), pl.col("amt_mean") + 1e-6,     "amt_cv"),
        _safe_ratio(pl.col("n_txns"), pl.col("n_merchants") + 1e-6,   "txns_per_merchant"),
        _safe_ratio(pl.col("n_merchants"), pl.col("mcc_diversity") + 1e-6, "merchants_per_mcc"), # Отношение мерчантов к MCC
    ]).drop("n_countries_raw")

    # Сборка всех блоков воедино
    card = (
        card.join(top_merch, on=CARD_ID_COLUMN, how="left")
            .join(rep,       on=CARD_ID_COLUMN, how="left")
            .join(burst,     on=CARD_ID_COLUMN, how="left")
            .join(amt_q95,   on=CARD_ID_COLUMN, how="left")
            .fill_null(0)
    )
    card = card.with_columns(
        _safe_ratio(pl.col("top1_cnt"), pl.col("n_txns"), "same_merchant_ratio")
    ).drop("top1_cnt")

    if label is not None:
        card = card.with_columns(pl.lit(label).alias(TARGET_COLUMN))
    return card.sort(CARD_ID_COLUMN)


def build_dataset_features(biz_tx, cons_tx):
    return aggregate_card_features(biz_tx, 1), aggregate_card_features(cons_tx, 0)


# ─────────────────────────────────────────────────────────────────────────────
# ИСПРАВЛЕНИЕ УТЕЧКИ: РАЗДЕЛЕНИЕ СТАДИИ ОБУЧЕНИЯ ЦЕНТРОИД И СКОРИНГА
# ─────────────────────────────────────────────────────────────────────────────

def fit_biz_distance_predictor(X_train_fold, y_train_fold):
    """
    БЕЗОПАСНАЯ ФУНКЦИЯ: Расчет центроид СТРОГО на тренировочном фолде выборки.
    Вызывать внутри цикла кросс-валидации.
    """
    scaler = RobustScaler()
    Xtr_scaled = scaler.fit_transform(X_train_fold)
    
    # Считаем центроиды признаков на основе масштабированных тренировочных данных
    biz_centroid = Xtr_scaled[y_train_fold == 1].mean(axis=0)
    con_centroid = Xtr_scaled[y_train_fold == 0].mean(axis=0)
    
    return scaler, biz_centroid, con_centroid


def compute_biz_distance_score(scaler, biz_centroid, con_centroid, X_score):
    """
    БЕЗОПАСНАЯ ФУНКЦИЯ: Применяет уже готовые центроиды к оцениваемой матрице X_score.
    Утечка исключена, так как X_score не влияет на координаты центроид.
    """
    common = [c for c in X_score.columns]
    
    # Применяем готовый скейлер
    Xsc = scaler.transform(X_score[common])
    
    # Считаем расстояния до эталонов
    diff = np.linalg.norm(Xsc - con_centroid, axis=1) - np.linalg.norm(Xsc - biz_centroid, axis=1)
    
    # Сигмоидальное сглаживание
    score = 1.0 / (1.0 + np.exp(-diff / (diff.std() + 1e-6)))
    return score.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# ИСПРАВЛЕНИЕ БАГА МАСШТАБИРОВАНИЯ В ISOLATION FOREST
# ─────────────────────────────────────────────────────────────────────────────

def compute_isolation_score(X_cons_train, X_cons_score, random_state=42):
    """
    Обучение Isolation Forest с обязательным RobustScaler, чтобы выровнять 
    влияние мелких долей (ratio) и крупных суммарных счетчиков.
    """
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_cons_train)
    X_score_scaled = scaler.transform(X_cons_score)
    
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          random_state=random_state, n_jobs=-1)
    iso.fit(X_train_scaled)
    
    raw = iso.decision_function(X_score_scaled)
    score = -raw
    
    # Мин-макс нормировка итогового скора аномальности
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return score.astype(np.float32)