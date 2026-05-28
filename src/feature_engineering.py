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

    top_merch = (
        tx.group_by([CARD_ID_COLUMN, "merchant_id"]).agg(pl.len().alias("cnt"))
          .sort([CARD_ID_COLUMN, "cnt"], descending=[False, True])
          .group_by(CARD_ID_COLUMN).agg(pl.first("cnt").alias("top1_cnt"))
    )
    per_amt = (
        tx.with_columns(pl.col("amount").round(0).alias("amt_r"))
          .group_by([CARD_ID_COLUMN, "amt_r"]).agg(pl.len().alias("cnt"))
    )
    rep = per_amt.group_by(CARD_ID_COLUMN).agg(
        pl.col("cnt").max().fill_null(1).alias("max_same_amt_count")
    )
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
    amt_q95 = tx.group_by(CARD_ID_COLUMN).agg(
        pl.col("amt_abs").quantile(0.95).alias("amt_q95")
    )

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
    ])
    card = card.with_columns([
        _safe_ratio(pl.col("n_merchants"), pl.col("n_txns"),           "merchant_diversity"),
        _safe_ratio(pl.col("n_countries_raw") - 1, pl.col("n_txns"),  "foreign_ratio"),
        _safe_ratio(pl.col("amt_std"), pl.col("amt_mean") + 1e-6,     "amt_cv"),
        _safe_ratio(pl.col("n_txns"), pl.col("n_merchants") + 1e-6,   "txns_per_merchant"),
    ]).drop("n_countries_raw")

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


def compute_biz_distance_score(X_train, y_train, X_score):
    common = [c for c in X_train.columns if c in X_score.columns]
    scaler = RobustScaler()
    Xtr = scaler.fit_transform(X_train[common])
    Xsc = scaler.transform(X_score[common])
    biz_c = Xtr[y_train == 1].mean(axis=0)
    con_c = Xtr[y_train == 0].mean(axis=0)
    diff  = np.linalg.norm(Xsc - con_c, axis=1) - np.linalg.norm(Xsc - biz_c, axis=1)
    score = 1.0 / (1.0 + np.exp(-diff / (diff.std() + 1e-6)))
    return score.astype(np.float32)


def compute_isolation_score(X_cons_train, X_cons_score, random_state=42):
    iso = IsolationForest(n_estimators=300, contamination=0.05,
                          random_state=random_state, n_jobs=-1)
    iso.fit(X_cons_train)
    raw   = iso.decision_function(X_cons_score)
    score = -raw
    score = (score - score.min()) / (score.max() - score.min() + 1e-9)
    return score.astype(np.float32)
