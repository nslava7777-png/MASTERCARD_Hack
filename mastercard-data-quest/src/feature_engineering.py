"""Build business-oriented card-level features with Polars."""

from __future__ import annotations

import polars as pl

from src.config import (
    CARD_ID_COLUMN,
    LARGE_TRANSACTION_AMOUNT,
    SMALL_TRANSACTION_AMOUNT,
    TARGET_COLUMN,
)
from src.preprocessing import require_columns


FEATURE_REQUIRED_COLUMNS = [
    CARD_ID_COLUMN,
    TARGET_COLUMN,
    "transaction_amount_kzt",
    "merchant_id",
    "mcc",
    "country",
    "merchant_country",
    "channel",
    "tokenized",
    "Is_recurring",
    "recurring_capable",
    "transaction_date",
    "transaction_datetime",
]


def _add_behavior_indicators(transactions: pl.DataFrame) -> pl.DataFrame:
    amount_abs = pl.col("transaction_amount_kzt").abs()
    hour = pl.col("transaction_datetime").dt.hour()

    return transactions.with_columns(
        [
            (pl.col("channel") == "online").fill_null(False).alias("_is_online"),
            (pl.col("channel") == "offline").fill_null(False).alias("_is_offline"),
            pl.col("Is_recurring").cast(pl.Boolean, strict=False).alias(
                "_is_recurring"
            ),
            pl.col("tokenized").cast(pl.Boolean, strict=False).alias("_is_tokenized"),
            pl.col("recurring_capable")
            .cast(pl.Boolean, strict=False)
            .alias("_is_recurring_capable"),
            (pl.col("transaction_date").dt.weekday() >= 6)
            .fill_null(False)
            .alias("_is_weekend"),
            hour.is_between(0, 5, closed="both").fill_null(False).alias("_is_night"),
            hour.is_between(6, 11, closed="both").fill_null(False).alias("_is_morning"),
            hour.is_between(18, 23, closed="both")
            .fill_null(False)
            .alias("_is_evening"),
            (amount_abs < SMALL_TRANSACTION_AMOUNT)
            .fill_null(False)
            .alias("_is_small_txn"),
            (amount_abs >= LARGE_TRANSACTION_AMOUNT)
            .fill_null(False)
            .alias("_is_large_txn"),
        ]
    )


def build_card_level_features(transactions: pl.DataFrame) -> pl.DataFrame:
    """Aggregate transaction-level behavior into one row per card_number."""
    require_columns(transactions, FEATURE_REQUIRED_COLUMNS, "prepared transactions")
    transactions = _add_behavior_indicators(transactions)

    card_features = transactions.group_by(CARD_ID_COLUMN).agg(
        [
            pl.col(TARGET_COLUMN).first().cast(pl.Int8).alias(TARGET_COLUMN),
            pl.len().alias("total_transactions"),
            pl.col("transaction_amount_kzt").sum().alias("total_amount"),
            pl.col("transaction_amount_kzt").mean().alias("avg_amount"),
            pl.col("transaction_amount_kzt").median().alias("median_amount"),
            pl.col("transaction_amount_kzt").max().alias("max_amount"),
            pl.col("transaction_amount_kzt").std().alias("std_amount"),
            pl.col("transaction_date").drop_nulls().n_unique().alias("active_days"),
            pl.col("merchant_id").drop_nulls().n_unique().alias("unique_merchants"),
            pl.col("mcc").drop_nulls().n_unique().alias("unique_mcc"),
            pl.col("country").drop_nulls().n_unique().alias("unique_countries"),
            pl.col("merchant_country")
            .drop_nulls()
            .n_unique()
            .alias("unique_merchant_countries"),
            pl.col("_is_online").mean().alias("online_share"),
            pl.col("_is_offline").mean().alias("offline_share"),
            pl.col("_is_recurring").mean().alias("recurring_share"),
            pl.col("_is_tokenized").mean().alias("tokenized_share"),
            pl.col("_is_recurring_capable").mean().alias("recurring_capable_share"),
            pl.col("_is_weekend").mean().alias("weekend_share"),
            pl.col("_is_night").mean().alias("night_share"),
            pl.col("_is_morning").mean().alias("morning_share"),
            pl.col("_is_evening").mean().alias("evening_share"),
            pl.col("_is_small_txn").mean().alias("small_txn_share"),
            pl.col("_is_large_txn").mean().alias("large_txn_share"),
        ]
    )

    return (
        card_features.with_columns(
            [
                pl.when(pl.col("active_days") > 0)
                .then(pl.col("total_transactions") / pl.col("active_days"))
                .otherwise(0.0)
                .alias("transactions_per_active_day"),
                pl.when(pl.col("total_transactions") > 0)
                .then(pl.col("unique_merchants") / pl.col("total_transactions"))
                .otherwise(0.0)
                .alias("merchant_diversity_ratio"),
                pl.when(pl.col("total_transactions") > 0)
                .then(pl.col("unique_mcc") / pl.col("total_transactions"))
                .otherwise(0.0)
                .alias("mcc_diversity_ratio"),
                pl.when(pl.col("avg_amount").abs() > 0)
                .then(pl.col("std_amount") / pl.col("avg_amount").abs())
                .otherwise(0.0)
                .alias("amount_cv"),
            ]
        )
        .fill_null(0)
        .sort(CARD_ID_COLUMN)
    )

