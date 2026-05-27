"""Build business-oriented card-level features with Polars."""

from __future__ import annotations
import polars as pl

from src.config import (
CARD_ID_COLUMN,
TARGET_COLUMN,
)
from src.preprocessing import require_columns

FEATURE_REQUIRED_COLUMNS = [
CARD_ID_COLUMN,
TARGET_COLUMN,
"transaction_amount_kzt",
"mcc",
"channel",
"tokenized",
"transaction_date",
"transaction_datetime",
]


def _add_behavior_indicators(transactions: pl.DataFrame) -> pl.DataFrame:
    """Создание промежуточных флагов на уровне отдельных транзакций."""
    return transactions.with_columns(
        # channel может быть null — явно сравниваем со строкой
        (pl.col("channel") == "online")
        .fill_null(False)
        .alias("_is_online"),

        # tokenized: cast сначала в Int8, потом в Boolean — обходит mixed types
        pl.col("tokenized")
        .cast(pl.Int8, strict=False)
        .fill_null(0)
        .cast(pl.Boolean)
        .alias("_is_tokenized"),

        # weekday(): 1=пн ... 7=вс → выходные это 6 и 7
        (pl.col("transaction_date").dt.weekday() >= 6)
        .fill_null(False)
        .alias("_is_weekend"),

        # Вечер: 18–23 включительно
        pl.col("transaction_datetime")
        .dt.hour()
        .is_between(18, 23)
        .fill_null(False)
        .alias("_is_evening"),
    )

def _aggregate_to_cards(transactions_df: pl.DataFrame, include_target: bool) -> pl.DataFrame:
    """Внутренняя функция для группировки обогащенных транзакций по картам."""
    aggs = [
    pl.len().alias("total_transactions"),
    pl.col("transaction_amount_kzt").mean().alias("avg_amount"),
    pl.col("transaction_amount_kzt").std().alias("std_amount"),
    pl.col("mcc").drop_nulls().n_unique().alias("unique_mcc"),

    pl.col("_is_online").mean().alias("online_share"),
    pl.col("_is_tokenized").mean().alias("tokenized_share"),
    pl.col("_is_weekend").mean().alias("weekend_share"),
    pl.col("_is_evening").mean().alias("evening_share"),

    pl.col("mcc_b2b_weight").mean().alias("card_b2b_index"),
    ]

    # Добавляем целевую переменную только если это требуется (для трейна)
    if include_target and TARGET_COLUMN in transactions_df.columns:
        aggs.insert(0, pl.col(TARGET_COLUMN).first().cast(pl.Int8).alias(TARGET_COLUMN))

    card_features = transactions_df.group_by(CARD_ID_COLUMN).agg(aggs)

    return (
    card_features.with_columns(
    [
    pl.when(pl.col("avg_amount").abs() > 0)
    .then(pl.col("std_amount") / pl.col("avg_amount").abs())
    .otherwise(0.0)
    .alias("amount_cv"),
    ]
    )
    .drop("std_amount")
    .fill_null(0)
    .sort(CARD_ID_COLUMN)
    )


def build_features_safe(df_train_trans: pl.DataFrame, df_test_trans: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Безопасная генерация фичей без Data Leakage.
    Веса MCC рассчитываются СТРОГО по df_train_trans и проецируются на df_test_trans.
    """
    require_columns(df_train_trans, FEATURE_REQUIRED_COLUMNS, "train transactions")

    # 1. Добавляем временные/канальные флаги
    df_train_trans = _add_behavior_indicators(df_train_trans)
    df_test_trans = _add_behavior_indicators(df_test_trans)

    # 2. Обучение Target Encoder (ИЗОЛИРОВАНО: только на Train!)
    mcc_weights = (
    df_train_trans.filter(pl.col(TARGET_COLUMN).is_not_null())
    .group_by("mcc")
    .agg([pl.col(TARGET_COLUMN).mean().alias("mcc_b2b_weight")])
    )

    # 3. Применение весов к обоим датасетам
    df_train_trans = df_train_trans.join(mcc_weights, on="mcc", how="left").fill_null(0.0)
    df_test_trans = df_test_trans.join(mcc_weights, on="mcc", how="left").fill_null(0.0)

    # 4. Агрегация транзакций до уровня карт
    train_card_features = _aggregate_to_cards(df_train_trans, include_target=True)
    test_card_features = _aggregate_to_cards(
        df_test_trans,
        include_target=TARGET_COLUMN in df_test_trans.columns  # False при краш-тесте
    )

    return train_card_features, test_card_features