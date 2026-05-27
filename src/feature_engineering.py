"""Build business-oriented card-level features with Polars for PU-Learning."""

from __future__ import annotations
import polars as pl

# Обязательно убедитесь, что SMALL_TRANSACTION_AMOUNT и LARGE_TRANSACTION_AMOUNT
# прописаны в src.config, иначе код выдаст ошибку импорта.
from src.config import (
    CARD_ID_COLUMN,
    TARGET_COLUMN,
    SMALL_TRANSACTION_AMOUNT,
    LARGE_TRANSACTION_AMOUNT,
)
from src.preprocessing import require_columns

# TARGET_COLUMN убран из списка обязательных для сырых данных,
# так как теперь метки ставятся искусственно после агрегации.
# Добавлен merchant_id для расчета вашей фирменной фичи (diversity).
FEATURE_REQUIRED_COLUMNS = [
    CARD_ID_COLUMN,
    "transaction_amount_kzt",
    "mcc",
    "merchant_id",
    "channel",
    "tokenized",
    "transaction_date",
    "transaction_datetime",
]


def _add_behavior_indicators(transactions: pl.DataFrame) -> pl.DataFrame:
    """Создание промежуточных флагов на уровне отдельных транзакций."""
    amount_abs = pl.col("transaction_amount_kzt").abs()

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

        # Фирменные фичи: маркеры размеров транзакций
        (amount_abs <= SMALL_TRANSACTION_AMOUNT)
        .fill_null(False)
        .alias("_is_small_txn"),

        (amount_abs >= LARGE_TRANSACTION_AMOUNT)
        .fill_null(False)
        .alias("_is_large_txn"),
    )


def calculate_mcc_business_ratio(df_biz: pl.DataFrame, df_cons: pl.DataFrame) -> pl.DataFrame:
    """
    PU-Learning: Расчет бизнес-веса MCC как отношение его частоты
    в эталонном бизнесе к общей частоте (бизнес + потребители).
    Заменяет старый Target Encoder.
    """
    mcc_x = df_biz.group_by("mcc").len(name="count_biz")
    mcc_y = df_cons.group_by("mcc").len(name="count_cons")

    mcc_weights = mcc_x.join(mcc_y, on="mcc", how="full", coalesce=True).fill_null(0)

    # Считаем долю бизнеса. Добавлено 1e-5 для защиты от деления на ноль.
    mcc_weights = mcc_weights.with_columns(
        (pl.col("count_biz") / (pl.col("count_biz") + pl.col("count_cons") + 1e-5)).alias("mcc_business_ratio")
    )
    return mcc_weights.select(["mcc", "mcc_business_ratio"])


def _aggregate_to_cards(transactions_df: pl.DataFrame, mcc_weights: pl.DataFrame, label: int) -> pl.DataFrame:
    """Группировка обогащенных транзакций по картам и расчет относительных коэффициентов."""
    # Присоединяем веса MCC и добавляем флаги поведения
    df_proc = transactions_df.join(mcc_weights, on="mcc", how="left").fill_null(0.0)
    df_proc = _add_behavior_indicators(df_proc)

    # Базовая агрегация
    card_features = df_proc.group_by(CARD_ID_COLUMN).agg([
        pl.len().alias("total_transactions"),
        pl.col("transaction_amount_kzt").mean().alias("avg_amount"),
        pl.col("transaction_amount_kzt").std().alias("std_amount"),
        pl.col("mcc").drop_nulls().n_unique().alias("unique_mcc"),
        pl.col("merchant_id").drop_nulls().n_unique().alias("unique_merchants"),

        pl.col("_is_online").mean().alias("online_share"),
        pl.col("_is_tokenized").mean().alias("tokenized_share"),
        pl.col("_is_weekend").mean().alias("weekend_share"),
        pl.col("_is_evening").mean().alias("evening_share"),

        pl.col("_is_small_txn").mean().alias("small_txn_share"),
        pl.col("_is_large_txn").mean().alias("large_txn_share"),

        pl.col("mcc_business_ratio").mean().alias("card_b2b_index"),
    ])

    # Расчет финальных фирменных коэффициентов
    card_features = card_features.with_columns(
        [
            # amount_cv: показывает стабильность чека
            pl.when(pl.col("avg_amount").abs() > 0)
            .then(pl.col("std_amount") / pl.col("avg_amount").abs())
            .otherwise(0.0)
            .alias("amount_cv"),

            # merchant_diversity_ratio: показывает концентрацию закупок
            pl.when(pl.col("total_transactions") > 0)
            .then(pl.col("unique_merchants") / pl.col("total_transactions"))
            .otherwise(0.0)
            .alias("merchant_diversity_ratio")
        ]
    ).drop(["std_amount", "unique_merchants"]).fill_null(0)

    # Жестко проставляем целевую переменную (1 для бизнеса, 0 для потребителей)
    return (
        card_features
        .with_columns(pl.lit(label).cast(pl.Int8).alias(TARGET_COLUMN))
        .sort(CARD_ID_COLUMN)
    )


def build_pu_features(df_biz_trans: pl.DataFrame, df_cons_trans: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Главная функция генерации фичей.
    Заменяет старую build_features_safe, принимая на вход два раздельных потока данных.
    """
    require_columns(df_biz_trans, FEATURE_REQUIRED_COLUMNS, "business transactions")
    require_columns(df_cons_trans, FEATURE_REQUIRED_COLUMNS, "consumer transactions")

    # 1. Обучение весов MCC (ИЗОЛИРОВАНО: сравниваем X и Y)
    mcc_weights = calculate_mcc_business_ratio(df_biz_trans, df_cons_trans)

    # 2. Агрегация транзакций до уровня карт с проставлением правильных меток
    biz_card_features = _aggregate_to_cards(df_biz_trans, mcc_weights, label=1)
    cons_card_features = _aggregate_to_cards(df_cons_trans, mcc_weights, label=0)

    return biz_card_features, cons_card_features