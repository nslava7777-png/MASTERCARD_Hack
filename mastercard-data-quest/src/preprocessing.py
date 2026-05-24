"""Transaction-level cleaning and merchant reference merge."""

from __future__ import annotations

from collections.abc import Iterable

import polars as pl

from src.config import CARD_ID_COLUMN, TARGET_COLUMN


TRUTHY_VALUES = {"1", "true", "t", "yes", "y"}
FALSEY_VALUES = {"0", "false", "f", "no", "n"}

REQUIRED_TRANSACTION_COLUMNS = [
    CARD_ID_COLUMN,
    "transaction_amount_kzt",
    "merchant_id",
]

OPTIONAL_TRANSACTION_DEFAULTS = {
    "transaction_date": None,
    "transaction_timestamp": None,
    "mcc": None,
    "channel": "unknown",
    "bank_name": None,
    "country": None,
    "card_tier": None,
    "tokenized": False,
    "Is_recurring": False,
}


def require_columns(df: pl.DataFrame, required_columns: Iterable[str], name: str) -> None:
    """Raise a readable error if a required source column is absent."""
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        missing = ", ".join(missing_columns)
        raise ValueError(f"{name} is missing required columns: {missing}")


def _ensure_columns(df: pl.DataFrame, defaults: dict[str, object]) -> pl.DataFrame:
    expressions = [
        pl.lit(default).alias(column)
        for column, default in defaults.items()
        if column not in df.columns
    ]
    if not expressions:
        return df
    return df.with_columns(expressions)


def _bool_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8, strict=False).str.to_lowercase()
    numeric = pl.col(column).cast(pl.Int8, strict=False)
    parsed = (
        pl.col(column).cast(pl.Boolean, strict=False)
        | text.is_in(TRUTHY_VALUES)
        | (numeric == 1)
    )
    return (
        pl.when(text.is_in(FALSEY_VALUES))
        .then(False)
        .otherwise(parsed)
        .fill_null(False)
    )


def _parse_date_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8, strict=False)
    return pl.coalesce(
        [
            pl.col(column).cast(pl.Date, strict=False),
            text.str.strptime(pl.Date, strict=False),
            text.str.strptime(pl.Datetime, strict=False).dt.date(),
        ]
    )


def _parse_datetime_expr(column: str) -> pl.Expr:
    text = pl.col(column).cast(pl.Utf8, strict=False)
    return pl.coalesce(
        [
            pl.col(column).cast(pl.Datetime, strict=False),
            text.str.strptime(pl.Datetime, strict=False),
            text.str.strptime(pl.Date, strict=False).cast(pl.Datetime),
        ]
    )


def add_target(transactions: pl.DataFrame, label: int) -> pl.DataFrame:
    """Attach supervised label: business=1, consumer=0."""
    return transactions.with_columns(pl.lit(label).cast(pl.Int8).alias(TARGET_COLUMN))


def normalize_transactions(transactions: pl.DataFrame, name: str) -> pl.DataFrame:
    """Normalize transaction dtypes and basic missing values before aggregation."""
    require_columns(transactions, REQUIRED_TRANSACTION_COLUMNS, name)
    transactions = _ensure_columns(transactions, OPTIONAL_TRANSACTION_DEFAULTS)

    cleaned = transactions.with_columns(
        [
            pl.col(CARD_ID_COLUMN).cast(pl.Utf8, strict=False),
            pl.col("merchant_id").cast(pl.Utf8, strict=False),
            pl.col("transaction_amount_kzt").cast(pl.Float64, strict=False),
            pl.col("mcc").cast(pl.Utf8, strict=False),
            pl.col("channel")
            .cast(pl.Utf8, strict=False)
            .str.to_lowercase()
            .fill_null("unknown"),
            pl.col("country").cast(pl.Utf8, strict=False),
            _bool_expr("tokenized").alias("tokenized"),
            _bool_expr("Is_recurring").alias("Is_recurring"),
            _parse_date_expr("transaction_date").alias("transaction_date"),
            _parse_datetime_expr("transaction_timestamp").alias(
                "transaction_timestamp"
            ),
        ]
    )

    cleaned = cleaned.with_columns(
        pl.coalesce(
            [
                pl.col("transaction_timestamp"),
                pl.col("transaction_date").cast(pl.Datetime, strict=False),
            ]
        ).alias("transaction_datetime")
    )

    return cleaned.filter(
        pl.col(CARD_ID_COLUMN).is_not_null()
        & pl.col("transaction_amount_kzt").is_not_null()
    )


def normalize_merchants(merchants: pl.DataFrame) -> pl.DataFrame:
    """Prepare merchant reference columns for a safe join."""
    require_columns(merchants, ["merchant_id"], "merchants_reference")

    rename_map = {}
    if "mcc" in merchants.columns:
        rename_map["mcc"] = "merchant_reference_mcc"
    merchants = merchants.rename(rename_map) if rename_map else merchants

    merchants = _ensure_columns(
        merchants,
        {
            "merchant_name": None,
            "merchant_country": None,
            "recurring_capable": False,
        },
    )

    return (
        merchants.with_columns(
            [
                pl.col("merchant_id").cast(pl.Utf8, strict=False),
                pl.col("merchant_name").cast(pl.Utf8, strict=False),
                pl.col("merchant_country").cast(pl.Utf8, strict=False),
                _bool_expr("recurring_capable").alias("recurring_capable"),
            ]
        )
        .unique(subset=["merchant_id"], keep="first")
        .select(
            [
                "merchant_id",
                "merchant_name",
                "merchant_country",
                "recurring_capable",
                *(
                    ["merchant_reference_mcc"]
                    if "merchant_reference_mcc" in merchants.columns
                    else []
                ),
            ]
        )
    )


def combine_transactions(
    business_cards: pl.DataFrame,
    consumer_cards: pl.DataFrame,
) -> pl.DataFrame:
    """Normalize both segments, attach labels, and vertically concatenate."""
    business = add_target(
        normalize_transactions(business_cards, "business_cards"),
        label=1,
    )
    consumer = add_target(
        normalize_transactions(consumer_cards, "consumer_cards"),
        label=0,
    )
    return pl.concat([business, consumer], how="vertical_relaxed")


def merge_merchants(
    transactions: pl.DataFrame,
    merchants_reference: pl.DataFrame,
) -> pl.DataFrame:
    """Left join transaction rows with merchant reference data."""
    merchants = normalize_merchants(merchants_reference)
    merged = transactions.join(merchants, on="merchant_id", how="left")
    return merged.with_columns(
        [
            pl.col("merchant_country").fill_null("unknown"),
            pl.col("recurring_capable").fill_null(False),
        ]
    )


def prepare_transactions(
    business_cards: pl.DataFrame,
    consumer_cards: pl.DataFrame,
    merchants_reference: pl.DataFrame,
) -> pl.DataFrame:
    """Full transaction-level preparation step used by main.py."""
    transactions = combine_transactions(business_cards, consumer_cards)
    return merge_merchants(transactions, merchants_reference)

