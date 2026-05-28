
from __future__ import annotations
import polars as pl
from src.config import CARD_ID_COLUMN, PI2, SUSPICIOUS_MCCS, SMALL_AMOUNT, LARGE_AMOUNT


def _coalesce(df, candidates, alias):
    cols = [pl.col(c) for c in candidates if c in df.columns]
    return (pl.coalesce(cols) if cols else pl.lit(None)).alias(alias)


def prepare_transactions(df, merchants, source_name):
    df = df.with_columns([
        _coalesce(df, [CARD_ID_COLUMN, "card_id", "pan"],                                              CARD_ID_COLUMN).cast(pl.Utf8),
        _coalesce(df, ["transaction_amount_kzt", "amount_kzt", "amount", "transaction_amount"],        "amount").cast(pl.Float64, strict=False),
        _coalesce(df, ["mcc", "merchant_mcc"],                                                         "mcc").cast(pl.Utf8),
        _coalesce(df, ["merchant_id", "merchant"],                                                     "merchant_id").cast(pl.Utf8),
        _coalesce(df, ["channel", "entry_mode"],                                                       "channel").cast(pl.Utf8),
        _coalesce(df, ["tokenized", "is_tokenized"],                                                   "tokenized").cast(pl.Boolean, strict=False),
        _coalesce(df, ["recurring", "is_recurring"],                                                   "recurring").cast(pl.Boolean, strict=False),
        _coalesce(df, ["country", "merchant_country", "country_code"],                                 "country").cast(pl.Utf8),
        _coalesce(df, ["card_tier", "product_segment", "segment"],                                     "card_tier").cast(pl.Utf8),
        _coalesce(df, ["transaction_datetime", "transaction_timestamp", "timestamp", "transaction_date"], "ts_raw").cast(pl.Utf8),
    ])
    if merchants is not None and "merchant_id" in merchants.columns:
        m = merchants
        if "mcc" not in m.columns and "merchant_mcc" in m.columns:
            m = m.rename({"merchant_mcc": "mcc"})
        if "country" not in m.columns:
            for c in ["merchant_country", "country_code"]:
                if c in m.columns:
                    m = m.rename({c: "country"})
                    break
        keep = [c for c in ["merchant_id", "mcc", "country"] if c in m.columns]
        m = m.select(keep).unique("merchant_id")
        df = df.join(m, on="merchant_id", how="left", suffix="_ref")
        for col in ["mcc", "country"]:
            if f"{col}_ref" in df.columns:
                df = df.with_columns(
                    pl.coalesce([pl.col(col), pl.col(f"{col}_ref")]).alias(col)
                ).drop(f"{col}_ref")
    df = df.with_columns(
        pl.col("ts_raw").str.to_datetime(strict=False).alias("dt")
    ).with_columns([
        pl.col("dt").dt.date().alias("date"),
        pl.col("dt").dt.hour().alias("hour"),
        pl.col("dt").dt.weekday().alias("dow"),
        pl.col("channel").str.to_lowercase().fill_null("unknown"),
        pl.col("country").fill_null("UNK"),
        pl.col("card_tier").fill_null("unknown").str.to_lowercase(),
        pl.col("mcc").fill_null("UNK"),
        pl.col("merchant_id").fill_null("UNK"),
        pl.col("tokenized").fill_null(False),
        pl.col("recurring").fill_null(False),
        pl.lit(source_name).alias("source_name"),
    ]).drop("ts_raw")
    return df


def add_txn_flags(df):
    return df.with_columns([
        pl.col("amount").abs().alias("amt_abs"),
        pl.col("amount").abs().log1p().alias("log_amt"),
        (pl.col("channel") == "online").cast(pl.Int8).alias("f_online"),
        pl.col("tokenized").cast(pl.Int8, strict=False).fill_null(0).alias("f_token"),
        pl.col("recurring").cast(pl.Int8, strict=False).fill_null(0).alias("f_recur"),
        pl.col("hour").is_between(1, 5).cast(pl.Int8).alias("f_night"),
        (pl.col("dow") >= 6).cast(pl.Int8).alias("f_weekend"),
        (pl.col("amount").abs() <= SMALL_AMOUNT).cast(pl.Int8).alias("f_small"),
        (pl.col("amount").abs() >= LARGE_AMOUNT).cast(pl.Int8).alias("f_large"),
        pl.col("mcc").is_in(SUSPICIOUS_MCCS).cast(pl.Int8).alias("f_susp_mcc"),
        pl.col("card_tier").str.contains(
            "premium|affluent|black|platinum|signature|infinite", literal=False
        ).fill_null(False).cast(pl.Int8).alias("f_premium"),
        ((PI2 * pl.col("hour") / 24).sin()).alias("hour_sin"),
        ((PI2 * pl.col("hour") / 24).cos()).alias("hour_cos"),
        ((pl.col("channel") == "online") & pl.col("hour").is_between(1, 5)).cast(pl.Int8).alias("f_online_night"),
    ])
