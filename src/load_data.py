from __future__ import annotations
import polars as pl
from src.config import BUSINESS_CARDS_FILE, CONSUMER_CARDS_FILE, MERCHANTS_REFERENCE_FILE, REQUIRED_RAW_FILES

def load_raw_data():
    for p in REQUIRED_RAW_FILES:
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")
    return {
        "business_cards":      pl.read_parquet(BUSINESS_CARDS_FILE),
        "consumer_cards":      pl.read_parquet(CONSUMER_CARDS_FILE),
        "merchants_reference": pl.read_parquet(MERCHANTS_REFERENCE_FILE),
    }
