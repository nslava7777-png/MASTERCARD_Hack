"""Load raw parquet files with Polars."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from src.config import (
    BUSINESS_CARDS_FILE,
    CONSUMER_CARDS_FILE,
    MERCHANTS_REFERENCE_FILE,
    REQUIRED_RAW_FILES,
)


MISSING_DATA_MESSAGE = (
    "Please place business_cards.parquet, consumer_cards.parquet, "
    "and merchants_reference.parquet into data/raw/"
)


def validate_raw_data_files(required_files: list[Path] = REQUIRED_RAW_FILES) -> None:
    """Fail early with a human-readable message if hackathon files are missing."""
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        missing = ", ".join(missing_files)
        raise FileNotFoundError(f"{MISSING_DATA_MESSAGE}\nMissing files: {missing}")


def load_parquet(path: Path) -> pl.DataFrame:
    """Load a parquet file into an eager Polars DataFrame."""
    return pl.read_parquet(path)


def load_raw_data() -> dict[str, pl.DataFrame]:
    """Load all raw data sources required by the pipeline."""
    validate_raw_data_files()
    return {
        "business_cards": load_parquet(BUSINESS_CARDS_FILE),
        "consumer_cards": load_parquet(CONSUMER_CARDS_FILE),
        "merchants_reference": load_parquet(MERCHANTS_REFERENCE_FILE),
    }

