"""Central configuration for paths, constants, and model settings."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"

BUSINESS_CARDS_FILE = RAW_DATA_DIR / "business_cards.parquet"
CONSUMER_CARDS_FILE = RAW_DATA_DIR / "consumer_cards.parquet"
MERCHANTS_REFERENCE_FILE = RAW_DATA_DIR / "merchants_reference.parquet"

REQUIRED_RAW_FILES = [
    BUSINESS_CARDS_FILE,
    CONSUMER_CARDS_FILE,
    MERCHANTS_REFERENCE_FILE,
]

OUTPUT_DIRS = [
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
]

CARD_ID_COLUMN = "card_number"
TARGET_COLUMN = "label"

RANDOM_STATE = 42
TEST_SIZE = 0.20
CLASSIFICATION_THRESHOLD = 0.50

SMALL_TRANSACTION_AMOUNT = 2_000.0
LARGE_TRANSACTION_AMOUNT = 50_000.0

