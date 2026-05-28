from pathlib import Path

PROJECT_ROOT       = Path(__file__).resolve().parents[1]
DATA_DIR           = PROJECT_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUTS_DIR        = PROJECT_ROOT / "outputs"
FIGURES_DIR        = OUTPUTS_DIR / "figures"
METRICS_DIR        = OUTPUTS_DIR / "metrics"
MODELS_DIR         = OUTPUTS_DIR / "models"
PREDICTIONS_DIR    = OUTPUTS_DIR / "predictions"

BUSINESS_CARDS_FILE      = RAW_DATA_DIR / "business_cards_MDQ.parquet"
CONSUMER_CARDS_FILE      = RAW_DATA_DIR / "consumer_cards_MDQ.parquet"
MERCHANTS_REFERENCE_FILE = RAW_DATA_DIR / "merchants_reference.parquet"
REQUIRED_RAW_FILES = [BUSINESS_CARDS_FILE, CONSUMER_CARDS_FILE, MERCHANTS_REFERENCE_FILE]
OUTPUT_DIRS = [PROCESSED_DATA_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR, PREDICTIONS_DIR]

CARD_ID_COLUMN = "card_number"
TARGET_COLUMN  = "label"
RANDOM_STATE   = 42

# Grey zone thresholds
GREY_ZONE_LOW   = 0.30   # below -> confident consumer
GREY_ZONE_HIGH  = 0.70   # above -> confident business-like
CONF_BIZ_THRESH = 0.80   # OOF threshold for confident business from train set

# Tuning budget
TUNE_CV_SPLITS = 3
TUNE_N_ITER    = 6
EVAL_CV_SPLITS = 5

SUSPICIOUS_MCCS = ["7311","5968","5099","5172","4816","5912","5045","5065","5094"]
SMALL_AMOUNT    = 2_000.0
LARGE_AMOUNT    = 50_000.0
PI2             = 6.283185307179586

# Subtle grey-zone features
# These capture BEHAVIOURAL PATTERN rather than intensity of primary signals.
# Primary signals (token_ratio, online_ratio, susp_mcc_ratio, amt_mean, n_txns)
# are already "used up" by the base ensemble, so we exclude them here
# and focus on timing, commercial structure, and payment regularity.
GREY_ZONE_FEATURES = [
    "hour_mean",           # payment timing: business=9-18h, consumer=anytime
    "burst_cv",            # payment rhythm: business=steady, consumer=spikes
    "weekend_ratio",       # business rarely pays on weekends
    "same_merchant_ratio", # business concentrates spend at own suppliers
    "txns_per_merchant",   # many txns per merchant = business buying pattern
    "merchant_diversity",  # fewer unique merchants per txn volume = business
    "mcc_diversity",       # narrow MCC range = single industry = business
    "recur_ratio",         # recurring payments: rent, subscriptions, payroll
    "max_same_amt_count",  # fixed-amount payments: rent, salaries
    "amt_cv",              # stable amounts = business, variable = consumer
]
