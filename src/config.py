from __future__ import annotations
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ПУТИ К ДАННЫМ И ДИРЕКТОРИЯМ
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# ГЛОБАЛЬНЫЕ СУЩНОСТИ И НАСТРОЙКИ ОБУЧЕНИЯ
# ─────────────────────────────────────────────────────────────────────────────
CARD_ID_COLUMN = "card_number"
TARGET_COLUMN  = "label"
RANDOM_STATE   = 42

# Параметры кросс-валидации
TUNE_CV_SPLITS = 3
TUNE_N_ITER    = 6
EVAL_CV_SPLITS = 5

# Пороги для разметки и фильтрации серой зоны (Confident-Sampling)
GREY_ZONE_LOW   = 0.30   # Ниже этого порога — базовая модель уверена, что это физлицо (Label 0)
GREY_ZONE_HIGH  = 0.70   # Выше этого порога — серая модель начинает сглаживание
CONF_BIZ_THRESH = 0.80   # Выше этого порога — базовая модель уверена, что это бизнес (Label 1)

# ─────────────────────────────────────────────────────────────────────────────
# КОНСТАНТЫ ПРЕПРОЦЕССИНГА ТРАНЗАКЦИЙ (ФИКСИРОВАННЫЕ, БЕЗ УТЕЧЕК)
# ─────────────────────────────────────────────────────────────────────────────
PI2             = 6.283185307179586
SMALL_AMOUNT    = 2_000.0
LARGE_AMOUNT    = 50_000.0

# Коды категорий торговцев (MCC), характерные для коммерческой деятельности
SUSPICIOUS_MCCS = ["7311", "5968", "5099", "5172", "4816", "5912", "5045", "5065", "5094"]

# ─────────────────────────────────────────────────────────────────────────────
# ПРИЗНАКИ ДЛЯ СЕРОЙ ЗОНЫ (GREY ZONE FEATURES — v6_fixed)
# ─────────────────────────────────────────────────────────────────────────────
# Сюда входят тонкие поведенческие паттерны, которые базовый ансамбль моделей
# не утилизировал до конца. Исключены явные утекающие счетчики.
# Добавлены новые стабильные фичи, повышающие устойчивость к приватному лидерборду.
GREY_ZONE_FEATURES = [
    "hour_mean",               # Среднее время совершения транзакций
    "commercial_hours_ratio",   # НОВАЯ ФИЧА: Доля операций в будни с 9:00 до 18:00 (Бизнес-ритм)
    "burst_cv",                # Стабильность/всплески активности по дням (коэфф. вариации)
    "weekend_ratio",           # Доля транзакций по выходным дням
    "same_merchant_ratio",     # Концентрация трат у ключевого поставщика (топ-1 мерчант)
    "txns_per_merchant",       # Интенсивность закупок: сколько транзакций приходится на одну точку
    "merchant_diversity",      # Разнообразие мерчантов относительно общего объема операций
    "merchants_per_mcc",       # НОВАЯ ФИЧА: Отношение уникальных мерчантов к уникальным категориям MCC
    "mcc_diversity",           # Отраслевая диверсификация (количество уникальных MCC)
    "recur_ratio",             # Регулярные платежи (подписки, аренда, фиксированные B2B-контракты)
    "max_same_amt_count",      # Максимальное совпадение сумм операций (шаблоны выплат/переводов)
    "amt_cv",                  # Стабильность сумм операций (коэффициент вариации размеров трат)
    "biz_distance_score",      # НОВАЯ МЕТА-ФИЧА: Геометрическое расстояние до бизнес-центроиды (без утечек)
]