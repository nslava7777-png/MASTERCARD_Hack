# Mastercard Data Quest 2026

Reproducible ML pipeline for detecting hidden commercial activity in consumer card transaction behavior.

## Goal

The project trains a supervised card-level classifier that learns to distinguish business-card behavior from regular consumer-card behavior. The same logic can later be applied to consumer cards to rank clients who look similar to small businesses operating through personal cards.

Target definition:

- `business_cards.parquet` -> `label = 1`
- `consumer_cards.parquet` -> `label = 0`

The model is trained on one row per `card_number`, not on raw transaction rows.

## Project Structure

```text
mastercard-data-quest/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_explainability.ipynb
├── src/
│   ├── config.py
│   ├── load_data.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explain.py
│   └── utils.py
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── models/
│   └── predictions/
├── main.py
├── requirements.txt
└── README.md
```

## Data Setup

Place these files into `data/raw/`:

- `business_cards.parquet`
- `consumer_cards.parquet`
- `merchants_reference.parquet`

If files are missing, the pipeline exits with:

```text
Please place business_cards.parquet, consumer_cards.parquet, and merchants_reference.parquet into data/raw/
```

## Installation

```bash
cd mastercard-data-quest
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run

```bash
python main.py
```

## Pipeline

`main.py` runs the full reproducible workflow:

1. Load raw parquet files with Polars.
2. Add supervised labels.
3. Combine business and consumer transactions.
4. Merge merchant reference data by `merchant_id`.
5. Build card-level behavioral features.
6. Split cards into train/test sets.
7. Train and compare Logistic Regression and Random Forest.
8. Select the best model by ROC AUC, with F1 as fallback.
9. Save metrics, plots, predictions, and the model artifact.

## Outputs

The pipeline creates:

- `outputs/metrics/metrics.json`
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/feature_importance.png`
- `outputs/predictions/test_predictions.csv`
- `outputs/models/model.pkl`
- `data/processed/card_level_features.parquet`

`test_predictions.csv` contains:

- `card_number`
- `y_true`
- `y_pred`
- `y_proba`

## Feature Engineering

Current baseline features are built at card level:

- Activity: transaction count, amount totals, average, median, max, standard deviation, active days, transactions per active day.
- Merchant behavior: unique merchants, MCCs, transaction countries, merchant countries, merchant and MCC diversity ratios.
- Channel behavior: online and offline shares.
- Recurring/tokenized behavior: recurring share, tokenized share, recurring-capable merchant share.
- Time behavior: weekend, night, morning, and evening shares.
- Amount behavior: small transaction share, large transaction share, amount coefficient of variation.

The baseline does not use `card_number` as a feature and does not use raw `merchant_name` as a categorical feature.

## Model Interpretation

For tree models, `feature_importance.png` uses built-in feature importances. For linear models, it uses absolute coefficients. This gives the presentation team a simple explainability artifact before SHAP is added.

## Limitations

- Labels are proxy labels: business cards are treated as positive examples of business-like behavior.
- The current threshold is fixed at `0.50`; it should be tuned based on the bank's preferred precision/recall trade-off.
- The first version uses classical baseline models only. XGBoost/LightGBM/CatBoost and SHAP can be added later if time and environment allow.
- The current feature set intentionally avoids high-cardinality raw merchant names to keep the first pipeline stable and leakage-resistant.

## Team Handoff

- Data Engineering / Feature Engineering: extend `src/preprocessing.py` and `src/feature_engineering.py`.
- ML Engineering: extend model candidates and tuning in `src/train.py`.
- Metrics / Explainability: extend `src/evaluate.py` and `src/explain.py`, including SHAP if needed.
- Business Storytelling: use files in `outputs/` for slides and final recommendations.

