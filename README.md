# Mastercard Hidden Business Detection — v6

## Project Structure
```
mastercard_v6/
├── main.py                    <- Entry point
├── requirements.txt
├── README.md
├── src/
│   ├── config.py              <- All constants and paths
│   ├── load_data.py           <- Load 3 parquet files
│   ├── preprocessing.py       <- Transaction cleaning + flags
│   ├── feature_engineering.py <- 25 card-level features
│   ├── train.py               <- Tuning, OOF eval, grey zone refiner
│   └── evaluate.py            <- Metrics + 8 diagnostic plots
└── data/
    └── raw/
        ├── business_cards_MDQ.parquet
        ├── consumer_cards_MDQ.parquet
        └── merchants_reference.parquet
```

## Setup
```bash
pip install -r requirements.txt
python main.py
```

## Grey Zone Refiner (v6)

### Problem with v5 (linear rescaling)
Linear rescaling only changed score VALUES, not RANKING.
A card at 0.50 became 0.67 — still just as ambiguous. No real separation.

### Solution: Confident-Sampling Refiner
Training data built from boundary examples on BOTH sides of grey zone:

| Label | Source | Threshold |
|-------|--------|-----------|
| 0 (consumer) | Consumer cards | ensemble_score < **0.30** |
| 1 (business) | Business train cards | OOF score > **0.80** |

These are the most "certain" examples on each side.
The model learns: "given these subtle patterns, is this card more like a clear consumer or a clear business?"

### Subtle Features (10)
Instead of using primary signals already exploited by the base ensemble,
the grey zone model uses subtle behavioural patterns:

| Feature | Signal |
|---------|--------|
| `hour_mean` | Business pays 9-18h; consumer pays anytime |
| `burst_cv` | Business = steady rhythm; consumer = spikes |
| `weekend_ratio` | Business rarely pays on weekends |
| `same_merchant_ratio` | Business concentrates at own suppliers |
| `txns_per_merchant` | Business makes many txns per merchant |
| `merchant_diversity` | Business = fewer unique merchants per volume |
| `mcc_diversity` | Business = narrow MCC range (one industry) |
| `recur_ratio` | Recurring payments: rent, subscriptions, payroll |
| `max_same_amt_count` | Fixed-amount payments: rent, salaries |
| `amt_cv` | Business = stable amounts; consumer = variable |

### Blending
```
final_score[grey_zone] = 0.45 * ensemble + 0.55 * grey_model
```
Grey model gets majority weight (0.55) because it was trained specifically
to discriminate within the ambiguous region.

### Expected Result
Cards previously bunched at 0.45–0.65 now spread to 0.30–0.80
depending on whether subtle patterns say "consumer" or "business".

## Output Files
```
outputs/
├── predictions/final_submission.csv   <- card_number, score, risk_tier
├── metrics/metrics.json               <- all metrics + config
├── models/best_base_model.pkl
└── figures/
    ├── 01_confusion_matrix.png
    ├── 02_oof_score_dist.png
    ├── 03_roc_curves.png
    ├── 04_pr_curves.png
    ├── 05_feature_importance.png
    ├── 06_grey_zone_analysis.png      <- before/after/delta
    ├── 07_consumer_score_breakdown.png
    └── 08_top50_suspicious.png
```

## Risk Tiers
| Tier | Score | Meaning |
|------|-------|---------|
| HIGH | >= 0.70 | Very likely hidden business |
| MEDIUM | 0.30 – 0.70 | Grey zone — needs investigation |
| LOW | < 0.30 | Likely genuine consumer |
