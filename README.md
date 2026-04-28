## U.S. Treasury Curve Twist Tail Risk (Next 5–10 Trading Days)

Portfolio-ready, Kaggle-style prediction task built from U.S. Treasury yield curve data.

### What you’re predicting
- **Task**: binary classification
- **Predict**: `pred_curve_twist_tail_next`
- **Target**: `target_curve_twist_tail_next` — a **tail twist** happens within the next **5 or 10 trading days**
- **Panelization**: expanded across `segment_id` (level/slope/curvature variants) and `horizon_d ∈ {5,10}`

### Data & evaluation highlights
- **Rows**: train **40,675**, test **17,164**
- **Positive rate (train)**: ~**0.155**
- **Split**: time-based regime shift (details in `dataset_card.md`)
- **Metric**: composite scorer (LogLoss overall + slice LogLoss + (1 − AUPRC)); see `instruction.md`
- **Slice**: `slice_high_rate_vol` stresses performance during high-volatility rate regimes

### Repository contents
- `train.csv`, `test.csv`, `solution.csv`
- `sample_submission.csv`, `perfect_submission.csv`
- `build_dataset.py`, `build_meta.json`
- `score_submission.py`
- `instruction.md`, `golden_workflow.md`, `dataset_card.md`

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

Baseline tips:
- Treat `segment_id` and `horizon_d` as categorical
- Calibrate probabilities (LogLoss-heavy metric)

### Why this is interesting (and non-trivial)
- **Shape risk, not level**: twists are about relative moves across maturities (microstructure + macro), not “rates up/down.”
- **Feature leakage traps**: small lookahead mistakes in rolling features make backtests look unrealistically good—this repo avoids that, and your validation should too.
- **Slice pressure**: `slice_high_rate_vol` emphasizes stressed regimes where probability calibration tends to break.

### Target intuition
The label answers:
> “Will the curve experience an unusually large *twist* within the next 5–10 trading days?”

### Source
U.S. Treasury Daily Yield Curve Rates:
- `https://home.treasury.gov/resource-center/data-chart-center/interest-rates/TextView`

