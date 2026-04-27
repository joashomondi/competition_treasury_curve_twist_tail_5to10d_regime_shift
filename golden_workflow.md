## Data checks (curve-specific)

- **Sanity: what the label actually is**
  - This is a *tail move in a curve component* over 5–10 days, not “is the curve inverted?”.
  - Segments 1–3 are **negative-tail** tasks (flattening/curvature-down moves). Segment 0 is **positive-tail** (rate spike).
- **Base rates by segment and horizon**
  - Compute target rate by `segment_id` and `horizon_d` to ensure thresholds behave as intended.
  - If one segment is extremely imbalanced, your model may need segment-conditional calibration.
- **Regime drift**
  - Compare feature bin distributions by `event_era`; you should see differences (e.g., low-rate eras vs high-rate eras).
- **Missingness**
  - Treat `*_bin == -1` as missing; do not map it to “low”.
  - High `missing_count` usually corresponds to early rows without enough rolling history.

## Validation strategy (avoid leakage)

Random CV will overestimate performance because the split is time-based and the curve has persistent regimes.

Recommended:

- **Blocked time validation using `event_era`**
  - Train on earlier eras; validate on the newest era available in training.
  - Report calibration by era (LogLoss-heavy metric).
- **Segment-aware evaluation**
  - Always compute metrics by `segment_id` and `horizon_d`.
  - A model that is good on “level spike” can be miscalibrated on “slope crash”.
- **Slice-aware reporting**
  - Report metrics on:
    - all rows
    - `slice_high_rate_vol == 1` (30% metric weight)

## Why these features/segments matter (unique insight)

Yield curves are often described by three moving parts:

- **Level** (parallel shifts): captured by `level_10y` and yield changes
- **Slope** (front vs belly): captured by `slope_10y2y` and `front_slope_2y3m`
- **Curvature** (belly vs wings): captured by `curvature_2s5s10s` and long-end spread

The tail events here often occur when **one component moves sharply while others do not**, i.e., a “twist”.

Practical implications for modeling:

- Segment 0 (level spike) is driven by *rate volatility* and tends to correlate with `y_2y_std20` and `y_10y_chg5`.
- Segment 1 (slope crash) can happen via **short-end up / long-end flat** (policy repricing) or **long-end down / short-end flat** (growth scare); these are different regimes.
- Segment 2 (curvature crash) is often a **belly move** relative to wings; it is easy to miss without explicit curvature features.

## Baselines and tradeoffs

- **GBDT**
  - Pros: handles non-linear interactions across binned curve features (`segment_id × event_era × vol bins`).
  - Cons: can become overconfident under regime shift without calibration.
- **Regularized logistic regression**
  - Pros: often better calibrated; strong on LogLoss-heavy metrics.
  - Cons: needs explicit interactions (e.g., `segment_id × horizon_d`) to compete.

High-signal interactions to try:

- `segment_id × horizon_d × event_era`
- `slice_high_rate_vol × (y_2y_chg5_bin, y_10y_chg5_bin)`
- curvature bins × level bins (belly moves behave differently at different rate levels)

## Non-obvious failure modes

- **Training-serving skew via volatility**
  - The slice upweights high-vol periods; models that are well-ranked but overconfident lose on LogLoss.
- **Segment mixing**
  - If you do not condition on `segment_id`, the model averages incompatible tails (positive vs negative).
- **Low-rate era compression**
  - When yields are near zero, absolute moves are mechanically smaller; binning can mask this unless you use era/vol conditioning.

## Calibration plan

- Plot reliability by `event_era` and `segment_id`.
- Consider post-hoc calibration on the newest training-era block.
- If the slice is systematically miscalibrated, consider a slice-conditional calibrator or a conservative probability cap.

## Submission checklist

- `submission.csv` has exactly `row_id` and `pred_curve_twist_tail_next`
- ids match `test.csv`
- predictions are finite and in [0,1]
- run:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

