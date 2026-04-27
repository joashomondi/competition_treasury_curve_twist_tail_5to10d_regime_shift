## Objective

Predict whether the U.S. Treasury yield curve will experience an **unusually large “twist” tail move** over the next \(H\) trading days.

Each row is a **date × segment × horizon** example, where:

- `segment_id` selects which curve component is being stress-tested
- `horizon_d` is in {5, 10}

You must output `pred_curve_twist_tail_next` as a probability in \([0,1]\).

## Segments

Segments decompose the curve into level/slope/curvature components:

- `segment_id = 0` (**level spike**): 10Y yield increases sharply
- `segment_id = 1` (**slope crash**): \(10Y-2Y\) flattens sharply (tail negative move)
- `segment_id = 2` (**curvature crash**): \(2\cdot 5Y - 2Y - 10Y\) moves sharply negative
- `segment_id = 3` (**long-end flatten**): \(30Y-10Y\) moves sharply negative

## Target definition (train only)

For the chosen segment, define a future delta \(\Delta m_H(t)=m(t+H)-m(t)\).

The tail event threshold is computed **from training rows only** for each segment+horizon:

- `segment_id = 0` uses the **85th percentile** (positive tail): \(\Delta m_H(t)\ge \tau\)
- segments 1–3 use the **15th percentile** (negative tail): \(\Delta m_H(t)\le \tau\)

The binary target is:

- `target_curve_twist_tail_next = 1` if the tail condition holds, else 0.

## Inputs

- `train.csv`: features + `target_curve_twist_tail_next`
- `test.csv`: same features, no target

Notes:

- Exact dates are hidden; only coarse `event_era`, `season_bin`, and `woy_bin` are provided.
- Continuous signals are released as **train-fitted quantile bins**; missing is encoded as `-1` and summarized by `missing_count`.

## Submission format

Create `submission.csv` with exactly:

- `row_id`
- `pred_curve_twist_tail_next`

## Metric (lower is better)

\[
\text{Score} = 0.55\cdot \text{LogLoss}_{all}
             + 0.30\cdot \text{LogLoss}_{\text{high-vol slice}}
             + 0.15\cdot (1 - \text{AUPRC}_{all})
\]

The slice is provided as `slice_high_rate_vol`.

Deterministic scoring command:

`python score_submission.py --submission-path submission.csv --solution-path solution.csv`

## Regime shift & leakage notes

- The split is time-based with a regime shift (older years train, newer years test) plus deterministic bridge-year mixing.
- The label depends on future yields; future values are not provided as features.
- For honest offline evaluation, avoid random CV. Validate by `event_era` and by `segment_id`.

