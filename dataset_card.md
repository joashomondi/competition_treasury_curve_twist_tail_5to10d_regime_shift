## Overview

This competition uses U.S. Treasury par yield curve rates to forecast **tail “twist” moves** in level/slope/curvature components over the next 5–10 trading days. Rows are panelized across `segment_id` (which curve component) and `horizon_d` (forecast window).

## Source

U.S. Treasury par yield curve rates (cached file `par-yield-curve-rates-1990-2023.csv` in this workspace).

## License

Public-Domain-US-Gov

U.S. Treasury data products are generally public domain.

## Features

All numeric signals are released as **train-fitted quantile bins**; `-1` means missing.

- **Panel identifiers**: `row_id`, `segment_id`, `horizon_d`
- **Coarsened time**: `event_era`, `season_bin`, `woy_bin`
- **Slice**: `slice_high_rate_vol` (high 2Y-rate volatility regime)
- **Binned curve signals**:
  - yields (3M/2Y/5Y/10Y/30Y)
  - slope/curvature/long-spread
  - 1d/5d changes and 20d rolling mean/std
- **Missingness**: `missing_count`

## Splitting & Leakage

- **Split policy (deterministic, time-based)**:
  - Train: years \(\le 2012\)
  - Bridge: 2013–2016 (deterministic hashed assignment to test)
  - Test: years \(\ge 2017\)
- **Leakage mitigations**:
  - exact dates are excluded (only coarse time buckets)
  - bin edges are learned from training rows only
  - the label depends on future yields not available to features

