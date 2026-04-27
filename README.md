## U.S. Treasury Curve Twist Tail Risk (Next 5–10 Trading Days)

Kaggle-style competition package built from U.S. Treasury yield curve data.

- **Task**: binary classification — predict `pred_curve_twist_tail_next`
- **Target**: tail “twist” moves in level/slope/curvature components (panelized by horizon/segment)
- **Split**: time-based regime shift (see `dataset_card.md`)
- **Metric**: composite LogLoss + slice LogLoss + (1 - AUPRC) (see `instruction.md`)

### Quickstart

```bash
python build_dataset.py
python score_submission.py --submission-path sample_submission.csv --solution-path solution.csv
```

