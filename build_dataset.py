from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


COMP_DIR = Path(__file__).resolve().parent
ROOT = COMP_DIR.parent

UPSTREAM_CACHE = (
    ROOT
    / "competition_treasury_yieldcurve_inversion20d_regime_shift"
    / "_cache"
    / "par-yield-curve-rates-1990-2023.csv"
)

ID_COLUMN = "row_id"
SEGMENT_COLUMN = "segment_id"
HORIZON_COLUMN = "horizon_d"
ERA_COLUMN = "event_era"
TARGET_COLUMN = "target_curve_twist_tail_next"
PRED_COLUMN = "pred_curve_twist_tail_next"
SLICE_COLUMN = "slice_high_rate_vol"
MISSING_COLUMN = "missing_count"

N_BINS = 12
MIN_TRAIN_ROWS = 12_000


@dataclass(frozen=True)
class Config:
    # Regime shift split.
    train_max_year: int = 2012
    bridge_years: Tuple[int, ...] = (2013, 2014, 2015, 2016)
    test_min_year: int = 2017
    bridge_test_rate: int = 35

    # Deterministic sampling (stable but reduces size).
    keep_percent_train: int = 85
    keep_percent_test: int = 92

    # Horizons in trading days (approx).
    horizons_d: Tuple[int, ...] = (5, 10)

    # Tail quantiles (train only) by segment.
    q_pos: float = 0.85   # positive tail
    q_neg: float = 0.15   # negative tail
    min_train_obs_per_segment: int = 900

    # Slice threshold: top-quartile of 2Y 20d vol (train only).
    slice_vol_q: float = 0.75


def _hash_percent(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % 100


def _row_id(seg: int, horizon: int, dt: pd.Timestamp) -> int:
    s = f"tsy_twist::{seg}::{horizon}::{dt:%Y-%m-%d}"
    return int(hashlib.md5(s.encode("utf-8")).hexdigest()[:15], 16)


def _event_era(year: int) -> int:
    if year <= 1999:
        return 0
    if year <= 2007:
        return 1
    if year <= 2013:
        return 2
    if year <= 2019:
        return 3
    return 4


def _nanquantile_edges(x: np.ndarray, n_bins: int) -> np.ndarray:
    x = x.astype(float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([-1.0, 1.0], dtype=float)
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.nanquantile(x, qs)
    edges = np.unique(edges.astype(float))
    if edges.size < 2:
        v = float(edges[0]) if edges.size else 0.0
        return np.array([v - 1.0, v + 1.0], dtype=float)
    return edges


def _bin_with_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    out = np.full(arr.shape[0], -1, dtype=np.int16)
    mask = np.isfinite(arr)
    if mask.any():
        idx = np.digitize(arr[mask], edges, right=False) - 1
        idx = np.clip(idx, 0, max(0, edges.size - 2))
        out[mask] = idx.astype(np.int16)
    return pd.Series(out, index=series.index, dtype="Int16")


def _future_delta(s: pd.Series, horizon: int) -> pd.Series:
    return s.shift(-horizon) - s


def main() -> None:
    cfg = Config()
    if not UPSTREAM_CACHE.exists():
        raise FileNotFoundError(f"Missing upstream cache file: {UPSTREAM_CACHE}")

    df = pd.read_csv(UPSTREAM_CACHE, low_memory=True)
    if "date" not in df.columns:
        raise RuntimeError("Unexpected schema for Treasury par yield curve cache.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%m/%d/%Y")
    df = df[df["date"].notna()].copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["dayofyear"] = df["date"].dt.dayofyear.astype(int)
    df[ERA_COLUMN] = df["year"].map(_event_era).astype(int)

    # Pick robust tenors (they exist in this cache).
    col_3m = "3 mo"
    col_2y = "2 yr"
    col_5y = "5 yr"
    col_10y = "10 yr"
    col_30y = "30 yr"
    for c in [col_3m, col_2y, col_5y, col_10y, col_30y]:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Core level/slope/curvature decomposition.
    df["y_3m"] = df[col_3m]
    df["y_2y"] = df[col_2y]
    df["y_5y"] = df[col_5y]
    df["y_10y"] = df[col_10y]
    df["y_30y"] = df[col_30y]

    df["level_10y"] = df["y_10y"]
    df["slope_10y2y"] = df["y_10y"] - df["y_2y"]
    df["longspread_30y10y"] = df["y_30y"] - df["y_10y"]
    df["curvature_2s5s10s"] = 2.0 * df["y_5y"] - df["y_2y"] - df["y_10y"]
    df["front_slope_2y3m"] = df["y_2y"] - df["y_3m"]

    # Past-only dynamics.
    for c in ["y_2y", "y_10y", "level_10y", "slope_10y2y", "curvature_2s5s10s", "longspread_30y10y"]:
        df[f"{c}_chg1"] = df[c] - df[c].shift(1)
        df[f"{c}_chg5"] = df[c] - df[c].shift(5)
        df[f"{c}_mean20"] = df[c].shift(1).rolling(20, min_periods=10).mean()
        df[f"{c}_std20"] = df[c].shift(1).rolling(20, min_periods=10).std()

    # Coarsen day-of-year to avoid exact date leakage.
    df["woy_bin"] = pd.cut(
        df["date"].dt.isocalendar().week.astype(int),
        bins=[0, 13, 26, 39, 53],
        labels=["W01-13", "W14-26", "W27-39", "W40-53"],
        include_lowest=True,
    ).astype(str)
    df["season_bin"] = pd.cut(
        df["month"].astype(int),
        bins=[0, 3, 6, 9, 12],
        labels=["Q1", "Q2", "Q3", "Q4"],
        include_lowest=True,
    ).astype(str)

    # Split assignment (time-based + deterministic bridge mix).
    years = df["year"].astype(int)
    is_bridge = years.isin(cfg.bridge_years)
    key_pct = (df["woy_bin"].astype(str) + "-" + df["month"].astype(int).astype(str)).map(_hash_percent).astype(int)
    is_test = years >= cfg.test_min_year
    is_test |= is_bridge & (key_pct < cfg.bridge_test_rate)
    is_test &= years > cfg.train_max_year

    # Deterministic sampling.
    keep_key = df["date"].dt.strftime("%Y-%m-%d")
    keep_pct = keep_key.map(_hash_percent).astype(int)
    keep = (~is_test & (keep_pct < cfg.keep_percent_train)) | (is_test & (keep_pct < cfg.keep_percent_test))
    df = df[keep].reset_index(drop=True)
    is_test = is_test[keep].reset_index(drop=True)
    if df.empty:
        raise RuntimeError("No rows after sampling.")

    # Panelization: segments × horizons with different tail directions.
    # segment_id mapping:
    # 0: level spike (10Y up) -> positive tail of delta(level_10y)
    # 1: slope crash (flatten) -> negative tail of delta(slope_10y2y)
    # 2: curvature crash -> negative tail of delta(curvature_2s5s10s)
    # 3: long-end flatten -> negative tail of delta(longspread_30y10y)
    segments: List[Tuple[int, str, str]] = [
        (0, "level_10y", "pos"),
        (1, "slope_10y2y", "neg"),
        (2, "curvature_2s5s10s", "neg"),
        (3, "longspread_30y10y", "neg"),
    ]

    panel_rows = []
    taus: Dict[str, Dict[str, float]] = {}

    for seg_id, metric, tail in segments:
        for h in cfg.horizons_d:
            sub = df.copy()
            sub[SEGMENT_COLUMN] = int(seg_id)
            sub[HORIZON_COLUMN] = int(h)
            sub["delta_h"] = _future_delta(sub[metric], int(h))
            sub = sub[np.isfinite(sub["delta_h"].to_numpy(dtype=float))].copy()
            if sub.empty:
                continue

            train_sub = sub[~is_test.loc[sub.index]].copy()
            if train_sub.empty:
                continue

            vals = train_sub["delta_h"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            if tail == "pos":
                global_tau = float(np.quantile(vals, cfg.q_pos))
                q = cfg.q_pos
                op = ">="
            else:
                global_tau = float(np.quantile(vals, cfg.q_neg))
                q = cfg.q_neg
                op = "<="

            thr_by: Dict[int, float] = {}
            counts = train_sub.groupby(SEGMENT_COLUMN)["delta_h"].count().to_dict()
            # counts is per seg_id (constant), but we keep logic symmetric and simple.
            _ = counts

            # Use a single threshold per segment+horizon (not per instrument) to keep the task focused on dynamics,
            # while still being regime-shifted and calibration-sensitive.
            sub["tau"] = global_tau
            if tail == "pos":
                sub[TARGET_COLUMN] = (sub["delta_h"].to_numpy(dtype=float) >= float(global_tau)).astype(int)
            else:
                sub[TARGET_COLUMN] = (sub["delta_h"].to_numpy(dtype=float) <= float(global_tau)).astype(int)

            taus[f"seg{seg_id}_{metric}_h{h}"] = {"tau_quantile": float(q), "global_tau": global_tau, "op": op}
            panel_rows.append(sub)

    if not panel_rows:
        raise RuntimeError("No panel rows produced.")

    panel = pd.concat(panel_rows, axis=0, ignore_index=True)

    # Recompute is_test for the panel (based on date-derived fields).
    p_year = panel["year"].astype(int)
    p_is_bridge = p_year.isin(cfg.bridge_years)
    p_key_pct = (panel["woy_bin"].astype(str) + "-" + panel["month"].astype(int).astype(str)).map(_hash_percent).astype(int)
    panel_is_test = p_year >= cfg.test_min_year
    panel_is_test |= p_is_bridge & (p_key_pct < cfg.bridge_test_rate)
    panel_is_test &= p_year > cfg.train_max_year

    train_panel = panel[~panel_is_test].copy()
    if train_panel.empty:
        raise RuntimeError("Empty panel train split.")

    # Slice: high rate volatility based on 2Y std20 (train-only threshold).
    vol_vals = train_panel["y_2y_std20"].to_numpy(dtype=float)
    vol_vals = vol_vals[np.isfinite(vol_vals)]
    vol_thr = float(np.quantile(vol_vals, cfg.slice_vol_q)) if vol_vals.size else float("inf")
    panel[SLICE_COLUMN] = (panel["y_2y_std20"].to_numpy(dtype=float) >= vol_thr).astype(int)

    # Train-only bin edges.
    to_bin = [
        "y_3m",
        "y_2y",
        "y_5y",
        "y_10y",
        "y_30y",
        "level_10y",
        "slope_10y2y",
        "front_slope_2y3m",
        "longspread_30y10y",
        "curvature_2s5s10s",
        "y_2y_chg1",
        "y_2y_chg5",
        "y_10y_chg1",
        "y_10y_chg5",
        "slope_10y2y_chg1",
        "slope_10y2y_chg5",
        "curvature_2s5s10s_chg1",
        "curvature_2s5s10s_chg5",
        "longspread_30y10y_chg1",
        "longspread_30y10y_chg5",
        "y_2y_mean20",
        "y_2y_std20",
        "y_10y_mean20",
        "y_10y_std20",
        "level_10y_std20",
        "slope_10y2y_std20",
        "curvature_2s5s10s_std20",
        "longspread_30y10y_std20",
    ]

    edges: Dict[str, list] = {}
    for c in to_bin:
        e = _nanquantile_edges(train_panel[c].to_numpy(dtype=float), N_BINS)
        edges[c] = e.tolist()
        panel[f"{c}_bin"] = _bin_with_edges(panel[c], e)

    panel[MISSING_COLUMN] = panel[to_bin].isna().sum(axis=1).astype(int)

    # Stable row ids + deterministic permutation.
    panel[ID_COLUMN] = panel.apply(lambda r: _row_id(int(r[SEGMENT_COLUMN]), int(r[HORIZON_COLUMN]), pd.Timestamp(r["date"])), axis=1).astype("int64")
    perm_key = (
        panel[SEGMENT_COLUMN].astype(str)
        + "|"
        + panel[HORIZON_COLUMN].astype(str)
        + "|"
        + panel["date"].dt.strftime("%Y-%m-%d")
    ).map(lambda s: int(hashlib.md5(("perm:" + s).encode("utf-8")).hexdigest()[:8], 16))
    order = np.argsort(perm_key.to_numpy(dtype=np.int64), kind="mergesort")
    panel = panel.iloc[order].reset_index(drop=True)
    panel_is_test = panel_is_test.iloc[order].reset_index(drop=True)

    out_cols = [
        ID_COLUMN,
        SEGMENT_COLUMN,
        HORIZON_COLUMN,
        ERA_COLUMN,
        "season_bin",
        "woy_bin",
        SLICE_COLUMN,
        MISSING_COLUMN,
    ] + [f"{c}_bin" for c in to_bin]

    out = panel[out_cols + [TARGET_COLUMN]].copy()
    train_out = out[~panel_is_test].copy()
    test_out = out[panel_is_test].copy()
    if train_out.empty or test_out.empty:
        raise RuntimeError("Build produced empty train or test split.")
    if len(train_out) < MIN_TRAIN_ROWS:
        raise RuntimeError(f"Train set too small ({len(train_out)} rows). Need >= {MIN_TRAIN_ROWS}.")

    train_out.to_csv(COMP_DIR / "train.csv", index=False)
    test_out[out_cols].to_csv(COMP_DIR / "test.csv", index=False)
    test_out[[ID_COLUMN, TARGET_COLUMN, SLICE_COLUMN]].to_csv(COMP_DIR / "solution.csv", index=False)

    sample = pd.DataFrame({ID_COLUMN: test_out[ID_COLUMN].to_numpy(), PRED_COLUMN: 0.5})
    sample.to_csv(COMP_DIR / "sample_submission.csv", index=False)

    y = test_out[TARGET_COLUMN].astype(int).to_numpy()
    perf_p = np.where(y == 1, 0.999, 0.001)
    perfect = pd.DataFrame({ID_COLUMN: test_out[ID_COLUMN].to_numpy(), PRED_COLUMN: perf_p})
    perfect.to_csv(COMP_DIR / "perfect_submission.csv", index=False)

    meta = {
        "upstream_cache": str(UPSTREAM_CACHE),
        "segments": [
            {"segment_id": 0, "metric": "level_10y", "tail": "pos"},
            {"segment_id": 1, "metric": "slope_10y2y", "tail": "neg"},
            {"segment_id": 2, "metric": "curvature_2s5s10s", "tail": "neg"},
            {"segment_id": 3, "metric": "longspread_30y10y", "tail": "neg"},
        ],
        "horizons_d": list(cfg.horizons_d),
        "split": {
            "train_max_year": int(cfg.train_max_year),
            "bridge_years": list(cfg.bridge_years),
            "test_min_year": int(cfg.test_min_year),
            "bridge_test_rate": int(cfg.bridge_test_rate),
            "keep_percent_train": int(cfg.keep_percent_train),
            "keep_percent_test": int(cfg.keep_percent_test),
        },
        "thresholds_summary": taus,
        "slice_threshold_train_only": {"y_2y_std20_q75": vol_thr},
        "n_bins": int(N_BINS),
        "bin_edges": edges,
        "row_counts": {"train": int(len(train_out)), "test": int(len(test_out))},
        "positive_rate": {"train": float(train_out[TARGET_COLUMN].mean()), "test": float(test_out[TARGET_COLUMN].mean())},
    }
    (COMP_DIR / "build_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    print("Wrote competition files to", COMP_DIR)
    print("train rows:", int(len(train_out)), "test rows:", int(len(test_out)))
    print("train positive rate:", float(train_out[TARGET_COLUMN].mean()))
    print("test positive rate:", float(test_out[TARGET_COLUMN].mean()))


if __name__ == "__main__":
    main()

