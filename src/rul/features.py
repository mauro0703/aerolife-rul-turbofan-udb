"""Strictly causal, per-engine rolling features; no fitted statistics or targets."""

import numpy as np
import pandas as pd
from .data import SENSORS, SETTINGS


def _slope(values):
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)[valid]
    y = values[valid]
    centered = x - x.mean()
    return float(centered @ (y - y.mean()) / (centered @ centered))


def make_features(frame, windows=(5, 15, 30)):
    """Each feature at t uses only rows of the same engine at or before t.

    Missing early-window sample SD is left NaN for train-fitted median imputation.
    Input must be sorted by motor/cycle. ID and final lifetime never enter X.
    """
    if not frame.index.equals(pd.RangeIndex(len(frame))):
        frame = frame.reset_index(drop=True)
    order = frame.sort_values(["engine_id", "cycle"]).index
    if not np.array_equal(order, frame.index):
        raise ValueError("Ordene por engine_id y cycle antes de crear características.")
    columns = {"age_cycles": frame.cycle.astype(float)}
    columns.update({s: frame[s].astype(float) for s in [*SETTINGS, *SENSORS]})
    grouped = frame.groupby("engine_id", sort=False)
    for sensor in SENSORS:
        for window in windows:
            columns[f"{sensor}_mean_{window}"] = grouped[sensor].transform(
                lambda series, w=window: series.rolling(w, min_periods=1).mean()
            )
        columns[f"{sensor}_std_15"] = grouped[sensor].transform(
            lambda series: series.rolling(15, min_periods=2).std()
        )
        columns[f"{sensor}_slope_15"] = grouped[sensor].transform(
            lambda series: series.rolling(15, min_periods=2).apply(_slope, raw=True)
        )
    result = pd.DataFrame(columns, index=frame.index)
    return result.replace([np.inf, -np.inf], np.nan)


def choose_snapshots(labeled, seed=42, min_history=30):
    """Predeclared pseudo-censoring: one uniformly sampled eligible cycle per engine.

    Does not inspect official test labels or test length distribution. The final
    run-to-failure cycle is excluded to avoid validating only RUL=0 endpoints.
    """
    rng = np.random.default_rng(seed)
    indices = []
    for _, group in labeled.groupby("engine_id", sort=True):
        eligible = group[(group.cycle >= min_history) & (group.rul_true > 0)]
        if eligible.empty:
            raise ValueError("Motor sin suficientes ciclos para pseudocensura.")
        indices.append(int(rng.choice(eligible.index.to_numpy())))
    return np.array(indices)
