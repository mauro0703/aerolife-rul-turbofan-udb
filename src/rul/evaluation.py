"""Regression, asymmetric NASA score and finite-sample split-conformal intervals."""

import math
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def nasa_score(y_true, y_pred):
    delta = np.asarray(y_pred) - np.asarray(y_true)
    return float(
        np.where(delta < 0, np.expm1(-delta / 13.0), np.expm1(delta / 10.0)).sum()
    )


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "bias": float((y_pred - y_true).mean()),
        "nasa_score": nasa_score(y_true, y_pred),
        "n": len(y_true),
    }


def conformal_radius(y_true, y_pred, alpha=0.10):
    residuals = np.sort(np.abs(np.asarray(y_true) - np.asarray(y_pred)))
    n = len(residuals)
    rank = math.ceil((n + 1) * (1 - alpha))
    if rank > n:
        return float("inf")  # Insufficient calibration units for a finite interval.
    return float(residuals[rank - 1])


def intervals(prediction, radius):
    prediction = np.asarray(prediction)
    return np.maximum(0, prediction - radius), prediction + radius


def bootstrap_rmse(y_true, y_pred, seed=42, repetitions=2000):
    """One endpoint per engine: resample independent engines, never cycles."""
    rng = np.random.default_rng(seed)
    errors2 = (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    samples = rng.choice(errors2, size=(repetitions, len(errors2)), replace=True)
    return np.quantile(np.sqrt(samples.mean(axis=1)), [0.025, 0.975]).tolist()


def inspection_policy(lower, horizon=30, buffer=10, max_interval=20):
    """Demonstration cycle budget; not an OEM maintenance instruction."""
    lower = np.asarray(lower, dtype=float)
    budget = np.maximum(0, lower - buffer)
    status = np.where(
        lower <= buffer,
        "Revisión prioritaria",
        np.where(lower <= horizon + buffer, "Planificar inspección", "Seguimiento"),
    )
    revisit = np.floor(np.minimum(max_interval, budget)).astype(int)
    return status, revisit
