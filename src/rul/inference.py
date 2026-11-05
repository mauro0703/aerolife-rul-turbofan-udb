"""Shared inference for Streamlit and batch use, without access to RUL labels."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from .data import validate_and_clean, to_engineering_units
from .features import make_features
from .modeling import predict_nonnegative
from .evaluation import intervals, inspection_policy


def load_bundle(path="models/rul_predictor_model.pkl"):
    """Load ONLY the trusted locally generated model; never accept uploaded pickle."""
    with Path(path).open("rb") as handle:
        bundle = pickle.load(handle)
    if not {"pipeline", "metadata"}.issubset(bundle):
        raise ValueError("Modelo incompatible.")
    return bundle


def predict_history(raw_frame, bundle, horizon=30, buffer=10, max_interval=20):
    """Native NASA units in; predictions at every observed cycle out.

    For online use pass all history through the current cycle, starting at 1.
    The trained pipeline performs transform only; there is no fit in this path.
    """
    clean, audit = validate_and_clean(raw_frame)
    si = to_engineering_units(clean)
    cfg = bundle["metadata"]["config"]
    features = make_features(si, tuple(cfg["windows"]))
    expected = bundle["metadata"]["feature_columns"]
    if list(features.columns) != expected:
        raise ValueError("Las características no coinciden con el modelo entrenado.")
    pipe = bundle["pipeline"]
    pred = predict_nonnegative(pipe, features)
    lower, upper = intervals(pred, bundle["metadata"]["conformal_radius"])
    status, revisit = inspection_policy(lower, horizon, buffer, max_interval)
    scaled = pipe[:-1].transform(features)
    out = clean[["engine_id", "cycle"]].copy()
    out["rul_pred"] = pred
    out["lower90"] = lower
    out["upper90"] = upper
    out["status"] = status
    out["revisit_cycles"] = revisit
    out["short_history"] = out.cycle < cfg["min_history"]
    out["features_beyond_6sigma"] = (np.abs(scaled) > 6).sum(axis=1)
    return out, si, audit
