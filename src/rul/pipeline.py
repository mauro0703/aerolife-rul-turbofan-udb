"""End-to-end experiment. The official test is opened only after model freeze."""

from pathlib import Path
from time import perf_counter
import importlib.metadata
import json
import platform
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import GroupKFold
from .config import Config
from .data import (
    download_data,
    read_raw,
    validate_and_clean,
    to_engineering_units,
    add_training_targets,
    read_test_truth,
    sha256,
)
from .features import make_features, choose_snapshots
from .modeling import candidates, make_pipeline, predict_nonnegative
from .evaluation import (
    regression_metrics,
    conformal_radius,
    intervals,
    bootstrap_rmse,
    inspection_policy,
)
from .analysis import analyze_development, result_figures
from .artifacts import analytical_run


def save_json(path, value):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def _weights(groups):
    counts = groups.value_counts()
    return (len(groups) / (len(counts) * groups.map(counts))).to_numpy()


def fit_weighted(pipeline, X, y, groups):
    return pipeline.fit(X, y, regressor__sample_weight=_weights(groups))


def run_pipeline(config=None):
    """Publish a complete, fingerprinted run; readers reject interrupted runs."""
    cfg = config or Config.load()
    with analytical_run():
        return _run_pipeline(cfg)


def _run_pipeline(cfg):
    """Select with group CV; calibrate once; evaluate NASA labels once; save evidence."""
    started = perf_counter()
    for directory in ("data/processed", "models", "reports/figures"):
        Path(directory).mkdir(parents=True, exist_ok=True)
    provenance = download_data()
    raw, audit_train = validate_and_clean(read_raw("data/raw/train_FD001.txt"))
    labeled = add_training_targets(to_engineering_units(raw), cfg.rul_cap)
    X = make_features(labeled, cfg.windows)
    rng = np.random.default_rng(cfg.seed)
    all_ids = np.sort(labeled.engine_id.unique())
    if len(all_ids) - cfg.calibration_engines < cfg.cv_folds:
        raise ValueError("No quedan suficientes motores de desarrollo para GroupKFold.")
    calibration_ids = np.sort(
        rng.choice(all_ids, cfg.calibration_engines, replace=False)
    )
    dev_ids = np.setdiff1d(all_ids, calibration_ids)
    assert not set(dev_ids) & set(calibration_ids)
    dev_mask = labeled.engine_id.isin(dev_ids)
    dev = labeled.loc[dev_mask].copy()
    X_dev, y_dev, groups = (
        X.loc[dev_mask],
        labeled.loc[dev_mask, "rul_target"],
        labeled.loc[dev_mask, "engine_id"],
    )
    snapshots = choose_snapshots(labeled, cfg.seed + 1, cfg.min_history)
    dev_snapshots = np.intersect1d(snapshots, labeled.index[dev_mask])
    calibration_snapshots = np.intersect1d(snapshots, labeled.index[~dev_mask])
    eda, hypotheses = analyze_development(
        dev, X_dev, seed=cfg.seed, repetitions=cfg.bootstrap_repetitions
    )
    splitter = GroupKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed)
    folds = list(splitter.split(X_dev, y_dev, groups))
    choices = candidates(cfg)
    records, fold_audit, oof = [], [], []
    for fold_no, (train_pos, val_pos) in enumerate(folds, 1):
        train_idx, val_idx = X_dev.index[train_pos], X_dev.index[val_pos]
        snapshot_idx = np.intersect1d(val_idx, dev_snapshots)
        train_ids = sorted(groups.loc[train_idx].unique().tolist())
        val_ids = sorted(groups.loc[val_idx].unique().tolist())
        assert not set(train_ids) & set(val_ids)
        fold_audit.append(
            {
                "fold": fold_no,
                "fit_engine_ids": train_ids,
                "validation_engine_ids": val_ids,
                "fit_rows": len(train_idx),
                "validation_snapshots": len(snapshot_idx),
                "overlap": [],
            }
        )
        for name, estimator in choices.items():
            t0 = perf_counter()
            pipe = make_pipeline(clone(estimator))
            fit_weighted(
                pipe,
                X.loc[train_idx],
                labeled.loc[train_idx, "rul_target"],
                groups.loc[train_idx],
            )
            prediction = predict_nonnegative(pipe, X.loc[snapshot_idx])
            oof.extend({"candidate": name, "fold": fold_no,
                        "engine_id": int(labeled.loc[i, "engine_id"]),
                        "cycle": int(labeled.loc[i, "cycle"]),
                        "rul_true": float(labeled.loc[i, "rul_true"]),
                        "rul_pred": float(p)} for i, p in zip(snapshot_idx, prediction))
            metrics = regression_metrics(
                labeled.loc[snapshot_idx, "rul_true"], prediction
            )
            records.append(
                {
                    "candidate": name,
                    "fold": fold_no,
                    **metrics,
                    "seconds": perf_counter() - t0,
                }
            )
        print(f"CV fold {fold_no}/{cfg.cv_folds} completo", flush=True)
    cv = pd.DataFrame(records)
    pd.DataFrame(oof).to_csv("reports/cv_predictions.csv", index=False)
    cv.to_csv("reports/cv_results.csv", index=False)
    comparison = (
        cv.groupby("candidate")
        .agg(cv_rmse=("rmse", "mean"), cv_sd=("rmse", "std"), cv_mae=("mae", "mean"))
        .sort_values(["cv_rmse", "cv_mae"])
    )
    comparison.to_csv("reports/model_comparison.csv")
    winner = str(comparison.index[0])
    print(f"Modelo seleccionado por CV: {winner}", flush=True)
    # Learning curve uses a fixed CV validation fold, never calibration or official test.
    train_pos, val_pos = folds[0]
    train_ids = np.sort(groups.iloc[train_pos].unique())
    val_idx = np.intersect1d(X_dev.index[val_pos], dev_snapshots)
    curve = []
    for count in (16, 32, 48, len(train_ids)):
        selected = train_ids[:count]
        ti = labeled.index[labeled.engine_id.isin(selected)]
        pipe = make_pipeline(clone(choices[winner]))
        fit_weighted(
            pipe, X.loc[ti], labeled.loc[ti, "rul_target"], labeled.loc[ti, "engine_id"]
        )
        tsi = np.intersect1d(ti, dev_snapshots)
        curve.append(
            {
                "n_engines": len(selected),
                "train_rmse": regression_metrics(
                    labeled.loc[tsi, "rul_true"], predict_nonnegative(pipe, X.loc[tsi])
                )["rmse"],
                "validation_rmse": regression_metrics(
                    labeled.loc[val_idx, "rul_true"],
                    predict_nonnegative(pipe, X.loc[val_idx]),
                )["rmse"],
            }
        )
    learning = pd.DataFrame(curve)
    learning.to_csv("reports/learning_curve.csv", index=False)
    final = make_pipeline(clone(choices[winner]))
    fit_weighted(final, X_dev, y_dev, groups)
    cal_pred = predict_nonnegative(final, X.loc[calibration_snapshots])
    cal_true = labeled.loc[calibration_snapshots, "rul_true"].to_numpy()
    radius = conformal_radius(cal_true, cal_pred, cfg.alpha)
    lo, hi = intervals(cal_pred, radius)
    calibration = pd.DataFrame(
        {
            "engine_id": labeled.loc[calibration_snapshots, "engine_id"].to_numpy(),
            "cycle": labeled.loc[calibration_snapshots, "cycle"].to_numpy(),
            "rul_true": cal_true,
            "rul_pred": cal_pred,
            "lower90": lo,
            "upper90": hi,
        }
    )
    calibration.to_csv("reports/calibration_predictions.csv", index=False)
    fit_means = final.named_steps["scaler"].mean_.tolist()
    metadata = {
        "package_version": "1.0.0",
        "winner": winner,
        "config": cfg.to_dict(),
        "feature_columns": list(X.columns),
        "development_engine_ids": dev_ids.tolist(),
        "calibration_engine_ids": calibration_ids.tolist(),
        "conformal_radius": radius,
        "training_target": f"min(max_training_cycle - cycle, {cfg.rul_cap})",
        "test_target": "NASA official RUL, NOT capped",
        "units": "SI except documented native channels",
        "scaler_mean_train_only": fit_means,
        "versions": {
            n: importlib.metadata.version(n)
            for n in (
                "numpy",
                "pandas",
                "scikit-learn",
                "xgboost",
                "streamlit",
                "plotly",
            )
        },
    }
    bundle = {"pipeline": final, "metadata": metadata}
    # Freeze is persisted BEFORE official labels are read. Calibration is NEVER refitted.
    with open("models/rul_predictor_model.pkl.tmp", "wb") as handle:
        pickle.dump(bundle, handle, protocol=5)
    Path("models/rul_predictor_model.pkl.tmp").replace("models/rul_predictor_model.pkl")
    save_json("models/model_metadata.json", metadata)
    model_hash_before = sha256("models/rul_predictor_model.pkl")
    test_raw, audit_test = validate_and_clean(read_raw("data/raw/test_FD001.txt"))
    test = to_engineering_units(test_raw)
    Xt = make_features(test, cfg.windows)
    last_idx = test.groupby("engine_id", sort=True).tail(1).index.to_numpy()
    test_pred = predict_nonnegative(final, Xt.loc[last_idx])
    lower, upper = intervals(test_pred, radius)
    # No feature, model, policy threshold or hyperparameter is changed below.
    truth = read_test_truth("data/raw/RUL_FD001.txt", test.engine_id.unique())
    y_true = truth.loc[test.loc[last_idx, "engine_id"]].to_numpy()
    status, revisit = inspection_policy(lower)
    predictions = pd.DataFrame(
        {
            "engine_key": [f"test:{i}" for i in test.loc[last_idx, "engine_id"]],
            "engine_id": test.loc[last_idx, "engine_id"].to_numpy(),
            "last_cycle": test.loc[last_idx, "cycle"].to_numpy(),
            "rul_true": y_true,
            "rul_pred": test_pred,
            "error": test_pred - y_true,
            "lower90": lower,
            "upper90": upper,
            "status": status,
            "revisit_cycles": revisit,
        }
    )
    predictions.to_csv("reports/test_predictions.csv", index=False)
    # Additional capped metric is explicitly secondary, never substituted for NASA truth.
    official = regression_metrics(y_true, test_pred)
    official["rmse_ci95_engine_bootstrap"] = bootstrap_rmse(
        y_true, test_pred, cfg.seed, cfg.bootstrap_repetitions
    )
    official["interval_coverage90"] = float(
        ((y_true >= lower) & (y_true <= upper)).mean()
    )
    official["mean_interval_width"] = float((upper - lower).mean())
    official["conformal_radius"] = radius
    horizons = []
    for name, mask in (
        ("RUL<=30", y_true <= 30),
        ("30<RUL<=80", (y_true > 30) & (y_true <= 80)),
        ("RUL>80", y_true > 80),
    ):
        if mask.sum() > 1:
            horizons.append(
                {"horizon": name, **regression_metrics(y_true[mask], test_pred[mask])}
            )
    pd.DataFrame(horizons).to_csv("reports/metrics_by_horizon.csv", index=False)
    # Illustrative alert evaluation: no claimed causal reduction in AOG.
    policy = []
    for h in (15, 30, 45):
        urgent = y_true <= h
        alert = lower <= h + 10
        tp = int((urgent & alert).sum())
        fp = int((~urgent & alert).sum())
        fn = int((urgent & ~alert).sum())
        tn = int((~urgent & ~alert).sum())
        policy.append(
            {
                "horizon": h,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "recall": tp / max(1, tp + fn),
                "precision": tp / max(1, tp + fp),
                "illustrative_cost_units": fp + 20 * fn,
            }
        )
    pd.DataFrame(policy).to_csv("reports/policy_sensitivity.csv", index=False)
    reg = final.named_steps["regressor"]
    kept = final.named_steps["variance"].get_support()
    if hasattr(reg, "feature_importances_"):
        imp = reg.feature_importances_
    elif hasattr(reg, "coef_"):
        imp = np.abs(reg.coef_)
    else:
        imp = np.zeros(kept.sum())
    importance = pd.DataFrame(
        {"feature": np.array(X.columns)[kept], "importance": imp}
    ).sort_values("importance", ascending=False)
    importance.to_csv("reports/feature_importance.csv", index=False)
    # Persist clean data and fold-local transformed features using final DEVELOPMENT fit.
    labeled.assign(engine_key="train:" + labeled.engine_id.astype(str)).to_csv(
        "data/processed/train_clean_si.csv.gz", index=False
    )
    test.assign(engine_key="test:" + test.engine_id.astype(str)).to_csv(
        "data/processed/test_clean_si.csv.gz", index=False
    )
    transformer = final[:-1]
    feature_names = np.array(X.columns)[kept]
    for name, source, idx, prefix in (
        ("development", X, X_dev.index, "train"),
        ("calibration", X, calibration_snapshots, "train"),
        ("test", Xt, Xt.index, "test"),
    ):
        transformed = pd.DataFrame(
            transformer.transform(source.loc[idx]), columns=feature_names
        )
        parent = labeled if prefix == "train" else test
        transformed.insert(0, "cycle", parent.loc[idx, "cycle"].to_numpy())
        transformed.insert(
            0,
            "engine_key",
            prefix + ":" + parent.loc[idx, "engine_id"].astype(str).to_numpy(),
        )
        transformed.to_csv(f"data/processed/{name}_scaled.csv.gz", index=False)
    metrics = {
        "official_test": official,
        "secondary_capped_target": regression_metrics(
            np.minimum(y_true, cfg.rul_cap), test_pred
        ),
        "calibration": {
            **regression_metrics(cal_true, cal_pred),
            "n_engines": len(cal_true),
                "snapshot_protocol": f"uniform eligible observed cycle >={cfg.min_history}, one per engine, seed{cfg.seed + 1}",
        },
        "selected_model": winner,
        "cv_mean_rmse": float(comparison.iloc[0].cv_rmse),
        "cv_sd_rmse": float(comparison.iloc[0].cv_sd),
        "eda": eda,
        "execution_seconds": perf_counter() - started,
        "python_version": platform.python_version(),
        "model_sha256": model_hash_before,
    }
    save_json("reports/metrics.json", metrics)
    save_json(
        "reports/data_audit.json",
        {"train": audit_train, "test": audit_test, "source": provenance},
    )
    save_json(
        "reports/split_manifest.json",
        {
            "seed": cfg.seed,
            "namespace": "train:<id> and test:<id> are different trajectories",
            "development_ids": dev_ids.tolist(),
            "calibration_ids": calibration_ids.tolist(),
            "folds": fold_audit,
            "snapshot_indices": snapshots.tolist(),
            "official_test_ids": sorted(test.engine_id.unique().tolist()),
            "selection_criterion": "lowest mean CV RMSE against uncapped pseudo-endpoint RUL",
            "official_test_opened_after_model_freeze": True,
            "model_hash_unchanged_after_test": sha256("models/rul_predictor_model.pkl")
            == model_hash_before,
        },
    )
    result_figures(cv, learning, predictions, importance)
    print(
        json.dumps({"selected": winner, "official_test": official}, indent=2),
        flush=True,
    )
    return metrics
