"""Meaningful failure-mode tests: leakage, causal windows and NASA evaluation."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from rul.data import (
    COLUMNS,
    read_raw,
    validate_and_clean,
    to_engineering_units,
    add_training_targets,
    download_data,
    read_test_truth,
)
from rul.features import make_features, choose_snapshots
from rul.modeling import make_pipeline
from rul.evaluation import nasa_score, conformal_radius, inspection_policy
from rul.inference import load_bundle, predict_history


def fixture_frame():
    data = pd.DataFrame(np.ones((80, 26)), columns=COLUMNS)
    data.engine_id = np.repeat([1, 2], 40)
    data.cycle = np.tile(np.arange(1, 41), 2)
    data.s3 = np.arange(80, dtype=float) + 1000
    return data


def test_future_values_cannot_change_past_features():
    frame = fixture_frame()
    before = make_features(frame)
    changed = frame.copy()
    changed.loc[(changed.engine_id == 1) & (changed.cycle > 20), "s3"] += 1e6
    after = make_features(changed)
    pd.testing.assert_frame_equal(before.loc[:19], after.loc[:19])
    pd.testing.assert_frame_equal(before.loc[40:], after.loc[40:])
    assert not before.loc[39].equals(after.loc[39])


def test_features_exclude_id_lifetime_and_targets():
    labeled = add_training_targets(fixture_frame())
    features = make_features(labeled)
    assert not {"engine_id", "rul_true", "rul_target", "max_cycle"} & set(
        features.columns
    )
    assert labeled.groupby("engine_id").rul_true.last().eq(0).all()
    assert labeled.rul_target.max() <= 125


def test_preprocessing_statistics_unchanged_by_validation_shift():
    train = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [2.0, 6.0, 8.0, 10.0]})
    test = pd.DataFrame({"a": [10000.0, np.nan], "b": [10000.0, 20000.0]})
    pipe = make_pipeline(Ridge()).fit(train, [3.0, 5.0, 6.0, 9.0])
    median = pipe.named_steps["imputer"].statistics_.copy()
    mean = pipe.named_steps["scaler"].mean_.copy()
    pipe.predict(test)
    np.testing.assert_allclose(pipe.named_steps["imputer"].statistics_, median)
    np.testing.assert_allclose(pipe.named_steps["scaler"].mean_, mean)
    assert median[0] == 2
    assert mean[0] == 2.25


def test_invalid_identity_duplicate_and_time_gaps_rejected():
    frame = fixture_frame()
    with pytest.raises(ValueError, match="duplicada"):
        validate_and_clean(pd.concat([frame, frame.iloc[:1]], ignore_index=True))
    with pytest.raises(ValueError, match="consecutivos"):
        validate_and_clean(frame.drop(index=5))
    bad = frame.copy()
    bad.loc[0, "engine_id"] = 0
    with pytest.raises(ValueError, match="enteros positivos"):
        validate_and_clean(bad)


def test_physical_outliers_become_missing_without_destroying_wear():
    frame = fixture_frame()
    frame.loc[0, "s3"] = -2
    frame.loc[1, "s3"] = 100000
    clean, audit = validate_and_clean(frame)
    assert np.isnan(clean.loc[0, "s3"])
    assert clean.loc[1, "s3"] == 100000
    assert audit["rows_removed"] == 0


def test_rankine_and_absolute_pressure_conversion():
    frame = fixture_frame()
    frame.s3 = 900.0
    frame.s7 = 100.0
    si = to_engineering_units(frame)
    assert si.s3.iloc[0] == 500.0
    assert si.s7.iloc[0] == pytest.approx(689.4757293168)
    assert frame.s3.iloc[0] == 900.0  # Input not mutated.


def test_nasa_score_penalizes_dangerous_overestimate_more():
    assert nasa_score([50], [60]) > nasa_score([50], [40])
    assert nasa_score([50], [50]) == 0


def test_conformal_uses_finite_sample_rank():
    assert conformal_radius(np.arange(1, 21), np.zeros(20), 0.1) == 19
    assert np.isinf(conformal_radius([1, 2], [0, 0], 0.01))


def test_pseudocensoring_is_deterministic_and_not_failure_only():
    frame = add_training_targets(fixture_frame())
    a = choose_snapshots(frame, 43, 30)
    b = choose_snapshots(frame, 43, 30)
    np.testing.assert_array_equal(a, b)
    assert len(a) == 2 and frame.loc[a, "rul_true"].gt(0).all()


def test_policy_cannot_schedule_after_lower_bound_minus_reserve():
    lower = np.array([0.0, 10.0, 20.0, 100.0])
    status, revisit = inspection_policy(lower)
    assert (revisit <= np.maximum(0, lower - 10)).all()
    assert status[0] == "Revisión prioritaria"


def test_saved_split_has_no_engine_overlap_and_model_frozen():
    split = json.loads(Path("reports/split_manifest.json").read_text())
    dev = set(split["development_ids"])
    cal = set(split["calibration_ids"])
    assert len(dev) == 80 and len(cal) == 20 and not dev & cal
    for fold in split["folds"]:
        tr, va = set(fold["fit_engine_ids"]), set(fold["validation_engine_ids"])
        assert not tr & va and (tr | va) == dev and not (tr | va) & cal
    assert split["official_test_opened_after_model_freeze"]
    assert split["model_hash_unchanged_after_test"]


def test_saved_scaler_matches_development_only_statistics():
    from sklearn.impute import SimpleImputer
    from sklearn.feature_selection import VarianceThreshold

    bundle = load_bundle()
    raw, _ = validate_and_clean(read_raw("data/raw/train_FD001.txt"))
    x = make_features(to_engineering_units(raw))
    keep = raw.engine_id.isin(bundle["metadata"]["development_engine_ids"])
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    transformed = imputer.fit_transform(x.loc[keep])
    selector = VarianceThreshold(1e-12)
    transformed = selector.fit_transform(transformed)
    np.testing.assert_allclose(
        bundle["pipeline"].named_steps["imputer"].statistics_, imputer.statistics_
    )
    np.testing.assert_allclose(
        bundle["pipeline"].named_steps["scaler"].mean_, transformed.mean(axis=0)
    )


def test_saved_prediction_matches_fresh_inference_without_labels():
    raw = read_raw("data/raw/test_FD001.txt").query("engine_id==1")
    expected = pd.read_csv("reports/test_predictions.csv").query("engine_id==1").iloc[0]
    result, _, _ = predict_history(raw, load_bundle())
    assert result.rul_pred.iloc[-1] == pytest.approx(expected.rul_pred, rel=1e-6)


def test_raw_checksums_and_official_truth_alignment():
    download_data()  # Verifies local SHA-256; no network.
    truth = read_test_truth("data/raw/RUL_FD001.txt", range(1, 101))
    p = pd.read_csv("reports/test_predictions.csv")
    np.testing.assert_array_equal(p.rul_true, truth.values)
    assert p.engine_key.str.startswith("test:").all()
