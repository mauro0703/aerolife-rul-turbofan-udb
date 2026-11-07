"""Regression cases found during review: mixed artifacts and invalid histories."""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rul.artifacts import (CORE_FILES, MANIFEST, analytical_run, artifact_signature,
                           complete_run, validate_artifacts)
from rul.config import Config
from rul.data import COLUMNS, validate_and_clean, extract_fd001
from rul.evaluation import conformal_radius, inspection_policy
from rul.inference import load_bundle


def run_files(root):
    for name in (*CORE_FILES, "reports/split_manifest.json", "reports/data_audit.json"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test result")


@pytest.mark.parametrize("values", [
    {"alpha": 0.05}, {"alpha": np.nan}, {"windows": (5, 5)},
    {"windows": (0, 15)}, {"calibration_engines": 2}, {"cv_folds": 1},
    {"n_jobs": 0}, {"seed": -1}, {"bootstrap_repetitions": 0},
])
def test_invalid_experimental_contract_rejected(values):
    with pytest.raises(ValueError):
        Config(**values)


def test_empty_history_rejected_before_imputation():
    with pytest.raises(ValueError, match="vacío"):
        validate_and_clean(pd.DataFrame(columns=COLUMNS))


def test_identity_overflow_rejected_as_a_validation_error():
    frame = pd.DataFrame(np.ones((1, len(COLUMNS))), columns=COLUMNS)
    frame.engine_id = 1e100
    with pytest.raises(ValueError, match="enteros positivos"):
        validate_and_clean(frame)


@pytest.mark.parametrize("truth,pred,alpha", [([], [], 0.1), ([1], [0, 1], 0.1),
                                                      ([1], [np.nan], 0.1), ([1], [0], 0)])
def test_invalid_calibration_rejected(truth, pred, alpha):
    with pytest.raises(ValueError):
        conformal_radius(truth, pred, alpha)


@pytest.mark.parametrize("lower,kwargs", [([np.nan], {}), ([-1], {}),
    ([10], {"buffer": -1}), ([10], {"horizon": 0}), ([10], {"max_interval": 0})])
def test_invalid_policy_inputs_rejected(lower, kwargs):
    with pytest.raises(ValueError):
        inspection_policy(lower, **kwargs)


def test_failed_run_stays_unavailable_and_can_be_reproduced(tmp_path):
    run_files(tmp_path)
    complete_run(tmp_path)
    validate_artifacts(tmp_path, full=True)
    with pytest.raises(RuntimeError):
        with analytical_run(tmp_path):
            with pytest.raises(ValueError, match="incompleta"):
                artifact_signature(tmp_path)
            raise RuntimeError("interrupted")
    assert json.loads((tmp_path / MANIFEST).read_text())["state"] == "failed"
    with analytical_run(tmp_path):
        run_files(tmp_path)
    validate_artifacts(tmp_path, full=True)


def test_second_writer_rejected_without_replacing_active_state(tmp_path):
    run_files(tmp_path)
    with analytical_run(tmp_path):
        with pytest.raises(ValueError, match="en curso"):
            with analytical_run(tmp_path):
                pytest.fail("Second writer entered")
        assert json.loads((tmp_path / MANIFEST).read_text())["state"] == "running"


def test_changed_model_invalidates_cache_and_is_rejected_before_unpickle(tmp_path, monkeypatch):
    run_files(tmp_path)
    complete_run(tmp_path)
    before = artifact_signature(tmp_path)
    path = tmp_path / "models/rul_predictor_model.pkl"
    path.write_bytes(b"modified model payload")
    assert artifact_signature(tmp_path) != before
    monkeypatch.setattr("rul.inference.pickle.load", lambda _: pytest.fail("unpickled corrupt model"))
    with pytest.raises(ValueError, match="alterado"):
        load_bundle(path)


def test_incomplete_download_does_not_overwrite_existing_inputs(tmp_path):
    import io
    from zipfile import ZipFile
    original = tmp_path / "train_FD001.txt"
    original.write_bytes(b"existing train")
    payload = io.BytesIO()
    with ZipFile(payload, "w") as archive:
        archive.writestr("train_FD001.txt", b"new train")
    with pytest.raises(ValueError, match="ausente"):
        extract_fd001(payload.getvalue(), tmp_path)
    assert original.read_bytes() == b"existing train"
