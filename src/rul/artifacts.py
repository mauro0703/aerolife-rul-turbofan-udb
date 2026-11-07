"""Integrity contract for a complete analytical run and a single writer.

Hashes detect accidental corruption or mixed runs; they do not authenticate a
pickle downloaded from an untrusted source. Readers never load uploaded pickle.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

MANIFEST = "models/artifact_manifest.json"
CORE_FILES = (
    "models/rul_predictor_model.pkl", "models/model_metadata.json",
    "reports/metrics.json", "reports/test_predictions.csv",
    "reports/model_comparison.csv", "reports/hypothesis_tests.csv",
    "reports/metrics_by_horizon.csv", "data/raw/test_FD001.txt",
)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_state(root, state, files=None):
    path = Path(root) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"schema_version": 1, "state": state,
                                    "sha256": files or {}}, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def complete_run(root="."):
    root = Path(root)
    # Formal documents and execution logs are versioned separately from model outputs.
    paths = set(CORE_FILES)
    for pattern in ("reports/*.csv", "reports/figures/*.png", "data/processed/*.gz"):
        paths.update(p.relative_to(root).as_posix() for p in root.glob(pattern))
    paths.update(("reports/split_manifest.json", "reports/data_audit.json"))
    files = {p: digest(root / p) for p in sorted(paths)}
    write_state(root, "complete", files)


def artifact_signature(root="."):
    """Cache key changes when any input to displayed predictions is replaced."""
    root = Path(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("state") != "complete":
        raise ValueError("La ejecución analítica está incompleta. Ejecute scripts/run_pipeline.py.")
    return tuple((p, (root / p).stat().st_mtime_ns, (root / p).stat().st_size)
                 for p in (MANIFEST, *CORE_FILES))


def validate_artifacts(root=".", full=False):
    root = Path(root)
    artifact_signature(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    paths = manifest["sha256"] if full else CORE_FILES
    for name in paths:
        if name not in manifest["sha256"] or digest(root / name) != manifest["sha256"][name]:
            raise ValueError(f"Resultado alterado o de otra ejecución: {name}.")
    return manifest


@contextmanager
def analytical_run(root="."):
    """OS releases the single-writer lock even if the training process is killed."""
    path = Path(root) / "models/.training.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if path.stat().st_size == 0:
        handle.write(b"0")
        handle.flush()
        handle.seek(0)
    locked = False
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise ValueError("Ya hay un entrenamiento en curso; espere a que termine.") from error
        write_state(root, "running")
        try:
            yield
            complete_run(root)
        except BaseException:
            write_state(root, "failed")
            raise
    finally:
        if locked:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
