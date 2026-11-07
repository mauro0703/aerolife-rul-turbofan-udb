"""Auditable ingestion, physical consistency, provenance and immutable raw data."""

from pathlib import Path
from urllib.request import urlopen, Request
import hashlib
import io
import json
import zipfile
import numpy as np
import pandas as pd

SOURCE_PAGE = "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/"
SOURCE_URL = "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"
FILES = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
SENSORS = [f"s{i}" for i in range(1, 22)]
SETTINGS = [f"setting_{i}" for i in range(1, 4)]
COLUMNS = ["engine_id", "cycle", *SETTINGS, *SENSORS]
# Symbol order and original units: Saxena et al. (2008), Table 2.
SYMBOLS = [
    "T2",
    "T24",
    "T30",
    "T50",
    "P2",
    "P15",
    "P30",
    "Nf",
    "Nc",
    "epr",
    "Ps30",
    "phi",
    "NRf",
    "NRc",
    "BPR",
    "farB",
    "htBleed",
    "Nf_dmd",
    "PCNfR_dmd",
    "W31",
    "W32",
]
UNITS = [
    "K",
    "K",
    "K",
    "K",
    "kPa abs",
    "kPa abs",
    "kPa abs",
    "rpm",
    "rpm",
    "1",
    "kPa abs",
    "(kg/s)/kPa",
    "rpm",
    "rpm",
    "1",
    "1",
    "native",
    "rpm",
    "native",
    "kg/s",
    "kg/s",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_fd001(payload, destination):
    """Read only known basenames, including NASA's nested ZIP; no extractall."""
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    archive = zipfile.ZipFile(io.BytesIO(payload))
    if not any(Path(n).name == FILES[0] for n in archive.namelist()):
        nested = [n for n in archive.namelist() if n.endswith("CMAPSSData.zip")]
        if len(nested) != 1:
            raise ValueError("El archivo no contiene CMAPSSData.zip ni FD001.")
        archive = zipfile.ZipFile(io.BytesIO(archive.read(nested[0])))
    payloads = {}
    for filename in (*FILES, "readme.txt"):
        matches = [n for n in archive.namelist() if Path(n).name == filename]
        if len(matches) != 1:
            raise ValueError(f"Archivo ausente o ambiguo: {filename}")
        payloads[filename] = archive.read(matches[0])
    # Validate every member before modifying the existing dataset.
    for filename, contents in payloads.items():
        temporary = folder / (filename + ".tmp")
        temporary.write_bytes(contents)
        temporary.replace(folder / filename)
    manifest = {
        "source_page": SOURCE_PAGE,
        "download_url": SOURCE_URL,
        "citation": "A. Saxena and K. Goebel (2008). Turbofan Engine Degradation Simulation Data Set. NASA Ames PCoE.",
        "sha256": {f: sha256(folder / f) for f in FILES},
    }
    (folder / "provenance.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def download_data(destination="data/raw", force=False):
    """Download official NASA-linked bytes, or verify bundled bytes offline."""
    folder = Path(destination)
    manifest_path = folder / "provenance.json"
    if (
        not force
        and manifest_path.exists()
        and all((folder / f).exists() for f in FILES)
    ):
        manifest = json.loads(manifest_path.read_text())
        for filename in FILES:
            if sha256(folder / filename) != manifest["sha256"][filename]:
                raise ValueError(
                    f"SHA-256 alterado: {filename}. Restaure los datos originales."
                )
        return manifest
    req = Request(SOURCE_URL, headers={"User-Agent": "UDB-FD001-Academic-Project/1.0"})
    with urlopen(req, timeout=90) as response:
        payload = response.read()
    return extract_fd001(payload, folder)


def read_raw(path):
    """Reject schema drift rather than silently discarding extra columns."""
    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(COLUMNS):
        raise ValueError(f"Se requieren 26 columnas; se recibieron {frame.shape[1]}.")
    frame.columns = COLUMNS
    for col in COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    return frame


def validate_and_clean(frame):
    """Fail on corrupt identity/time; map impossible sensors to missing, never clip wear."""
    if list(frame.columns) != COLUMNS:
        raise ValueError(
            "Esquema incorrecto: use engine_id, cycle, 3 settings y s1..s21."
        )
    if frame.empty:
        raise ValueError("El historial está vacío; incluya al menos un registro motor-ciclo.")
    clean = frame.copy(deep=True)
    for col in ["engine_id", "cycle"]:
        a = clean[col].to_numpy(dtype=float)
        if (not np.isfinite(a).all() or (a <= 0).any()
                or (a >= 2**63).any() or (a != np.floor(a)).any()):
            raise ValueError(f"{col} debe contener enteros positivos y finitos.")
        clean[col] = clean[col].astype(int)
    if clean.duplicated(["engine_id", "cycle"]).any():
        raise ValueError(
            "Clave motor-ciclo duplicada: revisar origen; no borrar silenciosamente."
        )
    clean = clean.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    gaps = clean.groupby("engine_id")["cycle"].diff().dropna()
    if (gaps != 1).any():
        raise ValueError(
            "Ciclos no consecutivos: resolver huecos antes de crear ventanas."
        )
    if not (clean.groupby("engine_id")["cycle"].first() == 1).all():
        raise ValueError("Se requiere el historial desde el ciclo 1 de cada motor.")
    input_missing = int(clean.isna().sum().sum())
    values = clean[SENSORS + SETTINGS].astype(float)
    nonfinite = ~np.isfinite(values) & values.notna()
    values = values.mask(nonfinite)
    impossible = values[SENSORS].le(0)
    values[SENSORS] = values[SENSORS].mask(impossible)
    clean[SENSORS + SETTINGS] = values
    audit = {
        "rows": len(clean),
        "engines": int(clean.engine_id.nunique()),
        "duplicate_keys": 0,
        "input_missing_cells": input_missing,
        "nonfinite_cells": int(nonfinite.sum().sum()),
        "nonpositive_sensor_cells": int(impossible.sum().sum()),
        "missing_after_physics": int(clean.isna().sum().sum()),
        "rows_removed": 0,
        "cycle_gaps": 0,
    }
    return clean, audit


def to_engineering_units(frame):
    """Deterministic SI conversion; absolute pressures remain absolute."""
    result = frame.copy()
    result[["s1", "s2", "s3", "s4"]] *= 5.0 / 9.0  # Rankine -> kelvin, no offset.
    result[["s5", "s6", "s7", "s11"]] *= 6.894757293168
    result["s12"] *= 0.45359237 / 6.894757293168
    result[["s20", "s21"]] *= 0.45359237
    return result


def add_training_targets(frame, cap=125):
    result = frame.copy()
    result["rul_true"] = (
        result.groupby("engine_id")["cycle"].transform("max") - result.cycle
    )
    result["rul_target"] = result.rul_true.clip(upper=cap)
    return result


def read_test_truth(path, engine_ids):
    """Official truth is one row per sorted test engine, at its LAST observed cycle."""
    labels = pd.read_csv(path, sep=r"\s+", header=None)
    ids = sorted(engine_ids)
    if labels.shape != (len(ids), 1) or ids != list(range(1, len(ids) + 1)):
        raise ValueError("Vector NASA o correspondencia con IDs inválidos.")
    values = labels.iloc[:, 0].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("RUL oficial inválida.")
    return pd.Series(values, index=ids, name="rul_true")
