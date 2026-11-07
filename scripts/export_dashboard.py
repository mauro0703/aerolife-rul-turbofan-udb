"""Exporta el panel web de evaluación FD001 y su distribución local."""

import base64
import gzip
import json
import hashlib
from pathlib import Path
import sys
import re
import numpy as np
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rul.data import SYMBOLS, UNITS, read_raw
from rul.inference import load_bundle, predict_history
from rul.artifacts import validate_artifacts


def main():
    """Usa el modelo congelado y los resultados oficiales de la evaluación."""
    reports = ROOT / "reports"
    dashboard = ROOT / "dashboard"
    dashboard.mkdir(parents=True, exist_ok=True)
    validate_artifacts(ROOT, full=True)
    bundle = load_bundle(ROOT / "models/rul_predictor_model.pkl")
    raw = read_raw(ROOT / "data/raw/test_FD001.txt")
    history, sensors, _ = predict_history(
        raw, bundle
    )
    official = pd.read_csv(reports / "test_predictions.csv")
    last = history.groupby("engine_id", sort=True).tail(1).reset_index(drop=True)
    reference = official.sort_values("engine_id").reset_index(drop=True)
    np.testing.assert_array_equal(last.engine_id, reference.engine_id)
    np.testing.assert_array_equal(last.cycle, reference.last_cycle)
    np.testing.assert_allclose(last[["rul_pred", "lower90", "upper90"]],
                               reference[["rul_pred", "lower90", "upper90"]],
                               rtol=0, atol=1e-9)
    channels = ["s2", "s3", "s4", "s7", "s11", "s12", "s20", "s21"]
    engines = {}
    for engine_id, group in history.groupby("engine_id"):
        cycles = group.cycle.tolist()
        if cycles != list(range(1, len(cycles) + 1)):
            raise ValueError(f"Historial de ciclos inválido para el motor {engine_id}")
        engines[str(engine_id)] = {
            "c": cycles,
            "p": group.rul_pred.round(5).tolist(),
            "l": group.lower90.round(5).tolist(),
            "u": group.upper90.round(5).tolist(),
            "s": {
                channel: sensors.loc[group.index, channel].round(6).tolist()
                for channel in channels
            },
        }

    columns = ["engine_id", "last_cycle", "rul_true", "rul_pred", "lower90", "upper90"]
    payload = {
        "engines": engines,
        "study": {"min_history": bundle["metadata"]["config"]["min_history"],
                  "nominal_coverage": 0.9, "model": bundle["metadata"]["winner"]},
        "metrics": json.loads((reports / "metrics.json").read_text())["official_test"],
        "comparison": pd.read_csv(reports / "model_comparison.csv").to_dict("records"),
        "official": official[columns].to_dict("records"),
        "channels": {
            channel: {"symbol": SYMBOLS[int(channel[1:]) - 1], "unit": UNITS[int(channel[1:]) - 1]}
            for channel in channels
        },
    }
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = base64.b64encode(gzip.compress(data.encode("utf-8"), mtime=0)).decode("ascii")
    (dashboard / "datos.js").write_text(f'window.AEROLIFE_DATA_GZIP = "{encoded}";\n', encoding="utf-8")

    # A deployment may update HTML before a browser's cached JS/CSS expires.
    # Content-derived URLs keep the four portable files synchronized online.
    html_path = dashboard / "index.html"
    html = re.sub(r"\?v=[0-9a-f]+", "", html_path.read_text(encoding="utf-8"))
    signatures = {}
    for name in ("panel.css", "panel.js", "datos.js"):
        signatures[name] = hashlib.sha256((dashboard / name).read_bytes()).hexdigest()[:12]
    package_version = hashlib.sha256(
        (html + json.dumps(signatures, sort_keys=True)).encode("utf-8")
    ).hexdigest()[:12]
    for name, signature in signatures.items():
        html = html.replace(f'"{name}"', f'"{name}?v={signature}"')
    html = html.replace('"Panel_Interactivo_AEROLIFE.zip"',
                        f'"Panel_Interactivo_AEROLIFE.zip?v={package_version}"')
    html_path.write_text(html, encoding="utf-8")

    with ZipFile(dashboard / "Panel_Interactivo_AEROLIFE.zip", "w") as archive:
        for name in ("index.html", "panel.css", "panel.js", "datos.js"):
            archive.writestr(
                ZipInfo(name),
                (dashboard / name).read_bytes(),
                compress_type=ZIP_DEFLATED,
                compresslevel=9,
            )


if __name__ == "__main__":
    main()
