"""Exporta el panel web de evaluación FD001 y su distribución local."""

import base64
import gzip
import json
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rul.data import SYMBOLS, UNITS, read_raw
from rul.inference import load_bundle, predict_history


def main():
    """Usa el modelo congelado y los resultados oficiales de la evaluación."""
    reports = ROOT / "reports"
    dashboard = ROOT / "dashboard"
    raw = read_raw(ROOT / "data/raw/test_FD001.txt")
    history, sensors, _ = predict_history(
        raw, load_bundle(ROOT / "models/rul_predictor_model.pkl")
    )
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
        "metrics": json.loads((reports / "metrics.json").read_text())["official_test"],
        "comparison": pd.read_csv(reports / "model_comparison.csv").to_dict("records"),
        "official": pd.read_csv(reports / "test_predictions.csv", usecols=columns).to_dict("records"),
        "channels": {
            channel: {"symbol": SYMBOLS[int(channel[1:]) - 1], "unit": UNITS[int(channel[1:]) - 1]}
            for channel in channels
        },
    }
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    encoded = base64.b64encode(gzip.compress(data.encode("utf-8"), mtime=0)).decode("ascii")
    (dashboard / "datos.js").write_text(f'window.AEROLIFE_DATA_GZIP = "{encoded}";\n', encoding="utf-8")

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
