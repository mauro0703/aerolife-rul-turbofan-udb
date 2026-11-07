"""Verify the delivered offline package, cache versions and engine endpoints."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd


def test_portable_panel_matches_frozen_predictions_and_versioned_assets():
    folder = Path(__file__).resolve().parents[1] / "dashboard"
    html = (folder / "index.html").read_text(encoding="utf-8")
    for name in ("panel.css", "panel.js", "datos.js"):
        signature = hashlib.sha256((folder / name).read_bytes()).hexdigest()[:12]
        assert f'"{name}?v={signature}"' in html
    with ZipFile(folder / "Panel_Interactivo_AEROLIFE.zip") as archive:
        assert sorted(archive.namelist()) == ["datos.js", "index.html", "panel.css", "panel.js"]
        for name in archive.namelist():
            assert archive.read(name) == (folder / name).read_bytes()
    payload = json.loads(gzip.decompress(base64.b64decode(
        (folder / "datos.js").read_text().split('"')[1])))
    official = pd.read_csv(folder.parent / "reports/test_predictions.csv")
    assert len(payload["engines"]) == len(payload["official"]) == len(official) == 100
    for row in official.itertuples():
        engine = payload["engines"][str(row.engine_id)]
        assert engine["c"][-1] == row.last_cycle
        np.testing.assert_allclose(
            [engine["p"][-1], engine["l"][-1], engine["u"][-1]],
            [row.rul_pred, row.lower90, row.upper90], rtol=0, atol=6e-6)
