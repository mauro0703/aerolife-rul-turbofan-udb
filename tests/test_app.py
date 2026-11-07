"""Functional UI smoke test through Streamlit's official AppTest API."""

from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_app_loads_and_filters_and_replays_past():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    assert not app.exception

    assert app.title[0].value.startswith("AEROLIFE")
    app.selectbox(key="engine").select(2).run(timeout=120)
    assert not app.exception
    app.slider(key="cycle").set_value(30).run(timeout=120)
    assert not app.exception
    app.slider(key="horizon").set_value(45).run(timeout=120)
    app.selectbox(key="status_filter").select("Revisión prioritaria").run(timeout=120)
    assert not app.exception


def test_empty_status_is_valid_and_motor_one_scenario_matches_policy():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    app.selectbox(key="status_filter").select("Seguimiento").run(timeout=120)
    assert not app.exception
    assert any("No hay motores" in item.value for item in app.info)
    app.selectbox(key="status_filter").select("Todos").run(timeout=120)
    app.slider(key="horizon").set_value(10).run(timeout=120)
    assert not app.exception
    # The native test motor 1 ends at cycle 31, with L=22.8 and floor(L-10)=12.
    assert app.selectbox(key="engine").value == 1
    values = [m.value for m in app.metric]
    assert "124.5" in values and "12" in values
    assert "Seguimiento" in values
