"""Functional UI smoke test through Streamlit's official AppTest API."""

from streamlit.testing.v1 import AppTest


def test_app_loads_and_filters_and_replays_past():
    app = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value.startswith("AEROLIFE")
    app.selectbox(key="engine").select(2).run(timeout=120)
    assert not app.exception
    app.slider(key="cycle").set_value(30).run(timeout=120)
    assert not app.exception
    app.slider(key="horizon").set_value(45).run(timeout=120)
    app.selectbox(key="status_filter").select("Revisión prioritaria").run(timeout=120)
    assert not app.exception
