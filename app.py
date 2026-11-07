"""Interactive FD001 engineering dashboard. Launch: streamlit run app.py."""

from pathlib import Path
import io
import json
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from rul.data import COLUMNS, SENSORS, SYMBOLS, UNITS, read_raw
from rul.inference import load_bundle, predict_history
from rul.evaluation import inspection_policy
from rul.artifacts import artifact_signature, validate_artifacts

st.set_page_config(page_title="AEROLIFE | Turbofan RUL", page_icon="✈", layout="wide")
st.markdown(
    """<style>
.stApp {background:#f5f8fc;} .block-container {padding-top:2rem;max-width:1450px;}
h1,h2,h3 {color:#12314b;} div[data-testid="stMetric"] {background:white;border:1px solid #e0e8f0;border-radius:12px;padding:16px;}
</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def model(signature):
    return load_bundle(ROOT / "models/rul_predictor_model.pkl")


@st.cache_data
def artifacts(signature):
    validate_artifacts(ROOT)
    return (
        json.loads((ROOT / "reports/metrics.json").read_text()),
        pd.read_csv(ROOT / "reports/model_comparison.csv"),
        pd.read_csv(ROOT / "reports/test_predictions.csv"),
        pd.read_csv(ROOT / "reports/hypothesis_tests.csv"),
    )


@st.cache_data
def test_history(signature):
    return predict_history(read_raw(ROOT / "data/raw/test_FD001.txt"), model(signature))


@st.cache_data
def upload_history(payload, signature):
    frame = pd.read_csv(io.BytesIO(payload))
    if len(frame) > 100000:
        raise ValueError("Máximo 100 000 registros por carga.")
    if set(frame.columns) != set(COLUMNS):
        raise ValueError(
            "El CSV debe contener exactamente engine_id, cycle, setting_1..3 y s1..s21."
        )
    frame = frame[COLUMNS].apply(pd.to_numeric, errors="raise")
    return predict_history(frame, model(signature))


st.caption("UNIVERSIDAD DON BOSCO  /  DAS172  /  PROGNÓSTICO AERONÁUTICO")
st.title("AEROLIFE · Vida remanente del turbofán")
st.markdown(
    "Telemetría del motor → estimación de RUL → priorización de inspecciones del HPC."
)
st.caption(
    "Demostración con datos sintéticos NASA FD001. Los intervalos son marginales y no sustituyen el programa de mantenimiento aprobado."
)

try:
    signature = artifact_signature(ROOT)
    bundle = model(signature)
    metrics, comparison, official, hypotheses = artifacts(signature)
except (FileNotFoundError, ValueError) as error:
    st.error(f"No están completos los resultados: {error}")
    st.code("python scripts/run_pipeline.py")
    st.stop()

with st.sidebar:
    st.header("Escenario de planificación")
    horizon = st.slider("Horizonte de interés [ciclos]", 5, 60, 30, 5, key="horizon")
    buffer = st.slider("Reserva de planificación [ciclos]", 0, 25, 10, 1, key="buffer")
    maximum = st.slider(
        "Máximo hasta próxima revisión [ciclos]", 1, 40, 20, 1, key="maximum"
    )
    source = st.radio(
        "Fuente", ["Flota NASA FD001", "CSV propio (formato FD001)"], key="source"
    )
    st.caption(f"Modelo: {bundle['metadata']['winner']} · semilla {bundle['metadata']['config']['seed']}")
    st.caption(
        "Las decisiones cambian con los controles; el modelo permanece congelado."
    )

try:
    if source == "Flota NASA FD001":
        history, si, audit = test_history(signature)
    else:
        upload = st.file_uploader(
            "Historial en unidades nativas NASA", type="csv", max_upload_size=10
        )
        template = read_raw(ROOT / "data/raw/test_FD001.txt").query("engine_id == 1")
        st.download_button(
            "Descargar CSV de ejemplo",
            template.to_csv(index=False),
            "ejemplo_historial_fd001.csv",
            "text/csv",
        )
        if upload is None:
            st.info(
                "Cargue un historial que empiece en el ciclo 1 y tenga ciclos consecutivos por motor."
            )
            st.stop()
        history, si, audit = upload_history(upload.getvalue(), signature)
except (ValueError, TypeError, pd.errors.ParserError) as error:
    st.error(f"No se puede analizar este historial: {error}")
    st.stop()

history = history.copy()
if audit["missing_after_physics"]:
    st.warning(
        f"Se imputaron {audit['missing_after_physics']} celdas faltantes o físicamente inválidas "
        "con medianas del entrenamiento. Revise la calidad de la telemetría."
    )
history["status"], history["revisit_cycles"] = inspection_policy(
    history.lower90, horizon, buffer, maximum
)
fleet = history.groupby("engine_id", sort=True).tail(1).copy()
status_options = [
    "Todos",
    "Revisión prioritaria",
    "Planificar inspección",
    "Seguimiento",
]
filter_status = st.selectbox(
    "Filtrar estado de planificación", status_options, key="status_filter"
)
if filter_status != "Todos":
    fleet = fleet[fleet.status == filter_status]

summary, telemetry, evaluation, method = st.tabs(
    [
        "01 · Flota y decisiones",
        "02 · Motor y sensores",
        "03 · Validación",
        "04 · Método y trazabilidad",
    ]
)
with summary:
    a, b, c, d = st.columns(4)
    a.metric("Motores en vista", len(fleet))
    b.metric(
        "Revisión prioritaria", int((fleet.status == "Revisión prioritaria").sum())
    )
    c.metric(
        "Planificar inspección", int((fleet.status == "Planificar inspección").sum())
    )
    d.metric("Seguimiento", int((fleet.status == "Seguimiento").sum()))
    st.subheader("Prioridad basada en el límite inferior de RUL")
    st.caption(
        "Reserva = margen de planificación. La salida indica cuándo volver a evaluar, no una autorización para operar hasta ese límite."
    )
    if fleet.empty:
        st.info("No hay motores con el estado seleccionado.")
    else:
        ranked = fleet.sort_values(["lower90", "rul_pred", "engine_id"])
        fig = px.bar(
            ranked,
            x="engine_id",
            y="rul_pred",
            color="status",
            error_y=(ranked.upper90 - ranked.rul_pred),
            error_y_minus=(ranked.rul_pred - ranked.lower90),
            labels={
                "engine_id": "Motor de prueba",
                "rul_pred": "RUL estimada [ciclos]",
                "status": "Estado",
            },
            color_discrete_map={
                "Revisión prioritaria": "#b94e44",
                "Planificar inspección": "#d88736",
                "Seguimiento": "#158c83",
            },
            hover_data=["cycle", "lower90", "upper90", "revisit_cycles"],
        )
        fig.update_xaxes(type="category")
        fig.update_layout(
            template="plotly_white", height=390, legend=dict(orientation="h", y=1.15)
        )
        st.plotly_chart(fig, width="stretch")
        columns = [
            "engine_id",
            "cycle",
            "rul_pred",
            "lower90",
            "upper90",
            "status",
            "revisit_cycles",
            "short_history",
            "features_beyond_6sigma",
        ]
        st.dataframe(
            ranked[columns].round(2), hide_index=True, width="stretch",
            column_config={
                "engine_id": "Motor", "cycle": "Ciclo observado",
                "rul_pred": "RUL estimada [ciclos]", "lower90": "Límite inferior 90%",
                "upper90": "Límite superior 90%", "status": "Estado",
                "revisit_cycles": "Reevaluar en [ciclos]", "short_history": "Historial corto",
                "features_beyond_6sigma": st.column_config.NumberColumn(
                    "Variables con |z| > 6",
                    help="Conteo descriptivo en la escala del entrenamiento. Señala telemetría para revisar; no es una prueba formal de deriva ni un límite del fabricante.",
                ),
            },
        )
        st.download_button(
            "Exportar flota filtrada",
            ranked[columns].to_csv(index=False),
            "priorizacion_fd001.csv",
            "text/csv",
        )

with telemetry:
    engine = st.selectbox("Motor", sorted(history.engine_id.unique()), key="engine")
    h = history[history.engine_id == engine].copy()
    end = int(h.cycle.max())
    selected = (
        st.slider("Ciclo observado (solo se usa el pasado)", 1, end, end, key="cycle")
        if end > 1
        else 1
    )
    point = h[h.cycle == selected].iloc[0]
    a, b, c, d = st.columns(4)
    a.metric("RUL estimada [ciclos]", f"{point.rul_pred:.1f}")
    b.metric(
        "Intervalo nominal 90% [ciclos]", f"{point.lower90:.1f} – {point.upper90:.1f}"
    )
    c.metric("Estado", point.status)
    d.metric("Reevaluar en ≤ [ciclos]", int(point.revisit_cycles))
    if point.short_history:
        st.info(
            f"Historial menor a {bundle['metadata']['config']['min_history']} ciclos: el protocolo de calibración no respalda estos intervalos."
        )
    g = h[h.cycle <= selected]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=g.cycle,
            y=g.upper90,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g.cycle,
            y=g.lower90,
            fill="tonexty",
            fillcolor="rgba(36,107,160,.16)",
            line=dict(width=0),
            name="Intervalo nominal 90%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=g.cycle,
            y=g.rul_pred,
            line=dict(color="#246ba0", width=2),
            name="RUL estimada",
        )
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Ciclo observado",
        yaxis_title="RUL [ciclos]",
        height=340,
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "La banda no garantiza cobertura simultánea de toda la trayectoria; "
        f"los ciclos menores a {bundle['metadata']['config']['min_history']} son exploratorios."
    )
    sensor = st.selectbox(
        "Canal de telemetría",
        SENSORS,
        index=2,
        format_func=lambda s: f"{s} · {SYMBOLS[int(s[1:]) - 1]}",
        key="sensor",
    )
    sensor_unit = UNITS[int(sensor[1:]) - 1]
    trace = si[(si.engine_id == engine) & (si.cycle <= selected)].copy()
    trace["media_15"] = trace[sensor].rolling(15, min_periods=1).mean()
    fig = go.Figure(
        [
            go.Scatter(
                x=trace.cycle,
                y=trace[sensor],
                name="Medición",
                line=dict(color="#aac2d4", width=1),
            ),
            go.Scatter(
                x=trace.cycle,
                y=trace.media_15,
                name="Media causal 15",
                line=dict(color="#158c83", width=2),
            ),
        ]
    )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Ciclo",
        yaxis_title=f"{SYMBOLS[int(sensor[1:]) - 1]} [{sensor_unit}]",
        height=300,
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, width="stretch")

with evaluation:
    st.subheader("Prueba oficial independiente · un punto final por motor")
    m = metrics["official_test"]
    a, b, c, d = st.columns(4)
    a.metric("RMSE [ciclos]", f"{m['rmse']:.2f}")
    b.metric("MAE [ciclos]", f"{m['mae']:.2f}")
    c.metric("R²", f"{m['r2']:.3f}")
    d.metric("Cobertura observada", f"{100 * m['interval_coverage90']:.0f}%")
    st.caption(
        f"NASA score = {m['nasa_score']:.2f}; IC bootstrap 95% del RMSE = [{m['rmse_ci95_engine_bootstrap'][0]:.2f}, {m['rmse_ci95_engine_bootstrap'][1]:.2f}] ciclos. La verdad NASA se evalúa sin limitar a 125."
    )
    st.dataframe(comparison.round(3), hide_index=True, width="stretch")
    fig = px.scatter(
        official,
        x="rul_true",
        y="rul_pred",
        hover_data=["engine_id"],
        labels={
            "rul_true": "RUL NASA sin recortar [ciclos]",
            "rul_pred": "RUL estimada [ciclos]",
        },
    )
    lim = float(max(official.rul_true.max(), official.rul_pred.max()))
    fig.add_shape(
        type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(dash="dash", color="gray")
    )
    fig.update_layout(template="plotly_white", height=360)
    st.plotly_chart(fig, width="stretch")
    alert = official.lower90 <= horizon + buffer
    urgent = official.rul_true <= horizon
    tp = int((alert & urgent).sum())
    fp = int((alert & ~urgent).sum())
    fn = int((~alert & urgent).sum())
    st.info(
        f"Anchura media: {m['mean_interval_width']:.2f} ciclos. En la prueba oficial, "
        f"H={horizon} y B={buffer} generan {int(alert.sum())} alertas: "
        f"{tp} verdaderos positivos, {fp} falsos positivos y {fn} falsos negativos. "
        "La alta cobertura requiere una banda amplia y no demuestra utilidad operacional."
    )
    st.caption(
        "Estos resultados corresponden a la flota NASA original; no evalúan el CSV propio ni se recalculan con los filtros de planificación."
    )
    st.dataframe(
        pd.read_csv(ROOT / "reports/metrics_by_horizon.csv").round(3),
        hide_index=True,
        width="stretch",
    )
    st.download_button(
        "Exportar evaluación oficial",
        official.to_csv(index=False),
        "evaluacion_oficial_fd001.csv",
        "text/csv",
    )

with method:
    st.subheader("Contrato del experimento")
    st.markdown(
        "**80 motores de desarrollo** → **GroupKFold de 5 folds** → selección por RMSE → ajuste final en los 80 motores → **20 motores de calibración** → congelación → **100 motores oficiales de prueba**."
    )
    st.markdown(
        "El imputador, selector de varianza y escalador se ajustan dentro de cada fold. Las ventanas de 5, 15 y 30 ciclos son causales. El ID del motor y su vida total nunca son variables predictoras."
    )
    st.markdown(
        "**Objetivo de entrenamiento:** RUL limitada a 125 ciclos para representar la fase inicial estable. **Evaluación principal:** RUL oficial completa. La incertidumbre usa 20 errores de calibración, uno por motor; su traslado al protocolo NASA puede cambiar la cobertura."
    )
    st.dataframe(hypotheses.round(6), hide_index=True, width="stretch")
    st.json(audit, expanded=False)
    st.caption(
        "Contrastes Wilcoxon pareados con corrección Holm: una diferencia por motor. Asociación descriptiva, sin causalidad demostrada."
    )
    with st.expander("Diccionario de canales en unidades de visualización"):
        st.dataframe(
            pd.DataFrame({"canal": SENSORS, "símbolo": SYMBOLS, "unidad": UNITS}),
            hide_index=True,
        )
    st.markdown(
        "[Fuente de los datos: NASA Ames PCoE](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)"
    )
    report = ROOT / "reports/Reporte_Tecnico_RUL_Turbofan.pdf"
    if report.exists():
        st.download_button(
            "Descargar reporte técnico",
            report.read_bytes(),
            report.name,
            "application/pdf",
        )
