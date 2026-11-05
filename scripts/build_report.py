"""Build a 10-page evidence-driven report and Markdown from executed results."""

from pathlib import Path
from html import escape, unescape
import json
import re
import fitz
import pandas as pd
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    KeepTogether,
)

ROOT = Path(".")
R = Path("reports")
metrics = json.loads((R / "metrics.json").read_text())
meta = json.loads(Path("models/model_metadata.json").read_text())
audit = json.loads((R / "data_audit.json").read_text())
split = json.loads((R / "split_manifest.json").read_text())
cv = pd.read_csv(R / "model_comparison.csv")
hyp = pd.read_csv(R / "hypothesis_tests.csv")
policy = pd.read_csv(R / "policy_sensitivity.csv")
by_horizon = pd.read_csv(R / "metrics_by_horizon.csv")
pred = pd.read_csv(R / "test_predictions.csv")
learning = pd.read_csv(R / "learning_curve.csv")
m = metrics["official_test"]
eda = metrics["eda"]
rad = m["conformal_radius"]
winner = metrics["selected_model"]
f = lambda x, n=2: f"{x:.{n}f}"
NAVY = colors.HexColor("#12314b")
BLUE = colors.HexColor("#246ba0")
LIGHT = colors.HexColor("#edf3f8")
for variant, suffix in [
    ("", ""),
    ("-Bold", "-Bold"),
    ("-Italic", "-Italic"),
    ("-BoldItalic", "-BoldItalic"),
]:
    pdfmetrics.registerFont(
        TTFont(
            "RULSerif" + variant, str(Path("assets/fonts") / f"DejaVuSerif{suffix}.ttf")
        )
    )
pdfmetrics.registerFontFamily(
    "RULSerif",
    normal="RULSerif",
    bold="RULSerif-Bold",
    italic="RULSerif-Italic",
    boldItalic="RULSerif-BoldItalic",
)
styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        name="TitleRUL",
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=NAVY,
        spaceAfter=13,
    )
)
styles.add(
    ParagraphStyle(
        name="SubRUL",
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="BodyRUL",
        fontName="RULSerif",
        fontSize=9.8,
        leading=12.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        name="SmallRUL",
        fontName="RULSerif",
        fontSize=8.5,
        leading=10.5,
        spaceAfter=6,
        textColor=NAVY,
    )
)
styles.add(
    ParagraphStyle(
        name="CellRUL", fontName="Helvetica", fontSize=8.1, leading=10.5, spaceAfter=0
    )
)
styles.add(
    ParagraphStyle(
        name="CodeRUL",
        fontName="Courier",
        fontSize=8.2,
        leading=11,
        backColor=LIGHT,
        borderPadding=7,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        name="CaptionRUL",
        fontName="Helvetica-Oblique",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#536777"),
        spaceAfter=9,
    )
)
PAGE = []


def new_page(title):
    page = {"title": title, "blocks": []}
    PAGE.append(page)
    return page["blocks"]


def p(page, text):
    page.append(("p", text))


def sub(page, text):
    page.append(("sub", text))


def table(page, headers, rows, widths=None):
    page.append(("table", (headers, rows, widths)))


def fig(page, name, caption, width=490):
    page.append(("fig", (name, caption, width)))


def code(page, text):
    page.append(("code", text))


b = new_page("1. Problema, alcance y datos")
sub(b, "Resumen ejecutivo")
p(
    b,
    f"Se desarrolló un sistema reproducible para estimar la vida útil remanente (RUL) de turbofanes simulados mediante NASA C-MAPSS FD001. La selección se realizó con validación cruzada agrupada por motor y la incertidumbre se calibró con unidades reservadas. El modelo {winner} obtuvo RMSE de <b>{f(m['rmse'])} ciclos</b>, MAE de <b>{f(m['mae'])} ciclos</b> y R² de <b>{f(m['r2'], 3)}</b> en 100 motores oficiales, sin recortar su verdad de referencia. La cobertura observada del intervalo nominal de 90% fue {f(100 * m['interval_coverage90'], 1)}%.",
)
sub(b, "1.1. Pregunta de ingeniería y objetivos")
p(
    b,
    "¿Puede la telemetría multivariable ordenar prioridades de revisión del compresor de alta presión (HPC) antes del límite de vida simulado? El objetivo general es construir y evaluar una estimación de RUL que apoye escenarios de inspección boroscópica. Los objetivos específicos son validar la calidad física de los registros, contrastar tendencias por motor, comparar modelos sin fuga y convertir la estimación y su incertidumbre en una decisión trazable.",
)
p(
    b,
    "El uso de RUL responde a una necesidad de mantenimiento basado en condición: anticipar deterioro y preparar recursos para disminuir exposición a paradas no programadas (AOG). El dataset no contiene eventos AOG, inspecciones, costos ni decisiones de mantenimiento; por tanto, el proyecto no estima un ahorro real ni demuestra que una política reduzca AOG. La priorización es un escenario de ingeniería explícito.",
)
sub(b, "1.2. Fuente, población y alcance físico [1-3]")
table(
    b,
    ["Elemento", "Contenido verificado"],
    [
        [
            "Origen",
            "NASA Ames PCoE; descarga desde el enlace publicado por NASA. TXT originales y SHA-256 incluidos.",
        ],
        [
            "Entrenamiento",
            f"{audit['train']['rows']:,} filas; 100 motores hasta el fallo.",
        ],
        [
            "Prueba oficial",
            f"{audit['test']['rows']:,} filas; 100 motores truncados y 100 RUL verdaderas del último ciclo.",
        ],
        ["Esquema", "26 columnas: ID, ciclo, 3 ajustes y 21 canales de sensores."],
        [
            "FD001",
            "Una condición operacional (nivel del mar) y un modo de fallo: degradación del HPC.",
        ],
        [
            "Unidad de análisis",
            "Motor independiente; cada fila es una observación de ciclo simulado.",
        ],
    ],
    [100, 390],
)
p(
    b,
    "FD001 representa pérdida progresiva de capacidad de flujo y eficiencia del HPC dentro de una simulación termodinámica con control y ruido. La respuesta de temperatura, presión y caudal puede revelar esa degradación, pero una asociación predictiva no identifica daño de un álabe ni demuestra una relación causal. FD001 no es un conjunto de fallos de turbina de alta presión (HPT).",
)
p(
    b,
    "No se equipara un ciclo del simulador a una hora de vuelo ni se extrapolan los resultados a todas las condiciones de misión. Los ajustes operacionales se conservan como canales nativos; no se les asignan altitud o Mach con unidades que el TXT no documenta de forma inequívoca.",
)

b = new_page("2. ETL y consistencia física")
sub(b, "2.1. Integridad antes del modelado")
p(
    b,
    "La ingesta lee separadores de espacio y exige exactamente 26 columnas numéricas. Se conserva una copia inmutable en data/raw. El origen y el SHA-256 de cada TXT constan en provenance.json. Se comprueban ID y ciclo enteros positivos, clave motor-ciclo única, inicio en ciclo 1 y continuidad temporal. Un duplicado o un hueco provoca un error visible; no se elimina silenciosamente.",
)
table(
    b,
    ["Control", "Train", "Test"],
    [
        [
            "Nulos originales",
            str(audit["train"]["input_missing_cells"]),
            str(audit["test"]["input_missing_cells"]),
        ],
        [
            "Claves duplicadas",
            str(audit["train"]["duplicate_keys"]),
            str(audit["test"]["duplicate_keys"]),
        ],
        [
            "Valores no finitos",
            str(audit["train"]["nonfinite_cells"]),
            str(audit["test"]["nonfinite_cells"]),
        ],
        [
            "Sensores no positivos",
            str(audit["train"]["nonpositive_sensor_cells"]),
            str(audit["test"]["nonpositive_sensor_cells"]),
        ],
        [
            "Huecos entre ciclos",
            str(audit["train"]["cycle_gaps"]),
            str(audit["test"]["cycle_gaps"]),
        ],
        [
            "Filas descartadas",
            str(audit["train"]["rows_removed"]),
            str(audit["test"]["rows_removed"]),
        ],
    ],
    [290, 100, 100],
)
sub(b, "2.2. Unidades y plausibilidad [3, tabla 2]")
table(
    b,
    ["Canales", "Magnitud", "Conversión determinista"],
    [
        ["s1-s4", "Temperatura absoluta", "K = °R × 5/9; no se resta 273.15."],
        ["s5,s6,s7,s11", "Presión absoluta", "kPa abs = psia × 6.894757293168."],
        [
            "s12 (phi)",
            "Flujo / presión",
            "(kg/s)/kPa = (lbm/s)/psi × 0.45359237 / 6.894757293168.",
        ],
        ["s20,s21", "Flujo de sangrado", "kg/s = lbm/s × 0.45359237."],
        ["s8,s9,s13,s14,s18", "Velocidad", "rpm conservadas."],
        [
            "s10,s15,s16; s17,s19",
            "Ratios / canales nativos",
            "Ratios sin dimensión. s17 y s19 conservan escala nativa por ambigüedad de unidad.",
        ],
    ],
    [102, 108, 280],
)
p(
    b,
    "Las presiones se mantienen absolutas. No se convierten a manométricas restando una atmósfera arbitraria. Una temperatura absoluta, presión o canal positivo no puede ser ≤0 en este esquema; esos valores o los no finitos se marcan faltantes. Son verificaciones mínimas de consistencia, no envolventes completas de operación certificadas.",
)
sub(b, "2.3. Nulos y atípicos: tratar causas sin borrar deterioro")
p(
    b,
    f"Los límites descriptivos Q1-3·IQR y Q3+3·IQR se calculan solo en desarrollo: se señalan {eda['statistical_outlier_cells_3iqr']} celdas, con <b>cero eliminaciones por ser extremos estadísticos</b>. Cerca del fallo, un extremo puede contener la señal objetivo. El paquete incluye la auditoría por canal y las columnas constantes detectadas. No se crean nulos artificiales para aparentar limpieza.",
)
p(
    b,
    "El imputador de mediana aprende únicamente con las filas del ajuste en cada fold; los faltantes del resto se transforman con esas mismas medianas. Los valores iniciales sin desviación estándar o pendiente suficiente se dejan como NaN hasta ese paso. El selector de varianza y StandardScaler también se ajustan dentro del fold. Las conversiones SI son fórmulas fijas y no aprenden estadísticas poblacionales.",
)

b = new_page("3. Diseño del experimento sin fuga")
sub(b, "3.1. Objetivos y características causales")
p(
    b,
    "Para un motor i de entrenamiento que termina en el ciclo T<sub>i</sub>, se define RUL<sub>i,t</sub>=T<sub>i</sub>-t. La etiqueta de ajuste es y<sub>i,t</sub>=min(RUL<sub>i,t</sub>,125). La meseta representa menor observabilidad del daño inicial y limita la exigencia de diferenciar vidas lejanas. El valor 125 es un hiperparámetro fijado antes de evaluar; no es una constante física de fallo. La prueba y la CV se comparan con la RUL verdadera <b>sin recortar</b>.",
)
p(
    b,
    f"Se construyen {len(meta['feature_columns'])} características: edad en ciclos; 3 ajustes; 21 sensores actuales; medias por sensor en ventanas de 5, 15 y 30 ciclos; desviación estándar y pendiente lineal en 15 ciclos. Todas las ventanas terminan en t, usan solo el mismo motor y admiten historial parcial. Se excluyen ID, vida total, ciclos futuros y etiquetas. La prueba de perturbación del futuro verifica esta propiedad.",
)
code(
    b,
    "X(t) usa {x(u): u <= t, mismo motor}\nfit_transform: solo fold de ajuste\ntransform: validación, calibración y prueba",
)
sub(b, "3.2. Partición independiente y protocolo fijado [4]")
table(
    b,
    ["Etapa", "Motores", "Uso permitido"],
    [
        [
            "Desarrollo",
            "80 de train_FD001",
            "Selección con GroupKFold: 5 folds, 64 motores de ajuste y 16 de validación en cada fold.",
        ],
        [
            "Calibración",
            "20 restantes de train_FD001",
            "Un punto pseudo-censurado por motor; solo estima el radio de incertidumbre.",
        ],
        [
            "Prueba externa",
            "100 de test_FD001",
            "Solo inferencia; el vector NASA se abre tras congelar el modelo.",
        ],
    ],
    [88, 115, 287],
)
p(
    b,
    "La reserva de calibración se sortea por ID con semilla 42. Se elige un punto pseudo-censurado uniforme por motor entre ciclos ≥30 previos al fallo, con semilla 43, sin consultar longitudes ni etiquetas de prueba. Esa muestra evita validar exclusivamente RUL=0 al final de las trayectorias completas. Los mismos puntos y folds se usan para todos los candidatos. Las identidades se distinguen como train:ID y test:ID porque NASA reutiliza los enteros entre flotas independientes.",
)
p(
    b,
    "GroupKFold se baraja con semilla 42 y mantiene todas las filas de cada motor en un único lado. Cada candidato crea un Pipeline nuevo: mediana → varianza &gt;10<super>-12</super> → StandardScaler → regresor. La ponderación total es igual por motor, evitando que trayectorias largas dominen la pérdida. El RMSE medio de los cinco folds define el ganador; la desviación entre folds describe su variabilidad.",
)
p(
    b,
    "Se reajusta el candidato elegido en los 80 motores de desarrollo y se calibra con los 20 reservados, sin incorporarlos después al ajuste. Se serializan predictor, transformadores y metadatos antes de consultar las etiquetas oficiales. La curva de aprendizaje utiliza un fold de desarrollo y nunca la calibración o la prueba. El CV sirve para selección; su media no se presenta como evaluación final insesgada después de seleccionar.",
)

b = new_page("4. EDA y contraste estadístico")
p(
    b,
    f"El EDA se limita a los 80 motores de desarrollo ({eda['development_rows']:,} ciclos). Las vidas observadas tienen mínimo {eda['lifetime_min']}, mediana {f(eda['lifetime_median'], 1)} y máximo {eda['lifetime_max']} ciclos. Esta heterogeneidad respalda agrupar por unidad y evita asumir una vida fija del motor. Las trayectorias suavizadas muestran tendencias con ruido y diferencias entre unidades.",
)
fig(
    b,
    "eda_lifetime_trajectories.png",
    "Figura 1. Duración hasta fallo y evolución de T30 en motores de desarrollo. El suavizado es causal; la historia completa se usa aquí únicamente como análisis retrospectivo.",
    490,
)
sub(b, "4.1. Hipótesis y unidad independiente")
p(
    b,
    "Para cada canal predefinido se calcula una diferencia por motor: media de los últimos 30 ciclos menos media de los primeros 30. Se aplica Wilcoxon bilateral de rangos con signo: H0, distribución de diferencias simétrica alrededor de cero; su interpretación como contraste de localización requiere simetría. Se usa Holm para controlar el error familiar de ocho pruebas (α=0.05). El efecto es la correlación biserial de rangos con signo; el IC95% de la media se obtiene remuestreando motores 2 000 veces.",
)
table(
    b,
    ["Canal", "Δ media (SI)", "IC95% media", "p Holm", "Efecto"],
    [
        [
            r.sensor + " / " + r.symbol,
            f(r.mean_late_minus_early, 3),
            f"[{f(r.mean_ci95_lower, 3)}; {f(r.mean_ci95_upper, 3)}]",
            f"{r.p_holm:.2e}",
            f(r.rank_biserial, 3),
        ]
        for r in hyp.itertuples()
    ],
    [94, 91, 135, 85, 85],
)
p(
    b,
    "Los cambios conjuntos de temperatura, presión y flujo son compatibles con la degradación del HPC bajo el controlador simulado. El tamaño de efecto y las unidades complementan el valor p: significancia no implica por sí sola utilidad predictiva. Las ventanas tempranas y tardías se definen con la trayectoria completa únicamente para este contraste retrospectivo; no entran como predictores ni como normalización del modelo.",
)
p(
    b,
    "La matriz Spearman completa se entrega en CSV y figura. Es descriptiva: sus filas son ciclos correlacionados y no se le atribuyen p-valores independientes. La colinealidad entre sensores y ventanas obliga a interpretar la importancia del modelo con cautela. Los canales constantes se eliminan mediante un selector ajustado dentro de cada fold, sin usar su comportamiento en prueba.",
)

b = new_page("5. Modelado y selección")
sub(b, "5.1. Fundamento y espacio de búsqueda")
p(
    b,
    "La mediana proporciona una referencia sin telemetría; Ridge regulariza una relación lineal multivariable; Random Forest captura interacciones mediante agregación de árboles; XGBoost aproxima relaciones no lineales por boosting regularizado. Los ocho candidatos se fijaron antes de la prueba: mediana; Ridge α={10,100}; bosques de 180 árboles con hoja mínima {3,8}; XGBoost con profundidad/árboles {(3,450),(4,450),(4,700)}. [5]",
)
p(
    b,
    "XGBoost usa tasa 0.035, min_child_weight=8, subsample=0.85, colsample_bytree=0.85, λ=10, α L1=0.2 y entrenamiento histogram. Los bosques usan max_features=0.75. Semilla 42 y dos hilos. No se aplica parada temprana con prueba ni se tunea la meseta después de ver el resultado oficial. Se recortan predicciones negativas a cero; no se impone un techo de 125 a las predicciones.",
)
table(
    b,
    ["Candidato", "RMSE CV [ciclos]", "DE folds", "MAE CV"],
    [[r.candidate, f(r.cv_rmse), f(r.cv_sd), f(r.cv_mae)] for r in cv.itertuples()],
    [220, 110, 75, 85],
)
fig(
    b,
    "model_selection.png",
    "Figura 2. Selección por RMSE en puntos de motores no vistos y curva de aprendizaje del ganador. Las barras son desviaciones entre folds, no intervalos de confianza.",
    490,
)
p(
    b,
    f"Se eligió <b>{winner}</b>, con RMSE CV medio {f(metrics['cv_mean_rmse'])} ± {f(metrics['cv_sd_rmse'])} ciclos. La curva utiliza 16, 32, 48 y 64 motores de un fold, manteniendo fija su validación. En el extremo de 64 motores se observa RMSE de ajuste {f(learning.iloc[-1].train_rmse)} y validación {f(learning.iloc[-1].validation_rmse)}. Su brecha debe leerse junto con la variabilidad del CV; una única curva no prueba convergencia poblacional.",
)
p(
    b,
    "La CV usa pseudo-censura uniforme y puede incluir horizontes mayores que NASA; sus RMSE no son directamente intercambiables y la meseta influye en los errores lejanos. Las importancias internas, entregadas en figura y CSV, reparten peso entre variables correlacionadas y no son efectos causales. La prueba externa congelada sustenta la evaluación final.",
)

b = new_page("6. Desempeño e incertidumbre")
sub(b, "6.1. Evaluación oficial por último ciclo [2,3]")
table(
    b,
    ["Métrica principal", "Resultado"],
    [
        ["RMSE [ciclos]", f(m["rmse"])],
        ["MAE [ciclos]", f(m["mae"])],
        ["R²", f(m["r2"], 3)],
        ["Sesgo, estimación - verdad [ciclos]", f(m["bias"])],
        ["NASA score; menor es mejor", f(m["nasa_score"])],
        [
            "IC95% bootstrap RMSE [ciclos]",
            f"[{f(m['rmse_ci95_engine_bootstrap'][0])}; {f(m['rmse_ci95_engine_bootstrap'][1])}]",
        ],
        [
            "Cobertura / ancho medio intervalo 90%",
            f"{f(100 * m['interval_coverage90'], 1)}% / {f(m['mean_interval_width'])} ciclos",
        ],
    ],
    [310, 180],
)
p(
    b,
    "RMSE = raíz de la media de errores cuadrados; MAE = media del error absoluto; R² compara con la media de la verdad. Para d=RUL estimada-RUL real, NASA score suma exp(-d/13)-1 si d&lt;0 y exp(d/10)-1 si d≥0. Así penaliza más una sobreestimación peligrosa. El IC del RMSE remuestrea 100 motores independientes, no ciclos, y no cubre la incertidumbre de reentrenar el modelo.",
)
fig(
    b,
    "test_residuals.png",
    "Figura 3. Predicción, error según RUL y distribución de residuos en la prueba oficial sin recortar. Los errores positivos representan sobreestimación de vida.",
    490,
)
sub(b, "6.2. Calibración reservada y límites de interpretación [6]")
p(
    b,
    f"Con 20 motores de calibración se ordenan |y-ŷ| y se toma el estadístico de orden ceil((20+1)·0.9)=19. El radio obtenido es <b>{f(rad)} ciclos</b>; el intervalo es [max(0,ŷ-q), ŷ+q]. La cobertura empírica oficial es {f(100 * m['interval_coverage90'], 1)}%, que se comunica junto con su ancho en lugar de asumir el 90% nominal.",
)
p(
    b,
    f"La amplitud de la banda limita su utilidad: con el escenario por defecto, {int((pred.status == 'Revisión prioritaria').sum())} motores requieren revisión prioritaria, {int((pred.status == 'Planificar inspección').sum())} planificación y {int((pred.status == 'Seguimiento').sum())} seguimiento. La cobertura alta no implica buen poder de discriminación. No se cambió el protocolo para estrechar la banda tras observar la prueba.",
)
p(
    b,
    "La validez conformal marginal requiere motores intercambiables y el mismo mecanismo de muestreo del punto observado. Nuestra pseudo-censura uniforme no garantiza coincidir con el protocolo NASA: el cambio de distribución y el tamaño de calibración limitan el traslado de la garantía. El intervalo no asegura 90% para un motor concreto ni para toda su trayectoria. Los ciclos menores a 30 quedan fuera del protocolo de calibración.",
)
secondary = metrics["secondary_capped_target"]
p(
    b,
    f"Como análisis secundario, al limitar también la verdad a 125 el RMSE es {f(secondary['rmse'])} ciclos. No se sustituye con esa cifra el RMSE oficial {f(m['rmse'])}. La meseta y la menor observabilidad inicial pueden sesgar los horizontes largos; se entregan errores por horizonte y motor.",
)

b = new_page("7. Priorización de inspección")
sub(b, "7.1. Regla operativa demostrativa, fijada antes de prueba")
p(
    b,
    "La estimación no se convierte automáticamente en una fecha de fallo. Se propone usar el límite inferior L del intervalo para priorizar una revisión por ingeniería. El horizonte H=30 ciclos, reserva B=10 y máximo de reevaluación M=20 son supuestos configurables del escenario, no parámetros derivados del fabricante.",
)
table(
    b,
    ["Condición", "Salida", "Acción de planificación"],
    [
        [
            "L ≤ B",
            "Revisión prioritaria",
            "Solicitar evaluación de ingeniería; no diferir por esta herramienta.",
        ],
        [
            "B < L ≤ H+B",
            "Planificar inspección",
            "Reservar ventana y recursos para revisión del HPC.",
        ],
        [
            "L > H+B",
            "Seguimiento",
            "Mantener monitoreo y reevaluar con nueva telemetría.",
        ],
    ],
    [102, 122, 266],
)
code(
    b,
    "Presupuesto de reevaluación = floor(min(M, max(0, L-B)))\nCon H=30, B=10 y M=20: decisión conservadora de escenario.",
)
p(
    b,
    "Este presupuesto expresa cuándo volver a analizar datos; no autoriza operación hasta ese instante ni reemplaza intervalos OEM, tareas del programa aprobado o decisiones de aeronavegabilidad. Una banda ancha puede generar muchas alertas y demanda de inspección. El tablero muestra esa carga de trabajo al cambiar horizonte y reserva.",
)
sub(b, "7.2. Sensibilidad de alertas, sin modificar el modelo")
table(
    b,
    ["H [ciclos]", "TP / FP / FN / TN", "Recall", "Precisión", "Costo ilustrativo"],
    [
        [
            str(int(r.horizon)),
            f"{int(r.tp)} / {int(r.fp)} / {int(r.fn)} / {int(r.tn)}",
            f(r.recall, 3),
            f(r.precision, 3),
            str(int(r.illustrative_cost_units)),
        ]
        for r in policy.itertuples()
    ],
    [80, 150, 75, 75, 110],
)
p(
    b,
    "Un caso se considera urgente cuando la RUL verdadera ≤H; una alerta se activa cuando L≤H+10. TP es urgente alertado, FP alerta sin urgencia, FN urgencia no alertada y TN ausencia correcta de alerta. La pérdida FP+20·FN asigna 20 veces más costo a una omisión, en unidades arbitrarias. Es sensibilidad retrospectiva de la regla prefijada, no una optimización del umbral sobre prueba ni una demostración de rentabilidad.",
)
sub(b, "7.3. Ejemplos concretos de la flota de prueba")
example = pd.concat(
    [pred.sort_values("lower90").head(2), pred.sort_values("lower90").tail(2)]
)
table(
    b,
    ["Motor", "Ciclo", "RUL estimada", "Límite inferior", "Reevaluar ≤"],
    [
        [
            str(int(r.engine_id)),
            str(int(r.last_cycle)),
            f(r.rul_pred, 1),
            f(r.lower90, 1),
            str(int(r.revisit_cycles)) + " ciclos",
        ]
        for r in example.itertuples()
    ],
    [75, 75, 115, 115, 110],
)
p(
    b,
    "La app ofrece filtros de estado, selección por motor y ciclo, sensores con unidades, banda de incertidumbre, comparación de modelos, trazabilidad y exportación CSV. Un historial cargado debe contener todos los ciclos desde el primero y el esquema nativo NASA; una entrada inválida se rechaza con un mensaje. Las métricas oficiales se mantienen identificadas como evaluación de NASA, sin atribuirlas a un CSV nuevo.",
)

b = new_page("8. Software, reproducción y trazabilidad")
sub(b, "8.1. Arquitectura modular y contrato de ejecución")
table(
    b,
    ["Módulo / artefacto", "Responsabilidad"],
    [
        [
            "src/rul/data.py",
            "Descarga, SHA-256, esquema, validaciones físicas y conversiones SI.",
        ],
        [
            "features.py / modeling.py",
            "Ventanas causales; candidatos; Pipeline de preprocesamiento local al fold.",
        ],
        [
            "pipeline.py / evaluation.py",
            "CV por motor, selección, calibración, serialización, métricas y decisión.",
        ],
        [
            "analysis.py / inference.py",
            "EDA y pruebas por unidad; predicción sin acceso a etiquetas.",
        ],
        [
            "app.py / notebooks/",
            "Streamlit interactivo y notebook ejecutado desde datos originales.",
        ],
        [
            "models/ / reports/",
            "Modelo y metadatos; métricas, manifiestos, tablas, figuras y reporte.",
        ],
    ],
    [160, 330],
)
p(
    b,
    f"Entorno verificado: CPython {metrics['python_version']}. requirements.txt fija las versiones exactas de todas las dependencias directas; el archivo lock registra las transitivas. Se ejecuta desde la raíz con rutas relativas. Los datos y el modelo entrenado incluidos permiten usar la app sin descargar ni entrenar. La reproducción desde cero vuelve a verificar SHA-256 y recrea todos los derivados.",
)
code(
    b,
    "python -m venv .venv\n# Activar .venv según el sistema\npython -m pip install -r requirements-lock.txt\npython scripts/run_pipeline.py\npython scripts/execute_notebook.py\npython scripts/build_report.py\npython -m pytest -q\nstreamlit run app.py",
)
p(
    b,
    "El pipeline y el notebook son dos rutas equivalentes de reproducción completa: no es necesario ejecutar ambas para obtener resultados. El ejecutor del notebook usa el intérprete activo; los scripts no contienen rutas del equipo de desarrollo. Se incluyen guía de inicio, diccionario y guion de defensa individual. El pickle se carga únicamente desde el artefacto de confianza generado por el proyecto, nunca desde una carga de usuario.",
)
sub(b, "8.2. Evidencia contra descalificación y cobertura de rúbrica")
table(
    b,
    ["Criterio", "Evidencia verificable"],
    [
        [
            "ETL · 20%",
            "data_audit.json, provenance.json, outlier_audit.csv, pruebas de esquema y unidades.",
        ],
        [
            "EDA · 25%",
            "Hipótesis por motor, Holm, IC bootstrap, Spearman descriptivo y trazas temporales.",
        ],
        [
            "Modelado · 25%",
            "cv_results.csv, split_manifest.json, modelo congelado, métricas oficiales y residuos.",
        ],
        [
            "Interactividad · 15%",
            "AppTest de carga y filtros; motor, ciclo, escenario, sensores, CSV y Plotly.",
        ],
        [
            "Contexto / reporte · 15%",
            "Decisión con límite inferior, sensibilidad, límites físicos y reporte de 10 páginas.",
        ],
    ],
    [140, 350],
)
p(
    b,
    "Las pruebas verifican invariancia de características al perturbar el futuro, no solapamiento de motores, estadísticas de escalado calculadas solo en desarrollo, congelación del modelo, persistencia de predicciones y correspondencia exacta con el vector NASA. El paquete incluye el registro de ejecución y verificación. Estas evidencias facilitan la evaluación; la calificación final depende del docente y de la defensa individual.",
)

b = new_page("9. Conclusiones y referencias")
sub(b, "9.1. Conclusiones sustentadas")
p(
    b,
    f"<b>1.</b> El flujo completo ejecutado conserva independencia por motor y preprocesamiento local al ajuste. El modelo {winner}, elegido entre ocho candidatos por CV, alcanzó RMSE {f(m['rmse'])} ciclos, MAE {f(m['mae'])} y NASA score {f(m['nasa_score'])} en la prueba externa de 100 motores. Estas cifras describen FD001; no certifican precisión en motores comerciales reales.",
)
p(
    b,
    f"<b>2.</b> El EDA por unidad identifica diferencias térmicas y de flujo asociadas al deterioro simulado. El contraste usa ocho hipótesis con Holm y efectos por motor, evitando inflar la significancia mediante ciclos correlacionados. Las correlaciones e importancias complementan el análisis sin constituir pruebas de causalidad.",
)
p(
    b,
    f"<b>3.</b> La incertidumbre se calibra sin reutilizar motores de ajuste: radio {f(rad)} ciclos, cobertura oficial {f(100 * m['interval_coverage90'], 1)}% y ancho medio {f(m['mean_interval_width'])} ciclos. La cobertura marginal, la censura y el tamaño de calibración deben condicionar cualquier interpretación de riesgo individual.",
)
p(
    b,
    "<b>4.</b> La política basada en límite inferior traduce RUL a prioridades revisables y permite observar falsas alertas y omisiones. El proyecto entrega una herramienta de análisis y planificación de escenarios, no un calendario de mantenimiento aprobado ni evidencia de reducción de AOG.",
)
sub(b, "9.2. Limitaciones y trabajo futuro")
p(
    b,
    "La muestra es pequeña a nivel de motores, sintética y de un único régimen y modo de fallo. La meseta de 125 reduce resolución a horizontes largos; los errores de motores tempranos pueden ser mayores. Las ventanas y algoritmos no prueban todas las arquitecturas posibles. La selección puede beneficiarse de validación agrupada anidada o repeticiones con presupuesto mayor; el resultado externo actual permanece intacto. Se propone evaluar otras particiones con nuevos datos, FD002/FD004, calibración con censura operacional equivalente, modelos de supervivencia y ensayos de campo que midan carga de inspecciones, AOG y costo con un diseño causal.",
)
sub(b, "9.3. Referencias primarias y trazabilidad documental")
references = [
    (
        "[1] NASA Ames PCoE.",
        "Saxena, A. y Goebel, K. (2008). Turbofan Engine Degradation Simulation Data Set. Fuente y enlace de descarga:",
        "https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/",
    ),
    (
        "[2] NASA Open Data.",
        "CMAPSS Jet Engine Simulated Data. Descripción de las flotas, FD001 y etiquetas:",
        "https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data",
    ),
    (
        "[3] Saxena, A.; Goebel, K.; Simon, D.; Eklund, N. (2008).",
        "Damage Propagation Modeling for Aircraft Engine Prognostics. NTRS 20090029214; tabla 2 (canales) y ec. 9 (score):",
        "https://ntrs.nasa.gov/citations/20090029214",
    ),
    (
        "[4] scikit-learn.",
        "Documentación oficial: validación agrupada y prevención de data leakage. Versión de ejecución 1.8.0.",
        "https://scikit-learn.org/1.8/modules/cross_validation.html",
    ),
    (
        "[5] XGBoost.",
        "Documentación oficial de la API Python; versión del artefacto 3.4.1.",
        "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
    ),
    (
        "[6] Angelopoulos, A. N. y Bates, S. (2021; revisión 2022).",
        "A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification. arXiv:2107.07511.",
        "https://arxiv.org/abs/2107.07511",
    ),
    (
        "[7] Streamlit.",
        "Documentación oficial de AppTest para verificación del tablero.",
        "https://docs.streamlit.io/develop/api-reference/app-testing/st.testing.v1.apptest",
    ),
]
for author, title, url in references:
    b.append(
        (
            "reference",
            f'<b>{escape(author)}</b> {escape(title)}<br/><link href="{url}" color="#246ba0">{escape(url)}</link>',
        )
    )
b.append(
    (
        "small",
        "Documentos de la asignatura: Rúbrica de Proyecto: Análisis de Datos Aeronáuticos (8 páginas) y Reporte_Tecnico_Aerodinamica.pdf (portada modelo facilitada). Se conserva la portada con docente, integrantes y carnés; solo se adapta el tema y la fecha. Fuentes consultadas el 4 de noviembre de 2026. Código del proyecto redactado para esta entrega; no se reutilizaron notebooks públicos.",
    )
)

story = []
markdown = [
    "# Predicción de Vida Útil Remanente en Motores Turbofán",
    "",
    "Universidad Don Bosco · DAS172 · Docente: Eduardo Galicia · 4 de noviembre de 2026",
    "",
    "Integrantes: Salazar Guzmán, Adriana Zaira (SG231489); Martínez Linares, Marcela Lorena (ML220036); Mejía Martínez, Andrea Alejandra (MM210136); Hernández Martínez, Diego Josué (HM240091); Abarca Gómez, Mauro José (AG241380); Portillo Linares, Jaime Fernando (PL231163).",
    "",
]
for number, page in enumerate(PAGE):
    if number:
        story.append(PageBreak())
    story.append(Paragraph(page["title"], styles["TitleRUL"]))
    markdown += ["## " + page["title"], ""]
    for kind, value in page["blocks"]:
        if kind in ("p", "sub", "small", "reference", "code"):
            sty = {
                "p": "BodyRUL",
                "sub": "SubRUL",
                "small": "SmallRUL",
                "reference": "SmallRUL",
                "code": "CodeRUL",
            }[kind]
            if kind == "code":
                text = escape(value).replace("\n", "<br/>")
                markdown += [
                    "```bash" if value.startswith("python") else "```text",
                    value,
                    "```",
                    "",
                ]
            else:
                text = value
                plain = unescape(re.sub("<[^>]+>", "", text.replace("<br/>", " ")))
                markdown += [("### " if kind == "sub" else "") + plain, ""]
            story.append(Paragraph(text, styles[sty]))
        elif kind == "table":
            headers, rows, widths = value
            cells = [
                [
                    Paragraph("<b>" + escape(str(x)) + "</b>", styles["CellRUL"])
                    for x in headers
                ]
            ]
            cells += [
                [Paragraph(escape(str(x)), styles["CellRUL"]) for x in row]
                for row in rows
            ]
            t = Table(
                cells,
                colWidths=widths or [490 / len(headers)] * len(headers),
                hAlign="LEFT",
                repeatRows=1,
            )
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), LIGHT),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 7),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                        ("TOPPADDING", (0, 0), (-1, -1), 6),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.7, BLUE),
                        (
                            "LINEBELOW",
                            (0, 1),
                            (-1, -1),
                            0.25,
                            colors.HexColor("#d9e3eb"),
                        ),
                    ]
                )
            )
            story += [t, Spacer(1, 9)]
            markdown += [
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
            markdown += [
                "| " + " | ".join(str(x).replace("|", "/") for x in row) + " |"
                for row in rows
            ]
            markdown += [""]
        elif kind == "fig":
            filename, caption, width = value
            image = Image(str(R / "figures" / filename))
            ratio = image.imageHeight / image.imageWidth
            image.drawWidth = width
            image.drawHeight = width * ratio
            story += [image, Spacer(1, 4), Paragraph(caption, styles["CaptionRUL"])]
            markdown += [f"![{caption}](figures/{filename})", ""]


def footer(canvas, doc):
    w, h = A4
    canvas.setStrokeColor(colors.HexColor("#cbd8e3"))
    canvas.setLineWidth(0.5)
    canvas.line(50, h - 39, w - 50, h - 39)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(NAVY)
    canvas.drawString(50, h - 29, "Universidad Don Bosco | Proyecto Integrador DAS172")
    canvas.drawRightString(w - 50, h - 29, "NASA C-MAPSS FD001 | RUL")
    canvas.line(50, 39, w - 50, 39)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(50, 26, "Analítica de mantenimiento aeronáutico")
    canvas.drawRightString(w - 50, 26, str(doc.page + 1))


body = R / "_body.pdf"
doc = SimpleDocTemplate(
    str(body),
    pagesize=A4,
    rightMargin=52,
    leftMargin=52,
    topMargin=54,
    bottomMargin=51,
    title="Predicción de RUL en Motores Turbofán",
    author="Equipo DAS172 - Universidad Don Bosco",
)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
body_doc = fitz.open(body)
if len(body_doc) != 9:
    raise RuntimeError(
        f"El cuerpo debe tener 9 páginas, tiene {len(body_doc)}; revisar desbordes."
    )
final = fitz.open("assets/portada_proyecto.pdf")
final.insert_pdf(body_doc)
final.set_metadata(
    {
        "title": "Predicción de Vida Útil Remanente en Motores Turbofán - NASA C-MAPSS FD001",
        "author": "Equipo DAS172 - Universidad Don Bosco",
        "subject": "Proyecto de Analítica Aeronáutica",
        "keywords": "RUL, NASA, C-MAPSS, FD001, GroupKFold, HPC",
    }
)
final.save(R / "Reporte_Tecnico_RUL_Turbofan.pdf", garbage=4, deflate=True)
body_doc.close()
body.unlink()
(R / "Reporte_Tecnico_RUL_Turbofan.md").write_text(
    "\n".join(markdown), encoding="utf-8"
)
(R / "REFERENCIAS.md").write_text(
    "# Referencias y atribución\n\n"
    + "\n\n".join(f"{a} {t}\n\n{u}" for a, t, u in references)
    + "\n\nLa portada se adapta exclusivamente del PDF facilitado por el equipo. Los datos conservan atribución a NASA Ames PCoE. El código y la redacción del pipeline fueron creados para esta entrega.\n",
    encoding="utf-8",
)
print("Reporte PDF verificado: 10 páginas; Markdown y referencias generados.")
