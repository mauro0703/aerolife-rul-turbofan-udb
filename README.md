# AEROLIFE — RUL de turbofanes NASA C-MAPSS FD001

Proyecto Integrador · Desarrollo de Algoritmos para la Simulación de Sistemas · Universidad Don Bosco.

## Archivos para revisión y exposición

| Entregable | Abrir |
| --- | --- |
| Informe técnico | [PDF](reports/Reporte_Tecnico_RUL_Turbofan.pdf) · [Word editable](reports/Reporte_Tecnico_RUL_Turbofan.docx) |
| Dashboard interactivo | [Descargar HTML](reports/Dashboard_RUL_Turbofan.html) y abrirlo en un navegador |
| Notebook ejecutado | [Pipeline de análisis](notebooks/pipeline_analitico.ipynb) |
| Explicación para el equipo | [Guía de exposición en Word](reports/Explicacion_Proyecto_RUL_Para_Exposicion.docx) |

## Inicio rápido

El paquete contiene los datos originales verificados, el modelo entrenado, las transformaciones, el notebook ejecutado, el reporte PDF y el tablero Streamlit. No necesita descargar NASA ni volver a entrenar para abrir la app.

Entorno probado: **CPython 3.12.14**, CPU. Usar Python **3.12** para reproducir las versiones fijadas. Python 3.10+ era el mínimo de la consigna; esta entrega usa 3.12. El entrenamiento puede durar varios minutos según el equipo; no requiere GPU.

1. Abrir una terminal en la **raíz del proyecto**, donde están `app.py` y `README.md`. Si descargaste un ZIP, extráelo primero.
2. Crear y activar un entorno:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

3. Instalar las versiones exactas y abrir el tablero:

```bash
python -m pip install -r requirements-lock.txt
streamlit run app.py
```

La dirección local aparece en la terminal, normalmente `http://localhost:8501`. El tablero incluye cuatro áreas: flota y decisiones; historial de motor y sensores; validación; método y trazabilidad. El historial propio se carga como CSV en las **unidades nativas NASA**, con exactamente las 26 columnas del diccionario. El CSV de ejemplo se descarga desde la app.

El archivo `reports/Dashboard_RUL_Turbofan.html` es una exploración interactiva **sin instalación ni conexión**, basada en los resultados ya calculados. Se abre en un navegador e incluye motor, ciclo, telemetría, incertidumbre y reglas de planificación. La app Streamlit también admite inferencia sobre CSV nuevo.

## Reproducir desde los datos originales

Los tres TXT conservan los bytes enlazados por NASA y sus SHA-256. Al faltar datos, la descarga se realiza desde el enlace publicado en la página oficial NASA PCoE. La fuente incluye un ZIP anidado; la ingesta extrae exclusivamente los nombres esperados.

```bash
python scripts/download_data.py
python scripts/run_pipeline.py
python scripts/build_report.py
python scripts/build_interactive_export.py
python -m pytest -q
```

Para volver a descargar: `python scripts/download_data.py --force`. El archivo de procedencia se regenera con los hashes descargados; compárelos con la copia inicial si se sospecha un cambio de fuente.

El notebook es una segunda ruta integral, no requiere ejecutar primero `run_pipeline.py`:

```bash
python scripts/execute_notebook.py
```

En entornos que no permitan sockets locales, `python scripts/execute_notebook.py --in-process` ejecuta todas las celdas secuencialmente mediante IPython, incluyendo entrenamiento y salidas, sin iniciar un servidor Jupyter. Esa es la ruta utilizada para verificar este cuaderno en el entorno de entrega.

También puede abrirse con un editor compatible con `.ipynb`, elegir el entorno activo y ejecutar todas las celdas. El ejecutor incluido evita depender de JupyterLab o de una extensión concreta. `scripts/build_notebook.py` regenera el cuaderno explicativo sin salidas; no debe ejecutarse después de `execute_notebook.py` si desea conservar las salidas calculadas.

## Diseño del experimento

- NASA train: **80 motores de desarrollo + 20 motores de calibración**; semilla 42.
- Selección: **GroupKFold de 5 folds** sobre desarrollo; motor íntegro en un lado. Ocho candidatos predefinidos: mediana, dos Ridge, dos Random Forest y tres XGBoost.
- Validación: un punto pseudo-censurado por motor, uniforme entre ciclos >=30 antes del fallo, semilla 43. Ponderación igual por motor en el ajuste.
- Características: edad, ajustes, sensores y ventanas causales de 5, 15 y 30 ciclos. No se usa ID, duración final ni información futura.
- Preprocesamiento: mediana, selector de varianza y StandardScaler dentro de cada Pipeline; `.fit_transform()` solo en el ajuste, `.transform()` en el resto.
- Objetivo de ajuste: `min(RUL,125)`; **CV y RMSE principal oficial comparan contra RUL completa**. La métrica con verdad limitada se identifica como secundaria.
- Calibración split-conformal: un punto por cada uno de los 20 motores reservados. El modelo final nunca se reajusta con calibración.
- Prueba oficial: 100 trayectorias independientes; se predice el último ciclo y se abre la verdad después de serializar el modelo. Los ID se distinguen por origen: `train:1` y `test:1` son motores diferentes.

## Estructura

| Ruta | Contenido |
| --- | --- |
| `data/raw/` | TXT originales FD001, readme NASA y procedencia con hashes |
| `data/processed/` | Tablas limpias SI y matrices escaladas; CSV comprimidos |
| `src/rul/` | Ingesta, características, modelado, evaluación, EDA e inferencia |
| `notebooks/pipeline_analitico.ipynb` | Cuaderno integral ejecutable con resultados |
| `models/rul_predictor_model.pkl` | Pipeline completo + metadatos + radio conformal |
| `models/MODEL_CARD.md` | Uso previsto y límites |
| `reports/Reporte_Tecnico_RUL_Turbofan.pdf` | Reporte formal de 10 páginas con portada de referencia adaptada |
| `reports/Reporte_Tecnico_RUL_Turbofan.md` | Versión editable del reporte |
| `reports/Dashboard_RUL_Turbofan.html` | Exportación interactiva autónoma de resultados |
| `reports/*.csv`, `*.json`, `figures/` | Métricas, auditoría, hipótesis, residuos, decisiones y gráficos |
| `tests/` | Controles de causalidad, fuga, integridad, serialización y Streamlit |
| `app.py` | Aplicación Streamlit completa |
| `requirements.txt`, `requirements-lock.txt` | Dependencias directas y transitivas fijadas |

## Evidencia de ejecución

`reports/metrics.json` contiene las cifras oficiales; `reports/cv_results.csv` los 40 ajustes de selección; `reports/split_manifest.json` las identidades de cada fold y el hash del modelo congelado. `reports/verification.json` resume la verificación final. El notebook entregado conserva sus salidas. El PDF y el Markdown se generan desde esas tablas para evitar transcribir resultados manualmente.

Las conversiones SI son fijas y reversibles. Los extremos estadísticos de degradación se señalan sin eliminarlos. Los valores físicamente imposibles pasan a nulos y se imputan con estadísticas del ajuste. No se inventan nulos en datos completos. Los canales s17 y s19 conservan escala nativa debido a ambigüedad dimensional en la documentación original.

## Límites de uso

FD001 simula **degradación HPC**, una sola condición operacional. No mide AOG, horas reales de vuelo, inspecciones ni costo. La política `L<=B`, `L<=H+B` y reevaluación `floor(min(M,max(0,L-B)))` es un escenario de planificación configurable, no una instrucción OEM. La cobertura conformal nominal requiere intercambio entre motores y mecanismo de censura equivalente; se comunica la cobertura empírica observada, sin garantizar 90% en cada motor ni simultáneamente en su trayectoria. Las predicciones antes de 30 ciclos se identifican fuera del protocolo de calibración.

La calificación depende del docente y de la defensa individual. Se incluye `reports/GUIA_DEFENSA.md` para explicar el razonamiento, las ecuaciones y las limitaciones con transparencia.

## Autoría y fuentes

El equipo del proyecto está integrado por

- Adriana Zaira Salazar Guzmán
- Marcela Lorena Martínez Linares
- Andrea Alejandra Mejía Martínez
- Diego Josué Hernández Martínez
- Mauro José Abarca Gómez
- Jaime Fernando Portillo Linares

El docente de la asignatura es Eduardo Galicia. La portada del informe incluye los carnés de los integrantes.

Código original preparado para este proyecto, sin copiar cuadernos públicos. Se reconoce NASA Ames PCoE y Saxena/Goebel como fuente de los datos; la portada adapta únicamente la referencia suministrada. Referencias completas en `reports/REFERENCIAS.md`.
