"""Generate the original explanatory notebook; execute_notebook.py runs every cell."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
nb.cells = [
    md("""# AEROLIFE · Pronóstico de vida remanente del turbofán
**Universidad Don Bosco · DAS172 · Proyecto Integrador**  
NASA C-MAPSS FD001 · Semilla 42 · Código original modular.

Este notebook ejecuta el experimento completo: ingesta → validación física → características causales → CV por motor → selección → calibración independiente → congelación → evaluación NASA.

**Integrantes:** Adriana Zaira Salazar Guzmán (SG231489), Marcela Lorena Martínez Linares (ML220036), Andrea Alejandra Mejía Martínez (MM210136), Diego Josué Hernández Martínez (HM240091), Mauro José Abarca Gómez (AG241380), Jaime Fernando Portillo Linares (PL231163)."""),
    md("""## 1. Entorno y rutas relativas
Ejecutar con el entorno de `requirements.txt`. Se admite abrir desde la raíz o `notebooks/`. Los datos NASA se incluyen con SHA-256; si faltan, el pipeline descarga el archivo enlazado por NASA."""),
    code("""from pathlib import Path
import os, sys, json
if not Path('config.json').exists() and Path('../config.json').exists():
    os.chdir('..')
assert Path('config.json').exists(), 'Abra el notebook dentro del repositorio.'
sys.path.insert(0, str(Path('src')))
import pandas as pd
import numpy as np
from IPython.display import display, Image
from rul.config import Config
from rul.data import download_data, read_raw, validate_and_clean
cfg = Config.load()
print('Configuración predefinida:', cfg.to_dict())
print(Path('requirements.txt').read_text())"""),
    md("""## 2. Fuente y consistencia
ID, ciclo, tres ajustes y 21 sensores. Temperaturas absolutas: Rankine a kelvin; presiones: psia a kPa absoluto. Los canales ambiguos conservan unidades nativas. Los extremos estadísticos se señalan sin borrar degradación; los imposibles físicos pasan a faltantes y se imputan con mediana de entrenamiento, dentro de cada fold."""),
    code("""provenance = download_data()
print(json.dumps(provenance, indent=2))
train_raw, audit = validate_and_clean(read_raw('data/raw/train_FD001.txt'))
display(pd.DataFrame([audit]))
display(train_raw.head())"""),
    md("""## 3. Experimento completo sin fuga
20 motores reservados para calibrar; 80 para selección con GroupKFold de cinco folds. Imputador, selector de varianza y escalador se ajustan en el fold de entrenamiento exclusivamente. Ocho candidatos de cuatro familias se comparan con RMSE sobre **un punto pseudo-censurado por motor**, uniforme entre ciclos >=30 anteriores al fallo (semilla 43). Los objetivos de ajuste se limitan a 125; CV y prueba usan verdad completa. Peso total igual por motor.

La celda siguiente reentrena todo y reemplaza los resultados derivados. Puede tardar varios minutos. La repetición reproduce el protocolo fijado; no permite ajustar parámetros a la prueba oficial."""),
    code("""from rul.pipeline import run_pipeline
metrics = run_pipeline(cfg)
display(pd.read_csv('reports/model_comparison.csv'))
print(json.dumps(metrics['official_test'], indent=2))"""),
    md("""## 4. EDA e inferencia por motor
H0 bilateral: mediana de la diferencia entre media tardía (últimos 30 ciclos) y temprana (primeros 30) = 0. Wilcoxon pareado bajo simetría de diferencias, Holm para ocho contrastes y bootstrap de 2 000 motores. No se trata cada ciclo como observación independiente. El heatmap es descriptivo; una asociación no identifica una avería real ni demuestra causalidad."""),
    code("""display(pd.read_csv('reports/hypothesis_tests.csv'))
display(Image(filename='reports/figures/eda_lifetime_trajectories.png'))
display(Image(filename='reports/figures/eda_correlation.png'))"""),
    md("""## 5. Selección, residuos e incertidumbre
El predictor final se ajusta solo en 80 motores. Los 20 errores de calibración absolutos determinan el radio conformal de orden `ceil(21*0.9)=19`. Se serializa el modelo antes de abrir las etiquetas oficiales, sin reajustarlo con calibración.

La cobertura nominal 90% exige intercambio entre unidades calibradas y futuras; los protocolos de censura pueden diferir. Se reporta cobertura observada sin prometer cobertura condicional por motor ni simultánea por trayectoria. El RMSE oficial no recorta la verdad NASA; el RMSE con verdad limitada se reporta solo como secundario."""),
    code("""display(Image(filename='reports/figures/model_selection.png'))
display(Image(filename='reports/figures/test_residuals.png'))
display(Image(filename='reports/figures/test_intervals.png'))
display(pd.read_csv('reports/metrics_by_horizon.csv'))
display(pd.DataFrame([metrics['secondary_capped_target']]).assign(nota='Secundaria: verdad limitada a 125'))"""),
    md("""## 6. Decisiones de ingeniería
L = límite inferior. Revisión prioritaria si L<=10; planificación si 10<L<=40; seguimiento si L>40. Reevaluar como máximo en `floor(min(20,max(0,L-10)))` ciclos. Son supuestos configurables de demostración, no intervalos del fabricante. La sensibilidad usa costo ilustrativo `FP+20*FN`, sin dólares ni efecto causal demostrado sobre AOG."""),
    code("""display(pd.read_csv('reports/policy_sensitivity.csv'))
pred = pd.read_csv('reports/test_predictions.csv')
display(pred.sort_values('lower90').head(12))
display(Image(filename='reports/figures/feature_importance.png'))"""),
    md("""## 7. Separación e inferencia serializada
`train:1` y `test:1` son trayectorias diferentes aunque NASA reutilice el ID entero. La inferencia no consume etiquetas y solo transforma el historial pasado."""),
    code("""split = json.loads(Path('reports/split_manifest.json').read_text())
assert not set(split['development_ids']) & set(split['calibration_ids'])
for fold in split['folds']:
    assert not set(fold['fit_engine_ids']) & set(fold['validation_engine_ids'])
assert split['model_hash_unchanged_after_test']
from rul.inference import load_bundle, predict_history
example = read_raw('data/raw/test_FD001.txt').query('engine_id == 1')
replayed, _, _ = predict_history(example, load_bundle())
np.testing.assert_allclose(replayed.rul_pred.iloc[-1], pred.query('engine_id == 1').rul_pred.iloc[0], rtol=1e-6)
print('Separación, congelación e inferencia verificadas.')"""),
    md("""## 8. Reporte, aplicación y referencias
`python scripts/build_report.py`: PDF formal y Markdown. `streamlit run app.py`: tablero. `python -m pytest -q`: pruebas de fuga, causalidad, integridad y controles UI.

Fuentes primarias: NASA Ames PCoE, C-MAPSS; Saxena et al. (2008), NTRS 20090029214; Angelopoulos y Bates (2021/2022), arXiv:2107.07511; documentación oficial scikit-learn, XGBoost y Streamlit. URLs completas en `reports/REFERENCIAS.md`."""),
]
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3 (FD001)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12"},
}
nbf.write(nb, "notebooks/pipeline_analitico.ipynb")
print("Notebook creado:", len(nb.cells), "celdas")
