# Revisión técnica de AEROLIFE

Fecha de revisión: 7 de noviembre de 2026.

Se contrastaron la rúbrica de evaluación, los datos originales, el código modular, el cuaderno, el modelo serializado, las tablas derivadas, las dos interfaces y el informe editable. El protocolo de selección y los hiperparámetros del experimento original se conservaron. La prueba oficial conocida se reutilizó para verificar la reproducción; no se presenta como una nueva evaluación ciega.

## Correspondencia con los criterios de evaluación

| Criterio | Peso | Evidencia disponible |
| --- | ---: | --- |
| ETL y calidad de datos | 20% | Procedencia y SHA-256; esquema de 26 columnas; validación de identidad, continuidad y plausibilidad; conversión de unidades; auditoría de nulos y extremos |
| Análisis exploratorio | 25% | Distribuciones, trayectorias, Spearman descriptivo y ocho contrastes Wilcoxon por motor, con Holm, tamaño del efecto e IC bootstrap |
| Modelado y validación | 25% | Ocho candidatos, cinco folds por motor, predicciones CV identificadas, referencia mediana, curva de aprendizaje, calibración reservada y prueba oficial sin recortar |
| Interactividad | 15% | Controles H/B/M, filtros, historial causal y sensores; exportaciones; pruebas AppTest y navegación web; coherencia entre el modelo y el panel |
| Contexto e informe | 15% | Degradación del HPC, reglas y límites de planificación, métricas interpretadas, fuentes primarias, siete autores e informe de nueve páginas |

La evidencia permite revisar el trabajo por criterio. No sustituye la evaluación del docente ni la explicación individual del método.

## Hallazgos corregidos

| Hallazgo | Corrección y comprobación |
| --- | --- |
| La aplicación podía conservar un modelo anterior en caché | Las claves incluyen la firma de los archivos; modelo, métricas e historiales se renuevan juntos |
| Una ejecución interrumpida podía dejar derivados de ejecuciones distintas | Estado analítico explícito, manifiesto SHA-256 y rechazo de resultados incompletos o alterados |
| Dos entrenamientos podían escribir en los mismos archivos | Bloqueo del sistema operativo con liberación al terminar o interrumpir el proceso |
| La carga del modelo no comprobaba compatibilidad | Verificación de hashes, metadatos y versiones de NumPy y scikit-learn antes de deserializar |
| Faltaba una respuesta clara para historiales vacíos o identidades fuera de rango | Validación temprana y mensajes controlados; casos de regresión añadidos |
| Un ZIP de datos incompleto podía sobrescribir parte de una descarga previa | Se comprueban todos los miembros requeridos antes de modificar los TXT |
| Los parámetros podían generar intervalos o políticas inválidos | Contrato de configuración y validación de entradas de calibración y planificación; el esquema publicado mantiene cobertura nominal del 90% |
| Faltaba trazabilidad de cada pronóstico de validación | Se conserva `cv_predictions.csv`, con candidato, fold, motor, ciclo, verdad y predicción |
| El ejecutor del notebook podía elegir un kernel global ajeno al entorno activo | Kernel temporal ligado al intérprete activo y alternativa completa IPython para contenedores sin sockets |
| El exportador no contrastaba sus puntos finales con el CSV oficial | Comprobación de identidad, ciclo, pronóstico e intervalo antes de generar el panel |
| El navegador podía mezclar HTML nuevo con JS y CSS antiguos | URLs de recursos y descarga versionadas por contenido; comprobación en la página publicada |
| La precisión mostrada en el historial difería de la tabla | Los límites del intervalo se muestran con un decimal en ambos lugares |
| Las barras de flota dependían únicamente del ratón | Selección por Enter o espacio, etiquetas y agrupación accesible del gráfico |
| El informe contenía comandos y referencias a documentos retirados | Instrucciones verificadas, eliminación de referencias obsoletas y separación clara entre panel web y Streamlit |
| El índice omitía secciones principales y había referencias de paginación inconsistentes | Jerarquía de títulos y páginas revisadas en el PDF final; portada con los siete integrantes |
| La ponderación por motor y la media de RMSE podían interpretarse de forma imprecisa | Se especifica que los pesos corresponden a la pérdida del regresor, y que la selección usa la media de los RMSE por fold |

## Resultados reproducidos

- Instalación completa de las versiones fijadas en un entorno Python 3.12 independiente. Descarga nueva desde la fuente registrada y coincidencia SHA-256 de los tres TXT con los originales incluidos.
- Pipeline ejecutado desde los TXT originales y notebook ejecutado de principio a fin: siete celdas de código sin errores.
- Coincidencia de métricas y tablas numéricas entre ambas rutas con tolerancia absoluta de 10⁻¹⁰. Coincidencia exacta de los bytes del modelo y del contenido descomprimido de las cinco tablas procesadas.
- 41 pruebas superadas. Incluyen separación por motor, ausencia de ajuste en inferencia, causalidad de ventanas, unidades, integridad, interrupciones, exclusión de escritores simultáneos, controles de interfaz y distribución portátil.
- RMSE oficial 17.18649 ciclos, MAE 12.32762, R² 0.828953 y NASA score 491.72733 en 100 motores.
- ZIP descargado desde el botón real, con cuatro archivos idénticos a los publicados. CSV descargado con H=10 y filtro Seguimiento: motores 22, 69, 2, 85 y 1; pronósticos y reevaluación contrastados con Python.
- Inspección visual de las nueve páginas del informe, enlaces y coherencia entre Word y PDF. Los registros y hashes constan en `verification.json`.

## Limitaciones que deben conservarse en la interpretación

El resultado no demuestra una política de mantenimiento operacional. El bosque promedia objetivos limitados a 125 ciclos y el radio conformal es 101.73739; por tanto, L no supera 23.26261 ciclos. Con H=30 y B=10, el umbral de alerta es 40 y se alerta necesariamente a toda la flota: 25 TP, 75 FP, 0 FN y 0 TN. La cobertura del 100% y la anchura media de 172.76 ciclos deben presentarse juntas. Modificar los parámetros después de consultar la prueba para aparentar mejor desempeño invalidaría esa interpretación.

La calibración interna usa pseudocensura uniforme; su mecanismo puede diferir del oficial. La garantía conformal marginal requiere intercambio entre unidades y no es una garantía individual ni simultánea por trayectoria. Los ciclos menores a 30 son exploratorios. FD001 es sintético, incluye un régimen y degradación del HPC; no aporta evidencia de ahorro ni de reducción de AOG en servicio.

El navegador utilizado para la comprobación admite únicamente HTTP/HTTPS y bloquea `file://`. Se verificaron la descarga, el contenido del ZIP y sus referencias relativas, pero no se declara ejecutada una prueba de navegador del archivo local extraído. El kernel Jupyter tampoco pudo iniciar mediante TCP en este entorno; la ejecución completa del cuaderno se comprobó con la alternativa documentada `--in-process`.
