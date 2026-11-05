# Datos derivados

`train_clean_si.csv.gz` y `test_clean_si.csv.gz`: conversiones físicas deterministas, sin modificar los originales. La tabla de entrenamiento contiene etiquetas; nunca usarlas como predictores. `development_scaled.csv.gz`: transformación ajustada en los 80 motores de desarrollo. `calibration_scaled.csv.gz`: un snapshot por motor reservado, solo transformado. `test_scaled.csv.gz`: todo el historial de prueba, solo transformado.

La primera columna identifica el espacio de nombres (`train:ID` o `test:ID`), porque NASA reutiliza los números enteros entre flotas independientes. Las dos primeras columnas son metadatos y no forman parte de X. Las estadísticas aprendidas están dentro del pipeline serializado. Durante CV se crean transformadores nuevos para cada fold; estos archivos finales no se usan como entrada de la CV.
