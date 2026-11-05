# Datos originales NASA C-MAPSS FD001

Los tres TXT incluidos contienen bytes del enlace de descarga publicado por NASA PCoE, sin edición. `provenance.json` registra fuente y SHA-256; `python scripts/download_data.py` verifica las copias y `--force` vuelve a descargarlas. Se incluye `readme.txt` original. Solo FD001: 100 trayectorias de entrenamiento hasta fallo y 100 trayectorias de prueba truncadas; el vector RUL corresponde al último ciclo de cada motor de prueba, ordenado por ID.

Cita: A. Saxena y K. Goebel (2008), *Turbofan Engine Degradation Simulation Data Set*, NASA Ames Prognostics Data Repository.

Los datos conservan sus condiciones de origen. La licencia del código del proyecto no cambia las condiciones de los datos NASA.
