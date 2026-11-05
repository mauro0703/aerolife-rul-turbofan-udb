# Model card — FD001 RUL

**Uso previsto:** demostración de pronóstico de degradación del HPC y priorización de revisión, en ciclos simulados NASA FD001. **Entrenamiento:** 80 motores independientes de `train_FD001`; objetivos RUL con meseta de 125 ciclos, ponderación igual por motor. **Selección:** ocho candidatos predefinidos, cinco folds agrupados por motor, un snapshot de validación por motor. **Calibración:** 20 motores reservados, un snapshot pseudo-censurado por unidad; intervalo split-conformal nominal 90%. No se reajusta el predictor con calibración.

**Evaluación:** 100 motores oficiales; se conserva el vector verdadero sin recortarlo. Métricas, error por horizonte, cobertura e IC bootstrap por motor en `reports/`. El único resultado oficial usado en conclusiones es el que corresponde al modelo ya congelado.

**Entrada:** historial íntegro desde ciclo 1, motor-ciclo único, 3 ajustes y 21 canales en unidades nativas NASA. `predict_history` valida, convierte y crea ventanas causales. No admite paquetes pickle de terceros ni reentrena en inferencia.

**Límites:** datos sintéticos, una condición de operación y un modo de fallo, flota pequeña, distribución de pseudo-censura distinta de la prueba NASA, hipótesis de intercambio para conformal no asegurada entre protocolos, poca información de desgaste inicial y tope del objetivo. Cobertura marginal no equivale a 90% para cada motor ni a cobertura simultánea de una trayectoria. No se han medido AOG, costo monetario ni efecto causal de mantenimiento. No justifica modificar instrucciones OEM, intervalos aprobados, AD ni MEL.

**Mantenimiento del modelo:** controlar versiones, integridad y deriva; revisar calibración con motores nuevos independientes; probar FD002/FD004 y datos reales antes de proponer cualquier validación operacional.
