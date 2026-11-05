# Guía de defensa individual

## Explicación de un minuto

El proyecto estima cuántos ciclos le quedan a un turbofán simulado antes del límite de fallo. NASA FD001 contiene trayectorias completas de entrenamiento y otras truncadas para prueba. Extraemos señales y tendencias que podrían conocerse en el instante de predicción. Separamos motores completos, ajustamos transformadores solo en el entrenamiento de cada fold, comparamos ocho candidatos y calibramos la incertidumbre con otros 20 motores. Congelamos el modelo antes de abrir la verdad oficial. La app convierte el límite inferior estimado en prioridades de revisión. No afirmamos ahorros de AOG porque el dataset no contiene esos resultados.

## Preguntas que debe poder explicar cada integrante

| Pregunta | Respuesta que debe comprender |
| --- | --- |
| ¿Qué se degrada en FD001? | El compresor de alta presión, HPC. No es un dataset de múltiples fallos de turbina HPT. |
| ¿Qué es RUL? | Ciclos restantes hasta el criterio de fallo del simulador. En train: último ciclo menos ciclo actual. En test: NASA da la verdad del último ciclo observado. |
| ¿Por qué no dividir filas al azar? | Ciclos del mismo motor son dependientes y comparten historia; una mezcla produce una validación demasiado favorable. |
| ¿Por qué coinciden los números de ID entre train y test? | NASA reutiliza los enteros en dos flotas distintas. La identidad completa incluye origen: train:1 no es test:1. |
| ¿Dónde puede entrar fuga? | Medianas/escalado globales, ventanas centradas, relleno hacia atrás, normalización por vida total o motor completo, selección según la prueba, o reajuste con calibración tras calcular su intervalo. |
| ¿Cómo se evita? | Pipeline nuevo dentro de cada fold, ventanas hacia atrás, IDs fuera de X, reserva de calibración y prueba final tras congelación. |
| ¿Para qué limitar a 125? | Representar fase inicial de menor observabilidad; es una decisión de modelado, no una vida física. La verdad principal de evaluación queda completa. |
| ¿Por qué validar en un punto truncado? | Los finales de train tienen RUL=0 y no reproducen la tarea NASA. Se elige un ciclo elegible por motor antes del fallo, sin consultar prueba. |
| ¿Cuáles son los tamaños? | 80 motores desarrollo; CV con 64 ajuste/16 validación por fold; 20 calibración; 100 prueba oficial independiente. |
| ¿Se imputó de verdad? | Los TXT no tienen nulos. La lógica está preparada y cubre pendientes/desviaciones iniciales sin historial suficiente. No se inventan faltantes. |
| ¿Por qué conservar outliers? | Un sensor extremo cerca del fallo puede ser justamente el deterioro que se desea aprender. Se señalan, y solo se tratan imposibles físicos. |
| ¿Por qué Rankine ×5/9? | Ambas son escalas absolutas; no se aplica el desplazamiento de Celsius. Las presiones siguen siendo absolutas. |
| ¿Qué significa el contraste estadístico? | Una diferencia tardía-temprana por motor; Wilcoxon bajo hipótesis de simetría, corrección Holm de ocho pruebas, efecto e IC bootstrap. No se cuentan miles de ciclos como miles de motores independientes. |
| ¿Por qué cuatro familias? | Comparar una referencia simple, una relación lineal regularizada, bagging no lineal y boosting regularizado bajo el mismo protocolo. |
| ¿Qué mide NASA score? | Suma exponencial asimétrica: sobreestimar RUL cuesta más que subestimarla por el mismo número de ciclos. |
| ¿Qué es el intervalo conformal? | Predicción ± cuantil corregido de errores absolutos de motores reservados. Con 20 motores al 90%, orden 19. |
| ¿Garantiza que ese motor tenga 90%? | No. Es marginal bajo intercambio; además el mecanismo de pseudo-censura puede diferir de NASA. No es cobertura condicional ni simultánea de la trayectoria. |
| ¿Qué hace la regla operativa? | Prioriza revisión según límite inferior y reserva. El número de ciclos es presupuesto de reevaluación de un escenario; no cambia el mantenimiento aprobado. |
| ¿Cuánto dinero o AOG ahorra? | No está medido. Solo se analiza sensibilidad de alertas con costos relativos ilustrativos. Harían falta datos reales y un diseño de validación operacional. |
| ¿Qué cifra presentar? | RMSE y MAE oficiales sin recortar, NASA score, R², IC por motor y cobertura empírica; consultar reports/metrics.json y PDF. |

## Recorrido sugerido de la exposición

1. Problema de mantenimiento y alcance HPC FD001.
2. Calidad de fuente, unidades y diferencia entre extremo de desgaste e imposible físico.
3. Diagrama verbal 80 desarrollo → CV por motor → 20 calibración → modelo congelado → 100 prueba.
4. Una gráfica de evolución T30 y el contraste por motor.
5. Comparación de candidatos y cifras oficiales.
6. Ejemplo en la app: motor, ciclo pasado, banda y escenario de revisión.
7. Limitaciones y validación necesaria antes de cualquier uso real.

Todos deben poder explicar el pipeline completo. Dividir la exposición por secciones no reemplaza comprender las decisiones centrales.
