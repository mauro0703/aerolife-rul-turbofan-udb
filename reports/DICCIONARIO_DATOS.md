# Diccionario de entrada y salida

Entrada CSV de inferencia: nombres exactos y unidades **nativas NASA**, no unidades SI ya convertidas. Se requieren todas las observaciones desde el ciclo 1, consecutivas por motor; ID-ciclo único. s1..s21 conservan el orden de Saxena et al. (2008), tabla 2.

| Campo | Símbolo / contenido | Unidad original | Unidad de análisis |
| --- | --- | --- | --- |
| engine_id | Identificador de la unidad | entero positivo | metadato, nunca predictor |
| cycle | Ciclo de operación | ciclo | edad en ciclos |
| setting_1..3 | Ajustes operacionales | escala nativa | escala nativa |
| s1 | T2, entrada fan | °R | K |
| s2 | T24, salida LPC | °R | K |
| s3 | T30, salida HPC | °R | K |
| s4 | T50, salida LPT | °R | K |
| s5 | P2, entrada fan | psia | kPa absoluto |
| s6 | P15, ducto bypass | psia | kPa absoluto |
| s7 | P30, salida HPC | psia | kPa absoluto |
| s8 | Nf, fan físico | rpm | rpm |
| s9 | Nc, core físico | rpm | rpm |
| s10 | epr, P50/P2 | adimensional | adimensional |
| s11 | Ps30, estática HPC | psia | kPa absoluto |
| s12 | phi, flujo combustible/Ps30 | (lbm/s)/psi | (kg/s)/kPa |
| s13 | NRf, fan corregido | rpm | rpm |
| s14 | NRc, core corregido | rpm | rpm |
| s15 | BPR | adimensional | adimensional |
| s16 | farB | adimensional | adimensional |
| s17 | htBleed | sin unidad inequívoca en fuente | escala nativa |
| s18 | Nf_dmd | rpm | rpm |
| s19 | PCNfR_dmd | fuente tabla 2 etiqueta rpm; escala numérica ambigua | nativa, sin conversión supuesta |
| s20 | W31, sangrado refrigeración HPT | lbm/s | kg/s |
| s21 | W32, sangrado refrigeración LPT | lbm/s | kg/s |

Los canales refrigerantes HPT/LPT son mediciones de respuesta del motor; su presencia no convierte el modo de fallo FD001 en degradación HPT/LPT.

Salida: `rul_pred` [ciclos]; `lower90`,`upper90` [ciclos] del intervalo nominal; `status` y `revisit_cycles` del escenario; `short_history` para ciclos <30; `features_beyond_6sigma` como señal descriptiva de valores alejados del ajuste (no detector calibrado de anomalías). En prueba se agregan `rul_true` y `error` exclusivamente para evaluación.
