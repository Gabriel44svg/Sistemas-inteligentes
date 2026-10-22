# Optimización experimental RMSLE — NYC Taxi

## Resultado verificable

**No se alcanzó un RMSLE de 0.20 ni se ha superado el 0.3704 que el usuario reporta en una evaluación externa.** No es legítimo prometer ninguno de esos resultados sin evaluarlos sobre las mismas filas y la misma métrica.

| Evaluación | LightGBM anterior | Ensamble nuevo |
|---|---:|---:|
| Validación (70,000 viajes) | 0.375735 | 0.374624 |
| Holdout (70,000 viajes) | 0.381219 | 0.379288 |

El modelo nuevo se seleccionó exclusivamente en validación y, posteriormente, se evaluó en holdout. Estas cifras no corresponden al archivo ciego `test.csv`.

## Cambios ensayados

- Variables originales (81) conservadas; limpieza e imputación ajustada solo en entrenamiento.
- Agrupación espacial MiniBatchKMeans ajustada con datos geográficos de entrenamiento.
- Ejes de Manhattan y distancias urbanas; se seleccionan 15 variables continuas.
- Agrupaciones origen-destino, zona/horario, zona/día/horario.
- Codificación de media suavizada en `log1p(trip_duration)` con cinco folds **out-of-fold** para que cada viaje de entrenamiento no vea su propia etiqueta.
- LightGBM sobre objetivo logarítmico para optimizar RMSLE; prueba de combinaciones de predicciones en escala log.
- Variantes con muchas variables categóricas fueron descartadas porque empeoraron (0.392–0.394).
- Ensamble final 50% modelo con 81 variables y 50% modelo con 104 variables.

**Protocolo anti-leakage:** `test.csv` no contiene etiqueta, el holdout no se usó para elegir parámetros o ponderaciones, y la eliminación de duración superior a 12 horas se aplicó solo a las muestras empleadas en el ajuste de los modelos. La validación mantuvo todos los valores extremos, incluso cuando eso penalizó el RMSLE.

## Ejecución

1. Ejecutar celdas del `Notebook_Optimizacion_RMSLE_Sin_Leakage.ipynb` de forma secuencial. Requiere memoria RAM suficiente (recomendable >= 8 GB) y Python con dependencias de `requirements.txt`.
2. Alternativamente ejecutar `python entrenar_mejora_final.py` para reproducir el reentrenamiento final ya seleccionado, con `train.csv` y `test.csv` en la carpeta del proyecto.
3. `python predecir_mejora.py` reproduce las predicciones usando `mejora_taxis/modelo_compacto.joblib`.
4. `mejora_taxis/submission_experimental.csv` es un archivo válido de 100,000 predicciones en segundos.

**Precaución:** el valor externo **0.3704** puede provenir de otro conjunto de evaluación. Este proyecto no demuestra mejora con respecto a ese valor; por eso el CSV nuevo se denomina *experimental* y no reemplaza el `submission.csv` de la entrega anterior.

## Archivos

- Notebook paso a paso.
- `train.csv`, `test.csv` (requeridos para reproducir).
- `features_taxi.py`, `_mejora_geografia_v2.py`, `entrenar_mejora_final.py`, `predecir_mejora.py`.
- `mejora_taxis/experimentos.csv` (pruebas descartadas), `mejora_taxis/v2/experimentos.csv` (comparación final), `mejora_taxis/v2/metricas_mejora.json`.
- `mejora_taxis/submission_experimental.csv`, modelos entrenados y metadatos finales.
