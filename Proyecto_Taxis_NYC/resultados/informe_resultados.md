# Informe final: duración de viajes de taxi NYC

## Problema
Predicción supervisada de tiempo de viaje (segundos), evaluada por RMSLE.

## Datos
- Train: 700,000 viajes.
- Test ciego: 100,000 viajes.
- Entrenamiento usado en selección tras filtrar duraciones >12h: 559,212.
- Validación: 70,000; holdout: 70,000.
- Variables derivadas: 81.
- Observaciones con duración >12h en train completo: 984.

## Resultado de experimentos
- Modelo ganador en validación: **LightGBM profundo**.
- **RMSLE validación: 0.375735**.
- **RMSLE holdout final: 0.381219**.
- MAE holdout: 270.92 segundos.
- RMSE holdout: 3062.55 segundos.
- R² holdout: 0.03899.
- MAPE holdout: 25.48% (inestable cerca de cero).

## Entrega
- `submission.csv`: **100,000** filas, exactamente `id,trip_duration`.
- `modelos/modelo_final.joblib`: modelo ajustado en train completo.
- `resultados/comparacion_modelos.csv`, `resultados/metricas.json`, `resultados/auditoria_calidad.json` y gráficos.

## Interpretación
RMSLE es la función de evaluación oficial: menor es mejor. F1, accuracy, precision y recall requieren clases y no son indicadores válidos para predecir valores continuos. Test ciego no permite calcular resultados contra la verdad hasta que se proporcionen sus etiquetas. El holdout jamás se usó para elegir el modelo.

## Fuentes docentes
- **U5 - 01 - ML.pdf:** clasificación vs. regresión (pág. 7).
- **U5 - 02 - Perceptrón.pdf:** regresión lineal y función de costo (págs. 3–7).
- **03 - Ciclo de vida del ML.pdf:** CRISP-DM, auditoría, partición, imputación, escalado, fuga de datos y métricas (págs. 2–5 y 8–25).
