# Proyecto de aprendizaje automático: duración de viajes NYC

**Asignatura:** Sistemas Inteligentes · FES Acatlán (UNAM)

## Objetivo
Predecir la duración de cada viaje (`trip_duration`, **segundos**) del archivo `test.csv`, usando `train.csv`. El archivo de entrega a la plataforma de evaluación es **`submission.csv`** y tiene **exactamente dos columnas: `id,trip_duration`**, con **100,000 registros** y valores numéricos no negativos.

## Cómo revisar la actividad (paso a paso)
1. Abre `Prediccion_Taxis_NYC_Paso_a_Paso.ipynb` en Jupyter Notebook, JupyterLab, VS Code o Google Colab (en Colab debes cargar los CSV en la misma carpeta).
2. El notebook **ya está ejecutado**, con resultados, tablas y visualizaciones. Lee las explicaciones de cada sección antes de ejecutar nuevamente.
3. Sigue la secuencia: definición del problema y RMSLE → carga y auditoría → anomalías → exploración → partición 80/10/10 → ingeniería de 81 atributos → imputación → escalado Ridge → LightGBM/XGBoost/CatBoost → selección por RMSLE → evaluación holdout → modelo final → predicción test → validación del CSV.
4. Para reproducir, instala `pip install -r requirements.txt` y ejecuta las celdas del notebook en orden. Este proceso puede tardar varios minutos en CPU y necesita suficiente memoria RAM.

## Resultados y archivos
- `submission.csv`: **ENTREGA DIRECTA**; contiene únicamente `id,trip_duration`.
- `Prediccion_Taxis_NYC_Paso_a_Paso.ipynb`: notebook desarrollado y ejecutado.
- `modelos/modelo_final.joblib`: modelo ganador reajustado sobre entrenamiento completo utilizable.
- `predecir.py` y `features_taxi.py`: script reutilizable para generar nuevas predicciones con el modelo guardado.
- `resultados/informe_resultados.md`: informe de resultados y conclusiones.
- `resultados/metricas.json`: RMSLE holdout/validación y parámetros.
- `resultados/comparacion_modelos.csv`: registro de experimentos.
- `resultados/auditoria_calidad.json`: resumen de calidad de los datos.
- `resultados/error_por_distancia.csv`: diagnóstico según la distancia.
- `resultados/importancia_variables.csv` y `resultados/graficos/`: interpretaciones y gráficos del proceso.
- `train.csv`, `test.csv`: datos originales incluidos para repetir la actividad.
- `material_clase/`: los tres PDF de referencia aportados en clase.
- `requirements.txt`: dependencias utilizadas.

## Cómo generar otra entrega sin reentrenar
Desde la carpeta descomprimida:

```bash
python predecir.py test.csv submission_recalculada.csv
```

El script carga los pesos guardados, reproduce el feature engineering y verifica la validez del CSV.

## Por qué no se reportan F1, accuracy, precision o recall
Se trata de **regresión de una variable continua**. F1, accuracy, precision y recall son métricas de clasificación y requieren clases discretas. No se discretiza la duración, pues hacerlo cambiaría el objetivo del concurso. Aquí es **RMSLE** la métrica oficial que se minimiza, complementada por MAE, RMSE, R² y MAPE. **No se conoce el puntaje en `test.csv` sin sus verdaderas duraciones**: la evaluación holdout es una estimación del desempeño de generalización.

## Fuente de trabajo
Se retoman las diapositivas: `U5 - 01 - ML.pdf`, `U5 - 02 - Perceptrón.pdf` y `03 - Ciclo de vida del ML.pdf`. La construcción de variables geográficas y comparación de algoritmos son extensiones al reto, no contenido atribuido a esas diapositivas.
