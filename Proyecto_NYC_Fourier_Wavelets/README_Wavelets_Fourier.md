# NYC Taxi — mejoras con análisis espectral y wavelets Haar

## Resumen comprobado

Se evaluaron variables basadas en la transformada rápida de Fourier (FFT) y wavelets Haar, además de combinar un LightGBM espectral con el ensamble geográfico anterior. El **resultado optimizado no es 0.20** y no puede afirmarse que se haya superado el **0.3704 que reportaste en el test ciego** sin someter el nuevo CSV a la evaluación oficial.

| Evaluación | Mejor ensamble previo | Nuevo ensamble Fourier + Haar | Reducción absoluta |
| --- | ---: | ---: | ---: |
| Validación (70,000 viajes) | 0.374624 | **0.373800** | 0.000824 |
| Holdout (70,000 viajes) | 0.379288 | **0.378590** | 0.000699 |

En validación el ganador fue **75% ensamble geográfico anterior + 25% modelo Fourier/Haar**, ponderados en escala `log1p`. Se escogió exclusivamente en validación; el holdout se midió después, una única vez para ese candidato. La métrica oficial es RMSLE: **más bajo es mejor**.

## ¿Cómo se usaron las wavelets?

- Se agregaron **45 bases deterministas** (armónicos diarios/semanales/estacionales y geográficos, funciones Haar de fase horaria). No usan `trip_duration`.
- Se construyeron **27 variables de demanda** con conteos de viajes por 30 minutos en entrenamiento, reconstrucciones Haar de aproximación/detalle a diversas escalas y bandas FFT.
- Se crearon **14 estadísticas de duración espectrales** (medias de `log1p(trip_duration)` por franja de media hora semanal, suavizadas, y sus reconstrucciones FFT y Haar). **Todas se calculan out-of-fold** durante el entrenamiento: las etiquetas del propio fold se excluyen. La validación y el test se transforman solo con información del entrenamiento.
- LightGBM profundo se ajustó sobre `log1p(trip_duration)` y se comparó en la misma partición que el proyecto anterior, manteniendo casos extremos en validación/holdout.
- Se compararon **126, 153 y 167 variables**. Por separado, los modelos espectrales fueron inferiores al anterior, pero sus errores complementarios permitieron una mejora medible en el ensamble.
- No fue necesario escalar con StandardScaler o MinMaxScaler para LightGBM, ya que divide por umbrales. La imputación numérica por mediana se ajusta exclusivamente sobre train. El escalado para Ridge fue parte del notebook anterior, no de esta etapa.

## Por qué no es fácil llegar a RMSLE 0.20

Entre 70,000 observaciones de validación hay **103 duraciones superiores a 12 horas**. Con el modelo final, su contribución representa cerca del **24.4% del error cuadrático logarítmico agregado**, aunque son aproximadamente el 0.15% de las filas. Esto es un diagnóstico de errores *del modelo actual*, no una cota matemática infranqueable. Otras variables ausentes, como velocidad real, tráfico o ruta seguida, limitan la precisión.

## Protocolo anti-data-leakage

1. Partición fija inicial: 80% train, 10% validación, 10% holdout (`random_state=42`).
2. Limpieza por objetivo (`1 <= trip_duration <= 43200`) únicamente **en los datos para ajustar los modelos**. No se filtran los valores extremos para medir RMSLE.
3. Las estadísticas que usan `trip_duration` se construyen con **5 folds OOF** dentro de train. La prueba del notebook altera el objetivo de una fila y verifica que *sus propias variables OOF* no cambian.
4. El imputador solo utiliza entrenamiento. `test.csv` se procesa sin acceder a una etiqueta desconocida y no se usa para elegir hiperparámetros.
5. Se ensambla en escala logarítmica; no se seleccionan pesos con holdout ni con test ciego.
6. **Alcance:** este protocolo es válido para la evaluación retrospectiva aleatoria de este reto. Si el propósito fuera desplegar un servicio en fechas futuras, debe utilizarse una división por fecha y estadísticas estrictamente causales para impedir uso de etiquetas futuras en la línea de tiempo.

## Entregables

- `Notebook_Wavelets_Fourier_Taxis_NYC.ipynb`: 30 celdas, ejecutado, explicación completa, gráficos, pseudoprueba OOF, resultados comparados.
- `wavelet_fft_taxi.py`: implementación NumPy de reconstrucción Haar/FFT, y codificador OOF.
- `experimento_wavelets_fft.py`: entrena tres variantes, crea modelos/CSV de métricas. Usa modelos de referencia de `mejora_taxis/v2/`.
- `evaluar_ensambles_wavelet.py`: reconstruye y optimiza únicamente sobre validación las mezclas espectrales + referencia anterior.
- `evaluar_holdout_wavelet.py`: prueba holdout para el candidato elegido; no debe usarse para buscar pesos.
- `entrenar_wavelet_final.py`: reentrena con datos elegibles y exporta `submission_wavelet_fft.csv`.
- `predecir_wavelet_fft.py`: recarga modelos guardados y reproduce el envío sin volver a entrenar.
- `resultados_wavelet/comparacion_completa.csv`, `diagnostico_por_duracion.csv`, `importancia_wavelet.csv`, `*.png`: métricas, diagnóstico y gráficos.
- `train.csv` y `test.csv`: archivos originales.
- `mejora_taxis/modelo_compacto.joblib`: ensamble geográfico anterior para reproducción.
- `resultados_wavelet/modelo_wavelet_final.joblib`: componente espectral seleccionado, una vez reentrenado.

## Ejecución

```bash
pip install -r requirements_wavelet.txt
jupyter notebook Notebook_Wavelets_Fourier_Taxis_NYC.ipynb
```

Para reproducir **toda la validación**, primero regenera los modelos del experimento geográfico previo; pueden requerir varios minutos y bastante RAM:

```bash
python _mejora_geografia_v2.py
python experimento_wavelets_fft.py
python evaluar_ensambles_wavelet.py
python evaluar_holdout_wavelet.py
```

El **CSV final** se produce con:

```bash
python entrenar_mejora_final.py     # si aún no existe el modelo anterior, tarda varios minutos
python entrenar_wavelet_final.py    # componente nuevo, tarda varios minutos
python predecir_wavelet_fft.py      # verificación exacta desde los modelos guardados
```

La nueva entrega propuesta se llama **`submission_wavelet_fft.csv`** y conserva exactamente las columnas `id,trip_duration`, con una predicción numérica no negativa por identificador y 100,000 filas. Es **experimental** hasta que la plataforma comunique su score. Conserva tu CSV anterior que obtuvo 0.3704.

## Consideraciones

- Se usa una implementación manual de Haar sobre NumPy, equivalente a reconstrucciones por niveles dyádicos, para no depender de `PyWavelets` que no estaba disponible en este entorno.
- Un mejor RMSLE local no garantiza un mejor valor en test ciego. No se ha obtenido 0.20.
- Computadora sugerida: memoria RAM de 8 GB o más y varias CPU; LightGBM puede tardar varios minutos.
