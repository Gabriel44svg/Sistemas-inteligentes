# Guía de entrega para exposición: Fourier y Wavelets Haar

Este ZIP contiene el flujo completo del proyecto, no solo las diapositivas. La presentación fue compilada con **pdfLaTeX** y el guion fue preparado para explicar **las 62 diapositivas**, desde la limpieza inicial hasta el ensamble final.

## 1. Para exponer inmediatamente
- Abre **`Presentacion_Completa_Fourier_Wavelets.pdf`** y proyecta las diapositivas.
- Lee **`Guion_Detallado_Fourier_Wavelets.pdf`** para saber qué explicar, por qué se hizo y cómo responder preguntas.
- Para modificar nombres, texto, estilo, fórmulas o añadir contenido, edita ambos archivos `.tex`.
- Los archivos `.tex` usan el directorio `figuras/` para incorporar las gráficas y están listos para **pdfLaTeX en Overleaf**.
- **Para Overleaf conviene subir el ZIP liviano de presentación**, no este ZIP con CSV y modelos grandes. También se entrega esa versión por separado.

## 2. Organización y evidencia del proyecto
- `Notebook_Wavelets_Fourier_Taxis_NYC.ipynb`: notebook ejecutado con explicación y gráficos.
- `train.csv` / `test.csv`: 700,000 viajes con etiqueta / 100,000 sin etiqueta.
- `submission_wavelet_fft.csv`: entrega final de 100,000 filas, columnas `id,trip_duration` (verificado de nuevo).
- `features_taxi.py`: 81 variables iniciales. `wavelet_fft_taxi.py`: 45 bases fijas, 27 señales de demanda y 14 estadísticas OOF.
- `experimento_wavelets_fft.py`, `evaluar_ensambles_wavelet.py`, `evaluar_holdout_wavelet.py`: trazabilidad de experimentos y selección.
- `entrenar_wavelet_final.py` y `predecir_wavelet_fft.py`: entrenamiento y reproducción del CSV.
- `resultados_wavelet/`: gráficos, comparación de modelos, diagnóstico y archivos del modelo final.
- `mejora_taxis/`: ensamble geográfico anterior y sus resultados.
- `material_complementario/`: notebooks de los experimentos original y geográfico, así como la exposición de la etapa inicial.
- `material_clase/`: tres diapositivas originales de clase en PDF.

## 3. Resultados para explicar
- Mejor ensamble geográfico previo: **validación 0.374624**, holdout **0.379288**.
- Nuevo ensamble (75% geográfico + 25% Fourier/Haar): **validación 0.373800**, holdout **0.378590**.
- Mejor componente Fourier/Haar aislado: **0.376372** en validación (no superó la referencia).
- Meta RMSLE 0.20: **no alcanzada**.
- Puntaje 0.3704 comunicado anteriormente por el usuario: **no es una medición local del nuevo CSV**.

## 4. Cómo ejecutar
```bash
pip install -r requirements_wavelet.txt
jupyter notebook Notebook_Wavelets_Fourier_Taxis_NYC.ipynb
```
Reproducción desde cero de selección/validación:
```bash
python _mejora_geografia_v2.py
python experimento_wavelets_fft.py
python evaluar_ensambles_wavelet.py
python evaluar_holdout_wavelet.py
```
Reentrenamiento del CSV y auditoría reproducida:
```bash
python entrenar_mejora_final.py
python entrenar_wavelet_final.py
python predecir_wavelet_fft.py
```
**Aviso:** requiere memoria RAM y tiempo de cómputo. Los modelos preentrenados y un CSV correcto ya están incluidos, así que no necesitas volver a entrenar para exponer.

## 5. Importante sobre data leakage
Se separó train/validación/holdout y se generaron variables supervisadas fuera de cada fold de entrenamiento. Esto protege la evaluación retrospectiva aleatoria. Para un producto que prediga viajes futuros, la división por fecha y las agregaciones estrictamente pasadas serían necesarias. No afirmar que la evaluación aleatoria demuestra ausencia de fuga temporal en producción.

## 6. Procedencia
Las definiciones de clasificación/regresión, CRISP-DM, imputación, escalado y prevención de leakage se conectan con las tres presentaciones del curso. Fórmulas de Fourier/Haar y resultados experimentales provienen de los scripts/notebook del proyecto.
