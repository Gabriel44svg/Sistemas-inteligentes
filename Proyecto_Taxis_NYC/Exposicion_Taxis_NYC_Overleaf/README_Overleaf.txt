PRESENTACIÓN EN LATEX - PROYECTO TAXIS NYC
========================================

Archivos principales:
- Presentacion_Taxis_NYC.tex  -> diapositivas de clase (Beamer, 37 páginas).
- Presentacion_Taxis_NYC.pdf  -> PDF listo para proyectar.
- Guion_Exposicion_Taxis_NYC.tex -> documento con explicación y justificación por diapositiva.
- Guion_Exposicion_Taxis_NYC.pdf -> guion listo para estudiar (9 páginas).
- figuras/ -> 4 gráficos extraídos del notebook ejecutado.

PARA UTILIZAR EN OVERLEAF
1. Descarga Exposicion_Taxis_NYC_Overleaf.zip.
2. En Overleaf: New Project > Upload Project > selecciona el ZIP.
3. Compilador: pdfLaTeX.
4. Documento principal: Presentacion_Taxis_NYC.tex.
5. Si quieres compilar el guion, cambia el documento principal a
   Guion_Exposicion_Taxis_NYC.tex.

Puedes cambiar el nombre del autor, asignatura o fecha en la portada
modificando Presentacion_Taxis_NYC.tex. Las figuras relativas se mantienen
si respetas la carpeta figuras/.

SOBRE LOS RESULTADOS
Los puntajes y números se basan en el notebook ejecutado
Prediccion_Taxis_NYC_Paso_a_Paso.ipynb y sus archivos de resultados.
Mejor candidato ensayado: LightGBM profundo.
RMSLE validación: 0.375735. RMSLE holdout: 0.381219.
El conjunto test ciego no incluye las duraciones reales, por tanto NO
podemos asegurar ni calcular aquí su RMSLE oficial.

NOTA DIDÁCTICA
Accuracy, F1, precision y recall son métricas de clasificación;
el reto de tiempos continuos se evalúa con RMSLE y otras métricas de regresión.

Las diapositivas de clase originales y los CSV no se repiten en este ZIP;
ya están en el ZIP del proyecto completo entregado previamente.
