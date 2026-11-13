# Spaceship Titanic — proyecto de Machine Learning

**Alumno:** Christian Gabriel Villafuerte Aguilar — UNAM FES Acatlán — Sistemas Inteligentes.

## Abrir y ejecutar

1. Descomprime este ZIP manteniendo subcarpetas.
2. Instala Python 3.10 o posterior. Abre terminal en la carpeta del proyecto.
3. Ejecuta `python -m pip install -r requirements.txt`.
4. Abre `jupyter notebook Spaceship_Titanic_ML_Completo.ipynb` o VS Code y ejecuta **Run All**.
5. Los experimentos pueden tardar varios minutos. Se ejecutan sin Internet.
6. Para enviar a Kaggle, sube `resultados/submission_spaceship_titanic.csv` al concurso **Spaceship Titanic**.

## Contenido

- `Spaceship_Titanic_ML_Completo.ipynb`: notebook paso a paso con resultados ejecutados y gráficas.
- `data/`: `train.csv` (8693 etiquetas), `test.csv` (4277), `sample_submission.csv`.
- `diapositivas/`: tres diapositivas PDF usadas como marco didáctico.
- `resultados/`: auditorías, gráficos, modelos comparados, métricas del holdout, CSV de Kaggle y reporte.
- `modelos/modelo_final.joblib`: objeto persistido con el modelo o ensamble y umbral.
- `requirements.txt`: bibliotecas usadas.

## Validez metodológica

Se usa F1 clase positiva como criterio de selección en validación cruzada de 3 folds agrupada por ID y un holdout aislado por grupos. La preparación y los escaladores se ajustan dentro del entrenamiento de cada fold. No se conocen las etiquetas del test de Kaggle; el notebook **no inventa** accuracy de Kaggle. La ingeniería de características combina covariables train y test sin etiquetas para obtener información del grupo (transductiva). La búsqueda no asegura el máximo posible sobre todas las arquitecturas existentes.

El notebook se puede ejecutar desde la carpeta del proyecto. Si quieres evaluar observaciones ajenas a esos archivos, debes actualizar la función de agregados por grupos del lote nuevo.

## Dos archivos de predicciones

- `resultados/submission_spaceship_titanic.csv`: optimiza F1 positivo, objetivo principal del notebook.
- `resultados/submission_kaggle_accuracy.csv`: optimiza accuracy en CV; en Spaceship Titanic es la métrica de clasificación del concurso.

Se ofrecen ambos sin optimizar contra el holdout; los resultados de holdout se reportan por separado. Las curvas de aprendizaje y tablas CSV muestran comparaciones reproducibles.
