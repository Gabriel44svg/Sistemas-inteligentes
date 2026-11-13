# Resultados del experimento — Spaceship Titanic

- Seleccionado en validación cruzada: **CatBoost**
- Umbral optimizado en validación cruzada: **0.38**
- Holdout independiente: **1739** pasajeros de grupos no vistos

| Métrica | Holdout |
|---|---:|
| F1 | 0.8163 |
| Accuracy | 0.8010 |
| Precision | 0.7629 |
| Recall | 0.8779 |
| Especificidad | 0.7231 |
| F1_macro | 0.7996 |
| BalancedAccuracy | 0.8005 |
| ROC_AUC | 0.9070 |
| PR_AUC | 0.9206 |
| LogLoss | 0.3736 |

La métrica de Kaggle es desconocida hasta evaluar allí `submission_spaceship_titanic.csv`. No se garantiza el óptimo global.

## Estrategia alternativa optimizada por accuracy (métrica oficial de Kaggle)

- Estrategia: **CatBoost**, umbral **0.47**

| Métrica | Holdout optimizado por accuracy |
|---|---:|
| F1 | 0.8138 |
| Accuracy | 0.8079 |
| Precision | 0.7952 |
| Recall | 0.8333 |
| Especificidad | 0.7822 |
| F1_macro | 0.8077 |
| BalancedAccuracy | 0.8077 |
| ROC_AUC | 0.9070 |
| PR_AUC | 0.9206 |
| LogLoss | 0.3736 |
