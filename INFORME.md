# Informe Técnico

## 1. Problema y antecedentes

El objetivo es analizar factores asociados a enfermedad cardíaca y construir un clasificador reproducible para `target` usando exclusivamente `heart.csv`. Esta entrega no parte de cero: toma como antecedente exploratorio el notebook de `Práctica2` y como antecedente inferencial/modelado el notebook de `tarea4`.

Los antecedentes usados fueron:

- `NICOLAS_CASANOVA_PRACTICA2/Nicolas_casanova.ipynb/Nicolas_casanova.ipynb`
- `tarea4/actividad_hipotesis_regresion.ipynb`

## 2. Datos y calidad

- Dataset fuente: `heart.csv`
- Filas originales: `1025`
- Columnas: `14`
- Variable objetivo: `target`
- Valores faltantes: `0`
- Duplicados exactos removidos para modelado: `723`
- Filas únicas finales para modelado: `302`

La discrepancia entre `1025` y `302` es central. El CSV original replica numerosas observaciones; por eso `tarea4` deduplica y esta entrega conserva esa decisión. Modelar sobre `1025` filas inflaría contrastes y métricas.

## 3. Qué aportaban los trabajos previos

### 3.1 Práctica2

El notebook previo trabajó el problema de enfermedad cardíaca con EDA, detección de outliers y dos regresiones simples:

- `age -> chol`: `R² = 0.0483`, correlación `0.2198`, `MSE = 2530.69`, `MAE = 38.37`
- `age -> thalach`: `R² = 0.1523`, correlación `-0.3902`

Conclusión útil: `age` por sí sola explica poco la variación de variables clínicas relevantes. Eso justificaba abandonar la regresión simple como estrategia principal.

### 3.2 tarea4

`tarea4` aportó tres piezas que se reproducen dentro de `tarea6`:

1. Contraste de hipótesis sobre `thalach` por grupos de `target`
2. Regresión lineal múltiple con `thalach` como dependiente
3. Baseline logístico con `thalach + exang`

Resultados de referencia reproducidos:

- Mann-Whitney U sobre `thalach`: `U = 5725`, `p ≈ 1.39772e-13`
- Regresión lineal múltiple:
  - `R² ≈ 0.2527`
  - `MSE ≈ 439.8511`
  - `R² ajustado train ≈ 0.2232`
- Baseline logístico:
  - mejor configuración `C=1`, `solver=liblinear`
  - `accuracy ≈ 0.7582`
  - `f1 ≈ 0.7755`
  - `specificity ≈ 0.7381`

## 4. Metodología de `tarea6`

### 4.1 Reproducción de antecedentes

El pipeline calcula de forma automática:

- regresiones simples de `Práctica2`
- normalidad por grupo con `scipy.stats.normaltest`
- Mann-Whitney U bilateral sobre `thalach`
- bootstrap IC 95% para diferencia de medianas de `thalach`
- chi-cuadrado entre `exang` y `target`
- regresión lineal múltiple con `statsmodels`
- baseline logístico de `tarea4`

### 4.2 Mejora iterativa del modelo

Sobre el dataset deduplicado se comparan tres familias:

1. `baseline_tarea4`
2. `logit_regularizado`
3. `random_forest`

Variables candidatas:

- `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`

Además, se añaden variables derivadas de apoyo:

- `chol_age_ratio`
- `pressure_oldpeak_interaction`
- `maxhr_age_gap`

Cada configuración se evalúa con:

- split estratificado 70/30
- validación cruzada de 5 folds
- accuracy
- precision
- recall
- specificity
- f1
- ROC AUC
- PR AUC

La selección del modelo final prioriza `cv_roc_auc_mean` y desempata por `f1`.

## 5. Resultados

### 5.1 Hipótesis

Los contrastes reproducen la evidencia de `tarea4`:

- `thalach` difiere significativamente entre grupos de `target`
- la normalidad no se sostiene limpiamente en ambos grupos, por lo que usar Mann-Whitney sigue siendo razonable
- `exang` también muestra asociación con `target` mediante chi-cuadrado

### 5.2 Regresión lineal

La regresión lineal múltiple explica una fracción moderada de `thalach`, pero no resuelve el objetivo de clasificación. Su valor en la entrega es interpretativo y como antecedente estadístico, no como modelo final.

### 5.3 Clasificación

El baseline reproducido de `tarea4` sirve como punto de comparación formal:

- `accuracy ≈ 0.7582`
- `f1 ≈ 0.7755`

La mejora de `tarea6` evalúa más predictores y regularización. El modelo elegido surge de la mejor combinación entre rendimiento en test y validación cruzada, con artefactos exportados para ROC, PR, matriz de confusión, probabilidades y ranking de variables.

## 6. Dashboard

El dashboard final incluye:

- KPIs de calidad de datos y rendimiento
- antecedente de `Práctica2` y `tarea4`
- EDA interactivo
- sección de hipótesis con boxplot de `thalach`
- comparación baseline vs modelo final
- curvas ROC / PR y matriz de confusión
- heatmap de experimentos

Se mantienen al menos tres callbacks interactivos:

1. cambio de variable continua para EDA
2. selección del experimento mostrado
3. cambio de métrica para el heatmap

## 7. Conclusiones

- El antecedente exploratorio mostró que la regresión simple era insuficiente.
- El antecedente de `tarea4` fijó un baseline útil y reproducible.
- La principal mejora de `tarea6` es convertir esos antecedentes en un pipeline reproducible y comparable, con validación cruzada y dashboard unificado.
- Usar las `302` filas únicas es una decisión metodológica obligatoria para evitar conclusiones optimistas por duplicación del dataset.
