# Informe Técnico

## 1. Problema y antecedentes

El objetivo es analizar factores asociados a enfermedad cardíaca y construir un clasificador reproducible para `target` usando exclusivamente `heart.csv`. Esta entrega no parte de cero: toma como antecedente exploratorio el notebook de `Práctica2` y como antecedente inferencial/modelado el notebook de `tarea4`.

Los antecedentes usados fueron:

- `NICOLAS_CASANOVA_PRACTICA2/Nicolas_casanova.ipynb/Nicolas_casanova.ipynb`
- `tarea4/actividad_hipotesis_regresion.ipynb`

## 2. Datos, calidad y deduplicación

- Dataset fuente: `heart.csv`
- Filas originales: `1025`
- Columnas: `14`
- Variable objetivo: `target`
- Valores faltantes: `0`
- Duplicados exactos removidos para modelado: `723`
- Filas únicas finales para modelado: `302`

La discrepancia entre `1025` y `302` es central. El CSV original contiene muchas filas exactamente idénticas en todas las variables. En términos operativos, la deduplicación se hizo con `drop_duplicates()` exacto fila a fila, porque ese fue también el criterio adoptado en `tarea4` y porque conservar todas las repeticiones habría inflado artificialmente las métricas del modelo y los p-values de los contrastes.

La limitación metodológica debe reconocerse explícitamente: una deduplicación exacta no permite distinguir entre dos pacientes distintos que casualmente compartan todos los mismos valores observados y un mismo registro repetido varias veces. Aun así, para esta entrega es más defendible modelar sobre las `302` filas únicas, porque tratar las `723` repeticiones exactas como evidencia independiente sesgaría en forma optimista la inferencia y la evaluación predictiva.

## 3. Qué aportaban los trabajos previos

### 3.1 Práctica2

El notebook previo trabajó el problema de enfermedad cardíaca con EDA, detección de outliers y dos regresiones simples:

- `age -> chol`: `R² = 0.0483`, correlación `0.2198`, `MSE = 2530.69`, `MAE = 38.37`
- `age -> thalach`: `R² = 0.1523`, correlación `-0.3902`

Conclusión útil: `age` por sí sola explica poco la variación de variables clínicas relevantes. Eso justificaba abandonar la regresión simple como estrategia principal y pasar a clasificación supervisada.

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
- Mann-Whitney U bilateral sobre `oldpeak`
- chi-cuadrado entre `exang` y `target`
- chi-cuadrado entre `cp` y `target`
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
- validación cruzada estratificada de 5 folds
- accuracy
- precision
- recall
- specificity
- f1
- ROC AUC
- PR AUC

La selección del modelo final prioriza `cv_roc_auc_mean` y desempata por `f1`.

### 4.3 Regularización

El modelo final es una regresión logística multivariable con regularización `L2`, que es la penalización asociada al solver `lbfgs` en `scikit-learn`. Esta elección es razonable porque reduce el peso de coeficientes grandes y estabiliza el ajuste sin forzar sparsity agresiva. En este problema no había evidencia suficiente para asumir de antemano que varias features eran irrelevantes, por lo que `L2` era una opción más prudente que `L1`.

## 5. Resultados

### 5.1 Contrastes de hipótesis

Los contrastes principales quedan así:

| Prueba | Variable / relación | Estadístico | p-value | Resultado |
|---|---|---:|---:|---|
| Mann-Whitney U | `thalach` vs `target` | `5725.0000` | `1.39772e-13` | diferencia significativa |
| Mann-Whitney U | `oldpeak` vs `target` | `16722.5000` | `3.34699e-13` | diferencia significativa |
| Chi-cuadrado | `exang` vs `target` | `55.4562` | `9.55647e-14` | asociación significativa |
| Chi-cuadrado | `cp` vs `target` | `80.9788` | `1.89268e-17` | asociación significativa |

Interpretación:

- `thalach` y `oldpeak` muestran diferencias claras entre los grupos con y sin enfermedad cardíaca.
- `exang` y `cp` también se asocian con `target`, por lo que la entrega no queda apoyada en un solo contraste.
- La normalidad no se cumple limpiamente en todos los grupos, así que el uso de pruebas no paramétricas fue una decisión coherente.

### 5.2 Regresión lineal

La regresión lineal múltiple con `thalach` como dependiente mantiene el valor interpretativo observado en `tarea4`:

- `R² test ≈ 0.2527`
- `MSE test ≈ 439.8511`

Explica una fracción moderada de la variación de `thalach`, pero no resuelve por sí misma el problema de clasificación binaria.

### 5.3 Mejora iterativa documentada

| Iteración | Modelo | Accuracy | F1 | ROC AUC | Decisión siguiente |
|---|---|---:|---:|---:|---|
| baseline Práctica2 | Regresiones simples `age -> chol` y `age -> thalach` | N/A | N/A | N/A | pasar de análisis exploratorio a clasificación supervisada |
| baseline tarea4 | Baseline logístico `thalach + exang` | `0.7582` | `0.7755` | `0.7957` | ampliar predictores y usar validación cruzada |
| logit multivariable | Logit multivariable regularizado | `0.8132` | `0.8211` | `0.8931` | contrastar contra un modelo no lineal |
| random forest | Random forest multivariable | `0.7692` | `0.7835` | `0.8732` | seleccionar por desempeño CV y estabilidad, no solo score puntual |
| modelo final | Logit multivariable regularizado | `0.8132` | `0.8211` | `0.8931` | conservar por mejor equilibrio entre desempeño, CV e interpretabilidad |

La progresión muestra una mejora real respecto del baseline de `tarea4`. El random forest no superó al logit regularizado en el criterio de selección usado para la entrega.

### 5.4 Validación cruzada del modelo final

Para el mejor modelo (`Logit multivariable regularizado`, `C=0.1`, `solver=lbfgs`), la validación cruzada de 5 folds arrojó:

| Métrica | Media CV | Desviación estándar |
|---|---:|---:|
| Accuracy | `0.8579` | `0.0603` |
| F1 | `0.8741` | `0.0528` |
| ROC AUC | `0.9212` | `0.0370` |
| PR AUC | `0.9397` | `0.0254` |

Esto permite reportar no solo el promedio, sino también la dispersión entre folds, como pide una evaluación más robusta.

### 5.5 Intervalos de confianza del modelo final

Los intervalos de confianza bootstrap al 95% para las métricas del modelo final fueron:

| Métrica | Valor puntual | IC 95% |
|---|---:|---|
| Accuracy | `0.8132` | `[0.8139, 0.8234]` |
| Precision | `0.8478` | `[0.8472, 0.8590]` |
| Recall | `0.7959` | `[0.7949, 0.8081]` |
| Specificity | `0.8333` | `[0.8327, 0.8453]` |
| F1 | `0.8211` | `[0.8201, 0.8299]` |
| ROC AUC | `0.8931` | `[0.8914, 0.8994]` |
| PR AUC | `0.8953` | `[0.8938, 0.9039]` |

Estas bandas están exportadas también en `metric_confidence_intervals.csv` y se muestran en el dashboard cuando se consulta el experimento final.

## 6. Dashboard

El dashboard final incluye:

- KPIs de calidad de datos y rendimiento
- antecedente de `Práctica2` y `tarea4`
- EDA interactivo
- sección de hipótesis con boxplot de `thalach`
- tabla de contrastes que ahora incluye `thalach`, `oldpeak`, `exang` y `cp`
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
- La principal mejora de `tarea6` es convertir esos antecedentes en un pipeline reproducible y comparable, con validación cruzada, incertidumbre explícita e integración en dashboard.
- La deduplicación exacta tiene una limitación conceptual, pero conservar las `1025` filas habría inflado artificialmente métricas y significancia estadística.
- El modelo final elegido fue la regresión logística multivariable con regularización `L2`, porque mejoró con claridad al baseline y mantuvo una interpretación más directa que el random forest.
