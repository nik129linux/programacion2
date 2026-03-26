# Dashboard de Ciencia de Datos

Proyecto autocontenido sobre enfermedad cardíaca construido a partir del trabajo previo del curso. La entrega usa como dataset principal `heart.csv` y extiende dos antecedentes:

- `NICOLAS_CASANOVA_PRACTICA2/Nicolas_casanova.ipynb/Nicolas_casanova.ipynb`
- `tarea4/actividad_hipotesis_regresion.ipynb`

Toda la implementación nueva queda dentro de `tarea6` y no modifica los directorios originales.

## Estructura

```text
.
├── app.py
├── binder/
├── dashboard/
│   ├── app.py
│   ├── assets/style.css
│   ├── callbacks/main.py
│   ├── layouts/main_layout.py
│   └── utils/data_loader.py
├── data/
│   ├── dashboard/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── screenshots/
│   └── superpowers/
├── models/
├── src/proyecto_ciencia_datos/
└── tests/
```

## Qué incluye

- Copia local de `heart.csv` en `data/raw/heart.csv`.
- Pipeline reproducible que separa:
  - dataset bruto: `1025` filas
  - dataset de modelado: `302` filas únicas tras `drop_duplicates()`
- Reproducción programática de antecedentes:
  - regresiones simples de `Práctica2`
  - contraste de hipótesis y baseline logístico de `tarea4`
  - regresión lineal múltiple de `tarea4`
- Comparación final entre:
  - baseline logístico `thalach + exang`
  - regresión logística regularizada multivariable
  - random forest ajustado
- Dashboard en Dash con EDA, pruebas de hipótesis, comparación de modelos y exploración de hiperparámetros.

## Instalación

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

## Ejecución

La primera vez que se arranca la aplicación se generan automáticamente los artefactos de `data/` y `models/`.

```bash
./.venv/bin/python app.py
```

La app queda disponible en `http://127.0.0.1:8050`.

## Verificación

```bash
./.venv/bin/pytest tests -q
```

## Hallazgos clave

- El CSV original contiene `1025` filas, pero `723` son duplicados exactos.
- El modelado se hace sobre `302` filas únicas para evitar métricas infladas.
- `Práctica2` mostró que las regresiones simples con `age` no explicaban suficientemente `chol` ni `thalach`.
- `tarea4` estableció como baseline una logística con `thalach + exang`, con:
  - `accuracy ≈ 0.7582`
  - `f1 ≈ 0.7755`
- `tarea6` amplía ese baseline con más predictores, regularización y validación cruzada.

## Dashboard

El dashboard incluye:

- resumen ejecutivo con KPIs de calidad de datos y rendimiento
- histogramas, boxplots y scatter matrix
- pruebas de normalidad, Mann-Whitney U y chi-cuadrado
- comparación entre baseline y modelo final
- curvas ROC / PR, matriz de confusión y tabla de métricas
- heatmap de configuraciones evaluadas

## Binder

Después de publicar el repositorio, el lanzamiento esperado es:

```text
https://mybinder.org/v2/gh/<usuario>/programacion2/HEAD?urlpath=proxy/8050/
```
