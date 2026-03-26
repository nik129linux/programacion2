# CODEX PROMPT — Dashboard de Ciencia de Datos (Actividad Final)

## 0 · CONTEXTO DEL PROYECTO
Eres un ingeniero de datos senior. Este repositorio contiene un proyecto de ciencia de datos
en curso (datos, notebooks, scripts, modelos). Tu trabajo es:
1. Leer y comprender **todos** los archivos del directorio actual (datos, notebooks, scripts, configs, READMEs).
2. Diagnosticar el estado actual del modelo estadístico.
3. Mejorarlo iterativamente.
4. Construir un dashboard interactivo con Dash + Plotly que presente los hallazgos.
5. Generar toda la documentación y archivos de despliegue necesarios.

Antes de escribir una sola línea de código, lista los archivos encontrados, resume qué hace
cada uno y propón un plan de trabajo numerado. Espera confirmación (o ajústalo tú mismo si
estás en modo autónomo) y luego ejecuta paso a paso.

---

## 1 · EXPLORACIÓN Y DIAGNÓSTICO (Pre-código)
- Escanea recursivamente el directorio: identifica datasets (`.csv`, `.xlsx`, `.parquet`, `.json`),
  notebooks (`.ipynb`), scripts (`.py`), modelos serializados (`.pkl`, `.joblib`, `.h5`), y configs.
- Resume en un bloque Markdown:
  - Problemática definida originalmente.
  - Variables objetivo y predictoras.
  - Métricas de rendimiento actuales del modelo (accuracy, RMSE, R², AUC, etc.).
  - Posibles problemas detectados: sobreajuste, datos faltantes, desbalance de clases,
    multicolinealidad, features irrelevantes.

---

## 2 · MEJORA ITERATIVA DEL MODELO
Aplica las siguientes técnicas **solo si** el diagnóstico lo justifica. Documenta cada decisión
con un comentario `# DECISIÓN: <justificación>`.

### 2.1 Datos
- Limpieza adicional: imputación, detección de outliers, encoding categórico.
- Feature engineering: crear o transformar variables que aporten poder predictivo.
- Si los datos son insuficientes, documenta la limitación.

### 2.2 Diseño de experimentos e incertidumbre
- Implementa **contraste de hipótesis** (t-test, chi², ANOVA, etc.) para validar supuestos.
- Calcula **intervalos de confianza** de las métricas clave.
- Aplica **validación cruzada** (k-fold ≥ 5) para estimar incertidumbre del rendimiento.
- Si es pertinente, realiza un **diseño factorial** o **A/B test** sobre hiperparámetros.

### 2.3 Optimización de hiperparámetros
- Usa `GridSearchCV`, `RandomizedSearchCV` o `Optuna`.
- Registra cada experimento en un DataFrame comparativo:
  `[experiment_id, params, metric_1, metric_2, …, timestamp]`.
- Guarda el mejor modelo serializado en `models/best_model.<ext>`.

### 2.4 Regularización
- Si se detecta sobreajuste (train >> val), aplica L1/L2, Dropout, early stopping, o pruning.
- Compara métricas antes y después; conserva ambas versiones.

### 2.5 Evaluación final
- Genera tabla resumen de todas las iteraciones con métricas.
- Confirma la fiabilidad del modelo con intervalos de confianza y p-values.
- Redacta conclusiones orientadas a la problemática original.

---

## 3 · DASHBOARD INTERACTIVO (Dash + Plotly)
Crea `app.py` (o `dashboard/app.py`) siguiendo esta arquitectura:
dashboard/ ├── app.py # Punto de entrada: servidor Dash ├── layouts/ # Componentes de layout (header, tabs, footer) ├── callbacks/ # Callbacks organizados por sección ├── assets/ # CSS personalizado, imágenes, favicon ├── data/ # Datos procesados listos para el dashboard └── utils/ # Funciones auxiliares de carga y transformación

text


### 3.1 Contenido obligatorio del dashboard
| Sección               | Componentes                                                    |
|------------------------|----------------------------------------------------------------|
| **Resumen ejecutivo**  | KPI cards, descripción de la problemática, conclusión principal |
| **Análisis exploratorio** | Histogramas, boxplots, scatter matrix con filtros interactivos |
| **Contraste de hipótesis** | Resultados de tests estadísticos, p-values, intervalos de confianza |
| **Rendimiento del modelo** | Curvas ROC/PR, matriz de confusión, tabla de métricas por iteración |
| **Experimentos con hiperparámetros** | Parallel coordinates o heatmap de búsqueda |
| **Conclusiones y recomendaciones** | Texto narrativo + gráficos de soporte |

### 3.2 Interactividad
- Mínimo **3 callbacks** funcionales: filtros por variable, selector de modelo/iteración, toggle de métricas.
- Usa `dcc.Loading` para feedback visual durante cargas.
- Toda visualización debe tener título, ejes etiquetados y tooltips.

### 3.3 Estilo
- Paleta de colores consistente (define en `assets/style.css`).
- Tipografía legible (sans-serif, tamaño mínimo 14px para cuerpo).
- Espacio en blanco adecuado; no saturar la pantalla.
- Responsivo: `className="row"` o CSS Grid/Flexbox.

---

## 4 · DOCUMENTACIÓN
Genera los siguientes archivos:

### 4.1 `INFORME.md` (o `.pdf`)
Estructura:
1. Introducción y problemática.
2. Descripción de los datos.
3. Metodología (técnicas estadísticas, modelos, diseño de experimentos).
4. Resultados por iteración con justificación de decisiones.
5. Conclusiones y recomendaciones.
6. Referencias bibliográficas (incluir Oehlert 2010, Dash docs, Kohavi et al. 2008).

### 4.2 `README.md`
- Descripción del proyecto.
- Estructura del repositorio.
- Instrucciones de instalación y ejecución.
- Capturas de pantalla del dashboard (genera placeholders si no puedes capturar).

### 4.3 `requirements.txt`
- Fija versiones exactas (`dash==2.x.x`, `plotly==5.x.x`, `pandas`, `scikit-learn`, etc.).

### 4.4 `links.txt`
GitHub: https://github.com/<usuario>/<repo> Binder: https://mybinder.org/v2/gh/<usuario>/<repo>/HEAD?urlpath=...

text


### 4.5 Archivos Binder
- Crea carpeta `binder/` con `environment.yml` o `requirements.txt` y `postBuild` si es necesario.
- Asegura que `app.py` se pueda lanzar con `voila` o un script de entrada compatible con Binder.

---

## 5 · VALIDACIÓN FINAL (Checklist)
Antes de terminar, verifica:
- [ ] `pip install -r requirements.txt && python app.py` funciona sin errores.
- [ ] Todos los callbacks responden correctamente.
- [ ] No hay datos hardcodeados; todo se lee desde archivos en `data/`.
- [ ] El informe justifica cada decisión técnica.
- [ ] Las métricas incluyen intervalos de confianza.
- [ ] El repositorio tiene `.gitignore` adecuado (excluir `__pycache__`, `.env`, datos crudos pesados).
- [ ] La estructura de carpetas está limpia y documentada.

---

## RESTRICCIONES TÉCNICAS
- Python ≥ 3.9.
- No usar frameworks pesados innecesarios; prioriza librerías estándar de ciencia de datos.
- Código limpio: docstrings, type hints, PEP 8.
- Si un archivo existente ya resuelve algo, **reutilízalo**; no reescribas desde cero.
- Commits atómicos con mensajes descriptivos (en español).

## TONO Y FORMATO DE SALIDA
- Código comentado en **español**.
- Mensajes de commit en **español**.
- Dashboard y textos de usuario en **español**.
- Al finalizar cada fase, imprime un resumen de lo realizado y lo pendiente.

