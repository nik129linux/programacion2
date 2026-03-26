# Dashboard de Ciencia de Datos Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un proyecto autocontenido de análisis y dashboard de clasificación de cáncer de mama.

**Architecture:** Un paquete analítico genera artefactos reproducibles en `data/` y `models/`. El dashboard consume exclusivamente esos archivos para presentar hallazgos y comparar experimentos.

**Tech Stack:** Python, pandas, SciPy, scikit-learn, Dash, Plotly, pytest.

---

### Task 1: Pipeline analítico

**Files:**
- Create: `src/proyecto_ciencia_datos/config.py`
- Create: `src/proyecto_ciencia_datos/dataset.py`
- Create: `src/proyecto_ciencia_datos/statistics.py`
- Create: `src/proyecto_ciencia_datos/modeling.py`
- Create: `src/proyecto_ciencia_datos/pipeline.py`
- Test: `tests/test_pipeline.py`

- [x] Escribir pruebas mínimas para artefactos y métricas.
- [x] Ejecutarlas en rojo.
- [x] Implementar carga de datos, features, contrastes, experimentos y serialización.
- [x] Ejecutar pruebas en verde.

### Task 2: Dashboard

**Files:**
- Create: `dashboard/app.py`
- Create: `dashboard/layouts/main_layout.py`
- Create: `dashboard/callbacks/main.py`
- Create: `dashboard/utils/data_loader.py`
- Create: `dashboard/assets/style.css`
- Create: `app.py`
- Test: `tests/test_dashboard.py`

- [x] Definir bundle de datos del dashboard.
- [x] Implementar layout con secciones obligatorias.
- [x] Registrar al menos 3 callbacks funcionales.
- [x] Verificar construcción de la app.

### Task 3: Documentación y despliegue

**Files:**
- Create: `README.md`
- Create: `INFORME.md`
- Create: `requirements.txt`
- Create: `links.txt`
- Create: `binder/requirements.txt`
- Create: `binder/postBuild`
- Create: `binder/start`
- Create: `.gitignore`

- [x] Documentar supuestos y resultados.
- [x] Fijar dependencias exactas.
- [x] Añadir compatibilidad básica con Binder.
