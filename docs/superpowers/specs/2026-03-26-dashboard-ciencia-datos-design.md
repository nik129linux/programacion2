# Diseño del Dashboard de Ciencia de Datos

## Contexto

El repositorio no incluía proyecto previo, así que la solución debía ser autocontenida y reproducible sin depender de archivos externos.

## Decisiones de diseño

- Dataset base: `load_breast_cancer` de `scikit-learn`.
- Arquitectura: paquete analítico en `src/`, dashboard en `dashboard/`, artefactos exportados a `data/`.
- Modelos: baseline con sobreajuste controlado, regresión logística regularizada y bosque aleatorio ajustado.
- Visualización: Dash + Plotly con filtros de variable, experimento y métrica.

## Salidas esperadas

- Dataset crudo y procesado.
- Experimentos comparables con métricas e intervalos de confianza.
- Dashboard interactivo.
- Documentación y archivos de despliegue.
