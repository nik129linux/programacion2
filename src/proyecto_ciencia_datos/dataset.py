from __future__ import annotations

from pathlib import Path

import pandas as pd

from proyecto_ciencia_datos.config import (
    SOURCE_HEART_DATASET_PATH,
    TARGET_COLUMN,
    TARGET_LABEL_MAP,
)


def resolve_source_dataset_path(local_raw_copy: Path | None = None) -> Path:
    if local_raw_copy and local_raw_copy.exists():
        return local_raw_copy
    if SOURCE_HEART_DATASET_PATH.exists():
        return SOURCE_HEART_DATASET_PATH
    raise FileNotFoundError(
        f"No se encontró heart.csv en {SOURCE_HEART_DATASET_PATH}"
    )


def load_source_dataset(local_raw_copy: Path | None = None) -> pd.DataFrame:
    source_path = resolve_source_dataset_path(local_raw_copy)
    return pd.read_csv(source_path)


def build_modeling_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    deduplicated = frame.drop_duplicates().reset_index(drop=True).copy()
    deduplicated["target_label"] = deduplicated[TARGET_COLUMN].map(TARGET_LABEL_MAP)

    # DECISIÓN: estas variables derivadas siguen el hilo clínico del notebook
    # previo y del baseline de tarea4; no se inventan features ajenas al dominio.
    deduplicated["chol_age_ratio"] = deduplicated["chol"] / deduplicated["age"]
    deduplicated["pressure_oldpeak_interaction"] = (
        deduplicated["trestbps"] * deduplicated["oldpeak"]
    )
    deduplicated["maxhr_age_gap"] = 220 - deduplicated["age"] - deduplicated["thalach"]
    return deduplicated


def summarize_data_quality(
    raw_frame: pd.DataFrame,
    modeling_frame: pd.DataFrame,
) -> dict[str, object]:
    duplicate_rows = int(raw_frame.duplicated().sum())
    target_distribution = (
        modeling_frame[TARGET_COLUMN]
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )
    return {
        "source_dataset_path": str(SOURCE_HEART_DATASET_PATH),
        "raw_n_observations": int(raw_frame.shape[0]),
        "raw_n_columns": int(raw_frame.shape[1]),
        "modeling_n_observations": int(modeling_frame.shape[0]),
        "duplicate_rows_removed": duplicate_rows,
        "null_values_total": int(raw_frame.isna().sum().sum()),
        "positive_class_rate": float(target_distribution.get(1, 0.0)),
        "target_distribution_pct": {
            str(key): round(float(value) * 100, 2)
            for key, value in target_distribution.items()
        },
    }
