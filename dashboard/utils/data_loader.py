from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from proyecto_ciencia_datos.config import build_project_paths


def load_dashboard_bundle(base_dir: Path | None = None) -> dict[str, object]:
    paths = build_project_paths(base_dir)
    with paths.summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    return {
        "dataset": pd.read_csv(paths.processed_dataset_path),
        "experiments": pd.read_csv(paths.experiments_path),
        "hypothesis_tests": pd.read_csv(paths.hypothesis_tests_path),
        "normality_checks": pd.read_csv(paths.normality_checks_path),
        "metric_confidence_intervals": pd.read_csv(
            paths.metric_confidence_intervals_path
        ),
        "roc_curve": pd.read_csv(paths.roc_curve_path),
        "pr_curve": pd.read_csv(paths.pr_curve_path),
        "confusion_matrix": pd.read_csv(paths.confusion_matrix_path),
        "feature_importance": pd.read_csv(paths.feature_importance_path),
        "predictions": pd.read_csv(paths.predictions_path),
        "summary": summary,
    }
