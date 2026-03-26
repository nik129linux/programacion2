from __future__ import annotations

import json
from pathlib import Path

from proyecto_ciencia_datos.config import (
    DEFAULT_FEATURE,
    DEFAULT_SCATTER_FEATURES,
    TARGET_COLUMN,
    build_project_paths,
    ensure_project_directories,
)
from proyecto_ciencia_datos.dataset import (
    build_modeling_dataset,
    load_source_dataset,
    summarize_data_quality,
)
from proyecto_ciencia_datos.modeling import (
    run_antecedent_linear_regression,
    run_classification_experiments,
    save_best_model,
)
from proyecto_ciencia_datos.statistics import compute_simple_regression, run_hypothesis_tests


def run_full_analysis(base_dir: Path | None = None) -> dict[str, str]:
    paths = build_project_paths(base_dir)
    ensure_project_directories(paths)

    raw_dataset = load_source_dataset(paths.raw_dataset_path)
    raw_dataset.to_csv(paths.raw_dataset_path, index=False)

    processed_dataset = build_modeling_dataset(raw_dataset)
    processed_dataset.to_csv(paths.processed_dataset_path, index=False)

    notebook_regressions = [
        compute_simple_regression(raw_dataset, "age", "chol"),
        compute_simple_regression(raw_dataset, "age", "thalach"),
    ]
    linear_baseline = run_antecedent_linear_regression(processed_dataset)
    hypothesis_tests, normality_checks, hypothesis_summary = run_hypothesis_tests(
        processed_dataset,
        TARGET_COLUMN,
    )
    modeling_results = run_classification_experiments(processed_dataset, TARGET_COLUMN)

    hypothesis_tests.to_csv(paths.hypothesis_tests_path, index=False)
    normality_checks.to_csv(paths.normality_checks_path, index=False)
    modeling_results.experiments.to_csv(paths.experiments_path, index=False)
    modeling_results.metric_confidence_intervals.to_csv(
        paths.metric_confidence_intervals_path,
        index=False,
    )
    modeling_results.roc_curves.to_csv(paths.roc_curve_path, index=False)
    modeling_results.pr_curves.to_csv(paths.pr_curve_path, index=False)
    modeling_results.confusion_matrices.to_csv(
        paths.confusion_matrix_path,
        index=False,
    )
    modeling_results.feature_importance.to_csv(
        paths.feature_importance_path,
        index=False,
    )
    modeling_results.predictions.to_csv(paths.predictions_path, index=False)

    summary = summarize_data_quality(raw_dataset, processed_dataset)
    summary.update(
        {
            "problem_statement": (
                "Identificar factores clínicos y demográficos asociados con "
                "la presencia de enfermedad cardíaca y mejorar el baseline del curso."
            ),
            "default_feature": DEFAULT_FEATURE,
            "scatter_features": DEFAULT_SCATTER_FEATURES,
            "predictor_count": len(processed_dataset.columns) - 2,
            "antecedent_notebook_regressions": notebook_regressions,
            "antecedent_linear_regression": linear_baseline,
            "main_conclusion": (
                "El baseline de tarea4 con thalach y exang es útil como punto "
                "de partida, pero el modelo multivariable mejora la calidad "
                "predictiva sin perder interpretación clínica."
            ),
        }
    )
    summary.update(hypothesis_summary)
    summary.update(modeling_results.summary_updates)

    with paths.summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    save_best_model(modeling_results.best_model, paths.best_model_path)

    return {
        "raw_dataset": str(paths.raw_dataset_path),
        "processed_dataset": str(paths.processed_dataset_path),
        "experiments": str(paths.experiments_path),
        "hypothesis_tests": str(paths.hypothesis_tests_path),
        "normality_checks": str(paths.normality_checks_path),
        "metric_confidence_intervals": str(paths.metric_confidence_intervals_path),
        "model": str(paths.best_model_path),
        "summary": str(paths.summary_path),
    }
