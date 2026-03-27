from __future__ import annotations

import json
from math import isclose
from pathlib import Path

import pandas as pd
import pytest

from proyecto_ciencia_datos.dataset import resolve_source_dataset_path
from proyecto_ciencia_datos.pipeline import run_full_analysis
from proyecto_ciencia_datos.statistics import confidence_interval


def test_confidence_interval_contains_sample_mean() -> None:
    values = [0.72, 0.76, 0.79, 0.81, 0.84]

    lower, upper = confidence_interval(values)

    assert lower < upper
    assert lower <= sum(values) / len(values) <= upper


def test_resolve_source_dataset_path_uses_project_relative_heart_csv(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "data" / "raw" / "heart.csv"
    dataset_path.parent.mkdir(parents=True)
    dataset_path.write_text("age,target\n60,1\n", encoding="utf-8")

    resolved = resolve_source_dataset_path(project_root=tmp_path)

    assert resolved == dataset_path


def test_resolve_source_dataset_path_raises_clear_error_when_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="Dataset heart.csv no encontrado en data/raw/. Coloque el archivo ahí antes de ejecutar.",
    ):
        resolve_source_dataset_path(project_root=tmp_path)


def test_run_full_analysis_generates_required_heart_artifacts(tmp_path: Path) -> None:
    artifacts = run_full_analysis(base_dir=tmp_path)

    required_keys = {
        "raw_dataset",
        "processed_dataset",
        "experiments",
        "hypothesis_tests",
        "normality_checks",
        "metric_confidence_intervals",
        "model",
        "summary",
    }

    assert required_keys.issubset(artifacts.keys())
    for key in required_keys:
        assert Path(artifacts[key]).exists(), key

    raw_dataset = pd.read_csv(artifacts["raw_dataset"])
    processed_dataset = pd.read_csv(artifacts["processed_dataset"])
    experiments = pd.read_csv(artifacts["experiments"])
    hypothesis_tests = pd.read_csv(artifacts["hypothesis_tests"])
    normality_checks = pd.read_csv(artifacts["normality_checks"])

    assert raw_dataset.shape == (1025, 14)
    assert processed_dataset.shape[0] == 302
    assert "target" in processed_dataset.columns
    assert "target_label" in processed_dataset.columns
    assert {
        "chol_age_ratio",
        "pressure_oldpeak_interaction",
        "maxhr_age_gap",
    }.issubset(processed_dataset.columns)

    assert {
        "experiment_id",
        "experiment_family",
        "is_best_config",
        "model_name",
        "accuracy_test",
        "roc_auc_test",
        "cv_roc_auc_mean",
        "timestamp",
    }.issubset(experiments.columns)
    assert {
        "baseline_tarea4",
        "logit_regularizado",
        "random_forest",
    }.issubset(set(experiments["experiment_family"]))

    baseline_row = experiments.loc[
        (experiments["experiment_family"] == "baseline_tarea4")
        & (experiments["is_best_config"])
    ].iloc[0]
    assert isclose(float(baseline_row["accuracy_test"]), 0.7582, rel_tol=0.02)

    mann_whitney_row = hypothesis_tests.loc[
        (hypothesis_tests["test_name"] == "mann_whitney_u")
        & (hypothesis_tests["feature"] == "thalach")
    ].iloc[0]
    assert float(mann_whitney_row["p_value"]) < 1e-10

    assert {"group", "estadistico_normalidad", "p_valor_normalidad"}.issubset(
        normality_checks.columns
    )

    with Path(artifacts["summary"]).open(encoding="utf-8") as handle:
        summary = json.load(handle)

    assert summary["raw_n_observations"] == 1025
    assert summary["modeling_n_observations"] == 302
    assert summary["duplicate_rows_removed"] == 723
    assert summary["default_feature"] == "thalach"
    assert isclose(float(summary["antecedent_logit_accuracy"]), 0.7582, rel_tol=0.02)
    assert float(summary["antecedent_mann_whitney_p_value"]) < 1e-10
    assert summary["best_model_name"]
