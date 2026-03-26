from __future__ import annotations

from pathlib import Path

from dashboard.app import create_dash_app
from dashboard.callbacks.main import (
    build_distribution_figures,
    build_experiment_heatmap,
    build_model_figures,
)
from dashboard.utils.data_loader import load_dashboard_bundle
from proyecto_ciencia_datos.pipeline import run_full_analysis


def test_dashboard_bundle_contains_expected_heart_sections(tmp_path: Path) -> None:
    run_full_analysis(base_dir=tmp_path)

    bundle = load_dashboard_bundle(base_dir=tmp_path)

    assert {
        "dataset",
        "experiments",
        "hypothesis_tests",
        "normality_checks",
        "metric_confidence_intervals",
        "summary",
    }.issubset(bundle.keys())
    assert not bundle["dataset"].empty
    assert not bundle["experiments"].empty
    assert bundle["summary"]["default_feature"] == "thalach"
    assert bundle["summary"]["duplicate_rows_removed"] == 723


def test_dashboard_helpers_return_labeled_figures_for_heart_project(
    tmp_path: Path,
) -> None:
    run_full_analysis(base_dir=tmp_path)
    bundle = load_dashboard_bundle(base_dir=tmp_path)

    feature_name = bundle["summary"]["default_feature"]
    experiment_id = bundle["summary"]["best_experiment_id"]

    histogram, boxplot, scatter_matrix = build_distribution_figures(
        bundle["dataset"],
        feature_name,
        bundle["summary"]["scatter_features"],
    )
    heatmap = build_experiment_heatmap(bundle["experiments"], "roc_auc_test")
    roc_figure, pr_figure, confusion_figure, metric_rows = build_model_figures(
        bundle,
        experiment_id,
    )

    assert "thalach" in histogram.layout.title.text.lower()
    assert boxplot.layout.yaxis.title.text
    assert scatter_matrix.layout.title.text
    assert heatmap.layout.title.text
    assert roc_figure.layout.xaxis.title.text
    assert pr_figure.layout.yaxis.title.text
    assert confusion_figure.layout.title.text
    assert metric_rows


def test_create_dash_app_registers_callbacks(tmp_path: Path) -> None:
    run_full_analysis(base_dir=tmp_path)

    app = create_dash_app(base_dir=tmp_path)

    assert len(app.callback_map) >= 3
