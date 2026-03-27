from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


RANDOM_STATE = 42
TARGET_COLUMN = "target"
POSITIVE_LABEL = 1
TARGET_LABEL_MAP = {
    0: "Sin enfermedad cardíaca",
    1: "Con enfermedad cardíaca",
}
DEFAULT_FEATURE = "thalach"
DEFAULT_SCATTER_FEATURES = ["age", "thalach", "oldpeak"]

BASELINE_LOGIT_FEATURES = ["thalach", "exang"]
LINEAR_BASELINE_FEATURES = ["age", "trestbps", "chol", "oldpeak"]
CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
ENGINEERED_FEATURES = [
    "chol_age_ratio",
    "pressure_oldpeak_interaction",
    "maxhr_age_gap",
]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]
FULL_MODEL_FEATURES = (
    CONTINUOUS_FEATURES + ENGINEERED_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES
)


@dataclass(frozen=True)
class ProjectPaths:
    base_dir: Path
    data_dir: Path
    raw_dir: Path
    processed_dir: Path
    dashboard_dir: Path
    models_dir: Path
    raw_dataset_path: Path
    processed_dataset_path: Path
    experiments_path: Path
    hypothesis_tests_path: Path
    normality_checks_path: Path
    metric_confidence_intervals_path: Path
    roc_curve_path: Path
    pr_curve_path: Path
    confusion_matrix_path: Path
    feature_importance_path: Path
    predictions_path: Path
    summary_path: Path
    best_model_path: Path


def build_project_paths(base_dir: Path | None = None) -> ProjectPaths:
    root_dir = base_dir or Path(__file__).resolve().parents[2]
    data_dir = root_dir / "data"
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    dashboard_dir = data_dir / "dashboard"
    models_dir = root_dir / "models"

    return ProjectPaths(
        base_dir=root_dir,
        data_dir=data_dir,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        dashboard_dir=dashboard_dir,
        models_dir=models_dir,
        raw_dataset_path=raw_dir / "heart.csv",
        processed_dataset_path=processed_dir / "heart_modelado.csv",
        experiments_path=dashboard_dir / "experiments.csv",
        hypothesis_tests_path=dashboard_dir / "hypothesis_tests.csv",
        normality_checks_path=dashboard_dir / "normality_checks.csv",
        metric_confidence_intervals_path=dashboard_dir
        / "metric_confidence_intervals.csv",
        roc_curve_path=dashboard_dir / "roc_curve.csv",
        pr_curve_path=dashboard_dir / "pr_curve.csv",
        confusion_matrix_path=dashboard_dir / "confusion_matrix.csv",
        feature_importance_path=dashboard_dir / "feature_importance.csv",
        predictions_path=dashboard_dir / "test_predictions.csv",
        summary_path=dashboard_dir / "summary.json",
        best_model_path=models_dir / "best_model.joblib",
    )


def ensure_project_directories(paths: ProjectPaths) -> None:
    for directory in (
        paths.raw_dir,
        paths.processed_dir,
        paths.dashboard_dir,
        paths.models_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
