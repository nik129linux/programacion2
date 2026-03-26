from __future__ import annotations

import json
import sys
from pathlib import Path

from dash import Dash

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dashboard.callbacks.main import register_callbacks
from dashboard.layouts.main_layout import create_layout
from dashboard.utils.data_loader import load_dashboard_bundle
from proyecto_ciencia_datos.config import build_project_paths
from proyecto_ciencia_datos.pipeline import run_full_analysis


def _artifacts_are_current(paths) -> bool:
    required_paths = [
        paths.summary_path,
        paths.processed_dataset_path,
        paths.experiments_path,
        paths.hypothesis_tests_path,
        paths.normality_checks_path,
    ]
    if not all(path.exists() for path in required_paths):
        return False

    with paths.summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)

    required_summary_keys = {
        "default_feature",
        "best_experiment_id",
        "positive_class_rate",
        "antecedent_logit_experiment_id",
        "antecedent_linear_regression",
    }
    return required_summary_keys.issubset(summary)


def create_dash_app(base_dir: Path | None = None) -> Dash:
    paths = build_project_paths(base_dir)
    if not _artifacts_are_current(paths):
        run_full_analysis(base_dir)

    bundle = load_dashboard_bundle(base_dir)
    app = Dash(__name__, title="Dashboard de Enfermedad Cardíaca")
    app.layout = create_layout(bundle)
    register_callbacks(app, bundle)
    return app


app = create_dash_app()
server = app.server
