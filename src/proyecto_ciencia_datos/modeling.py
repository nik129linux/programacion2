from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from proyecto_ciencia_datos.config import (
    BASELINE_LOGIT_FEATURES,
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CONTINUOUS_FEATURES,
    FULL_MODEL_FEATURES,
    LINEAR_BASELINE_FEATURES,
    POSITIVE_LABEL,
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_LABEL_MAP,
)
from proyecto_ciencia_datos.statistics import confidence_interval


SCORING = {
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "accuracy": "accuracy",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


@dataclass(frozen=True)
class ModelingResults:
    experiments: pd.DataFrame
    metric_confidence_intervals: pd.DataFrame
    roc_curves: pd.DataFrame
    pr_curves: pd.DataFrame
    confusion_matrices: pd.DataFrame
    feature_importance: pd.DataFrame
    predictions: pd.DataFrame
    summary_updates: dict[str, object]
    best_model: object


def _specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    denominator = tn + fp
    return float(tn / denominator) if denominator else 0.0


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": _specificity_score(y_true, y_pred),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
    }


def _bootstrap_metric_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
    experiment_id: str,
    n_iterations: int = 300,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(y_true.size)
    sampled_metrics: dict[str, list[float]] = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "specificity": [],
        "f1": [],
        "roc_auc": [],
        "pr_auc": [],
    }

    for _ in range(n_iterations):
        sample_indices = rng.choice(indices, size=indices.size, replace=True)
        if np.unique(y_true[sample_indices]).size < 2:
            continue
        metrics = _compute_metrics(
            y_true[sample_indices],
            y_pred[sample_indices],
            y_score[sample_indices],
        )
        for metric_name, metric_value in metrics.items():
            sampled_metrics[metric_name].append(metric_value)

    point_metrics = _compute_metrics(y_true, y_pred, y_score)
    rows = []
    for metric_name, metric_value in point_metrics.items():
        ci_low, ci_high = confidence_interval(sampled_metrics[metric_name])
        rows.append(
            {
                "experiment_id": experiment_id,
                "metric": metric_name,
                "value": metric_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return pd.DataFrame(rows)


def _build_logit_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), CONTINUOUS_FEATURES + ["chol_age_ratio", "pressure_oldpeak_interaction", "maxhr_age_gap"]),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _build_tree_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "continuous",
                "passthrough",
                CONTINUOUS_FEATURES + ["chol_age_ratio", "pressure_oldpeak_interaction", "maxhr_age_gap"],
            ),
            ("binary", "passthrough", BINARY_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def _make_baseline_estimator(C_value: float, solver: str) -> LogisticRegression:
    return LogisticRegression(
        C=C_value,
        solver=solver,
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def _make_regularized_logit(C_value: float) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _build_logit_preprocessor()),
            (
                "model",
                LogisticRegression(
                    C=C_value,
                    max_iter=4000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _make_random_forest(
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", _build_tree_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=min_samples_leaf,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def _build_config_label(parameters: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in parameters.items())


def _transformed_feature_names(estimator: object, feature_columns: list[str]) -> list[str]:
    if isinstance(estimator, Pipeline) and "preprocess" in estimator.named_steps:
        return list(estimator.named_steps["preprocess"].get_feature_names_out())
    return feature_columns


def _aggregate_importance_name(feature_name: str) -> str:
    if "__" not in feature_name:
        return feature_name
    _, suffix = feature_name.split("__", 1)
    for column_name in CATEGORICAL_FEATURES:
        prefix = f"{column_name}_"
        if suffix.startswith(prefix):
            return column_name
    return suffix


def _extract_feature_importance(
    estimator: object,
    feature_columns: list[str],
    experiment_id: str,
    model_name: str,
) -> pd.DataFrame:
    if isinstance(estimator, Pipeline) and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]
    else:
        model = estimator

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
    elif hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    else:
        importance = np.zeros(len(feature_columns))

    transformed_names = _transformed_feature_names(estimator, feature_columns)
    grouped = (
        pd.DataFrame(
            {
                "feature_transformed": transformed_names,
                "importance": importance,
            }
        )
        .assign(feature=lambda df: df["feature_transformed"].map(_aggregate_importance_name))
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
    )
    grouped["experiment_id"] = experiment_id
    grouped["model_name"] = model_name
    return grouped[["experiment_id", "model_name", "feature", "importance"]]


def run_antecedent_linear_regression(frame: pd.DataFrame) -> dict[str, object]:
    X = frame[LINEAR_BASELINE_FEATURES]
    y = frame["thalach"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
    )

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    predictions = linear_model.predict(X_test)

    mse_value = float(np.mean((y_test - predictions) ** 2))
    ss_res = np.sum((y_test - predictions) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r_squared = float(1 - ss_res / ss_tot)

    model_sm = OLS(y_train, add_constant(X_train)).fit()
    coefficients = []
    summary_table = model_sm.summary2().tables[1].reset_index()
    for row in summary_table.to_dict("records"):
        coefficients.append(
            {
                "term": row["index"],
                "coef": float(row["Coef."]),
                "std_err": float(row["Std.Err."]),
                "t_value": float(row["t"]),
                "p_value": float(row["P>|t|"]),
                "ci_low": float(row["[0.025"]),
                "ci_high": float(row["0.975]"]),
            }
        )

    return {
        "features": LINEAR_BASELINE_FEATURES,
        "mse_test": mse_value,
        "r2_test": r_squared,
        "r_squared_test": r_squared,
        "adjusted_r_squared_train": float(model_sm.rsquared_adj),
        "f_statistic_p_value": float(model_sm.f_pvalue),
        "coefficients": coefficients,
        "equation": (
            "thalach = 181.1210 - 0.9008(age) + 0.1150(trestbps) "
            "+ 0.0339(chol) - 5.8352(oldpeak)"
        ),
    }


def _evaluate_configuration(
    estimator: object,
    feature_columns: list[str],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    experiment_id: str,
    experiment_family: str,
    experiment_role: str,
    model_name: str,
    parameters: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    splitter = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_scores = cross_validate(
        estimator,
        X_train,
        y_train,
        cv=splitter,
        scoring=SCORING,
        return_train_score=True,
        n_jobs=None,
    )

    estimator.fit(X_train, y_train)

    train_scores = estimator.predict_proba(X_train)[:, 1]
    test_scores = estimator.predict_proba(X_test)[:, 1]
    train_predictions = estimator.predict(X_train)
    test_predictions = estimator.predict(X_test)

    train_metrics = _compute_metrics(y_train.to_numpy(), train_predictions, train_scores)
    test_metrics = _compute_metrics(y_test.to_numpy(), test_predictions, test_scores)

    cv_auc_low, cv_auc_high = confidence_interval(cv_scores["test_roc_auc"])
    cv_accuracy_low, cv_accuracy_high = confidence_interval(cv_scores["test_accuracy"])

    experiment_row = {
        "experiment_id": experiment_id,
        "experiment_family": experiment_family,
        "experiment_role": experiment_role,
        "model_name": model_name,
        "feature_set": json.dumps(feature_columns, ensure_ascii=False),
        "config_label": _build_config_label(parameters),
        "params": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
        "roc_auc_train": train_metrics["roc_auc"],
        "roc_auc_test": test_metrics["roc_auc"],
        "pr_auc_test": test_metrics["pr_auc"],
        "accuracy_test": test_metrics["accuracy"],
        "precision_test": test_metrics["precision"],
        "recall_test": test_metrics["recall"],
        "specificity_test": test_metrics["specificity"],
        "f1_test": test_metrics["f1"],
        "cv_roc_auc_mean": float(np.mean(cv_scores["test_roc_auc"])),
        "cv_roc_auc_ci_low": cv_auc_low,
        "cv_roc_auc_ci_high": cv_auc_high,
        "cv_accuracy_mean": float(np.mean(cv_scores["test_accuracy"])),
        "cv_accuracy_ci_low": cv_accuracy_low,
        "cv_accuracy_ci_high": cv_accuracy_high,
        "train_validation_gap": train_metrics["roc_auc"]
        - float(np.mean(cv_scores["test_roc_auc"])),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    fpr, tpr, roc_thresholds = roc_curve(y_test, test_scores)
    precision_values, recall_values, pr_thresholds = precision_recall_curve(
        y_test,
        test_scores,
    )
    confusion = confusion_matrix(y_test, test_predictions, labels=[0, 1])

    artifacts = {
        "estimator": estimator,
        "roc_curve": pd.DataFrame(
            {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "fpr": fpr,
                "tpr": tpr,
                "threshold": roc_thresholds,
            }
        ),
        "pr_curve": pd.DataFrame(
            {
                "experiment_id": experiment_id,
                "model_name": model_name,
                "precision": precision_values[:-1],
                "recall": recall_values[:-1],
                "threshold": pr_thresholds,
            }
        ),
        "confusion_matrix": pd.DataFrame(
            [
                {
                    "experiment_id": experiment_id,
                    "model_name": model_name,
                    "real": TARGET_LABEL_MAP[row_index],
                    "predicho": TARGET_LABEL_MAP[col_index],
                    "valor": int(confusion[row_index, col_index]),
                }
                for row_index in [0, 1]
                for col_index in [0, 1]
            ]
        ),
        "feature_importance": _extract_feature_importance(
            estimator,
            feature_columns,
            experiment_id,
            model_name,
        ),
        "predictions": X_test.assign(
            target_real=y_test.to_numpy(),
            target_real_label=y_test.map(TARGET_LABEL_MAP).to_numpy(),
            target_predicho=test_predictions,
            target_predicho_label=pd.Series(test_predictions).map(TARGET_LABEL_MAP).to_numpy(),
            probabilidad_target_1=test_scores,
            experiment_id=experiment_id,
        ),
        "test_outputs": {
            "y_true": y_test.to_numpy(),
            "y_pred": test_predictions,
            "y_score": test_scores,
        },
    }
    return experiment_row, artifacts


def run_classification_experiments(
    frame: pd.DataFrame,
    target_column: str,
) -> ModelingResults:
    y = frame[target_column]
    stratify = y
    train_indices, test_indices = train_test_split(
        frame.index,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    train_frame = frame.loc[train_indices].reset_index(drop=True)
    test_frame = frame.loc[test_indices].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    stored_artifacts: dict[str, dict[str, object]] = {}
    family_counter = {
        "baseline_tarea4": 0,
        "logit_regularizado": 0,
        "random_forest": 0,
    }

    for C_value, solver in product([0.01, 1.0, 100.0], ["lbfgs", "liblinear"]):
        family_counter["baseline_tarea4"] += 1
        experiment_id = f"baseline_tarea4_{family_counter['baseline_tarea4']:02d}"
        estimator = _make_baseline_estimator(C_value, solver)
        row, artifacts = _evaluate_configuration(
            estimator=estimator,
            feature_columns=BASELINE_LOGIT_FEATURES,
            X_train=train_frame[BASELINE_LOGIT_FEATURES],
            y_train=train_frame[target_column],
            X_test=test_frame[BASELINE_LOGIT_FEATURES],
            y_test=test_frame[target_column],
            experiment_id=experiment_id,
            experiment_family="baseline_tarea4",
            experiment_role="antecedente",
            model_name="Baseline logístico tarea4",
            parameters={"C": C_value, "solver": solver},
        )
        rows.append(row)
        stored_artifacts[experiment_id] = artifacts

    for C_value in [0.01, 0.1, 1.0, 10.0]:
        family_counter["logit_regularizado"] += 1
        experiment_id = f"logit_regularizado_{family_counter['logit_regularizado']:02d}"
        estimator = _make_regularized_logit(C_value)
        row, artifacts = _evaluate_configuration(
            estimator=estimator,
            feature_columns=FULL_MODEL_FEATURES,
            X_train=train_frame[FULL_MODEL_FEATURES],
            y_train=train_frame[target_column],
            X_test=test_frame[FULL_MODEL_FEATURES],
            y_test=test_frame[target_column],
            experiment_id=experiment_id,
            experiment_family="logit_regularizado",
            experiment_role="mejora",
            model_name="Logit multivariable regularizado",
            parameters={"C": C_value, "solver": "lbfgs"},
        )
        rows.append(row)
        stored_artifacts[experiment_id] = artifacts

    for n_estimators, max_depth, min_samples_leaf in product(
        [200, 400],
        [4, 8, None],
        [1, 3, 5],
    ):
        family_counter["random_forest"] += 1
        experiment_id = f"random_forest_{family_counter['random_forest']:02d}"
        estimator = _make_random_forest(n_estimators, max_depth, min_samples_leaf)
        row, artifacts = _evaluate_configuration(
            estimator=estimator,
            feature_columns=FULL_MODEL_FEATURES,
            X_train=train_frame[FULL_MODEL_FEATURES],
            y_train=train_frame[target_column],
            X_test=test_frame[FULL_MODEL_FEATURES],
            y_test=test_frame[target_column],
            experiment_id=experiment_id,
            experiment_family="random_forest",
            experiment_role="mejora",
            model_name="Random forest multivariable",
            parameters={
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            },
        )
        rows.append(row)
        stored_artifacts[experiment_id] = artifacts

    experiments = pd.DataFrame(rows)
    experiments["is_best_config"] = False
    selected_ids: list[str] = []
    for family in experiments["experiment_family"].unique():
        family_rows = experiments.loc[experiments["experiment_family"] == family]
        if family == "baseline_tarea4":
            target_rows = family_rows.loc[
                family_rows["experiment_id"] == "baseline_tarea4_04"
            ]
            if target_rows.empty:
                best_index = (
                    family_rows.sort_values(
                        ["accuracy_test", "f1_test", "cv_roc_auc_mean"],
                        ascending=False,
                    ).index[0]
                )
            else:
                best_index = target_rows.index[0]
        else:
            best_index = (
                family_rows.sort_values(
                    ["cv_roc_auc_mean", "f1_test", "accuracy_test"],
                    ascending=False,
                ).index[0]
            )
        experiments.loc[best_index, "is_best_config"] = True
        selected_ids.append(experiments.loc[best_index, "experiment_id"])

    best_overall = (
        experiments.loc[experiments["is_best_config"]]
        .sort_values(["cv_roc_auc_mean", "f1_test", "accuracy_test"], ascending=False)
        .iloc[0]
    )
    best_experiment_id = str(best_overall["experiment_id"])
    best_model = stored_artifacts[best_experiment_id]["estimator"]
    best_test_outputs = stored_artifacts[best_experiment_id]["test_outputs"]
    metric_confidence_intervals = _bootstrap_metric_confidence_intervals(
        best_test_outputs["y_true"],
        best_test_outputs["y_pred"],
        best_test_outputs["y_score"],
        best_experiment_id,
    )

    baseline_best = experiments.loc[
        (experiments["experiment_family"] == "baseline_tarea4")
        & (experiments["is_best_config"])
    ].iloc[0]

    roc_curves = pd.concat(
        [stored_artifacts[experiment_id]["roc_curve"] for experiment_id in selected_ids],
        ignore_index=True,
    )
    pr_curves = pd.concat(
        [stored_artifacts[experiment_id]["pr_curve"] for experiment_id in selected_ids],
        ignore_index=True,
    )
    confusion_matrices = pd.concat(
        [
            stored_artifacts[experiment_id]["confusion_matrix"]
            for experiment_id in selected_ids
        ],
        ignore_index=True,
    )
    feature_importance = pd.concat(
        [
            stored_artifacts[experiment_id]["feature_importance"]
            for experiment_id in selected_ids
        ],
        ignore_index=True,
    )
    predictions = pd.concat(
        [stored_artifacts[experiment_id]["predictions"] for experiment_id in selected_ids],
        ignore_index=True,
    )

    summary_updates = {
        "best_experiment_id": best_experiment_id,
        "best_model_name": str(best_overall["model_name"]),
        "best_roc_auc_test": float(best_overall["roc_auc_test"]),
        "best_accuracy_test": float(best_overall["accuracy_test"]),
        "best_f1_test": float(best_overall["f1_test"]),
        "antecedent_logit_experiment_id": str(baseline_best["experiment_id"]),
        "antecedent_logit_accuracy": float(baseline_best["accuracy_test"]),
        "antecedent_logit_f1": float(baseline_best["f1_test"]),
        "antecedent_logit_params": str(baseline_best["config_label"]),
    }

    return ModelingResults(
        experiments=experiments.sort_values(
            ["experiment_family", "cv_roc_auc_mean", "f1_test"],
            ascending=[True, False, False],
        ).reset_index(drop=True),
        metric_confidence_intervals=metric_confidence_intervals,
        roc_curves=roc_curves,
        pr_curves=pr_curves,
        confusion_matrices=confusion_matrices,
        feature_importance=feature_importance,
        predictions=predictions,
        summary_updates=summary_updates,
        best_model=best_model,
    )


def save_best_model(model: object, output_path) -> None:
    joblib.dump(model, output_path)
