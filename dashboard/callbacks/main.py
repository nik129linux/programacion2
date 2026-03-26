from __future__ import annotations

import pandas as pd
import plotly.express as px
from dash import Input, Output, dash_table


COLOR_SEQUENCE = ["#12355b", "#3ea5a1", "#f95738", "#f4d35e", "#0d3b66"]
TARGET_LABEL_COLUMN = "target_label"
TARGET_NUMERIC_COLUMN = "target"
TARGET_LABEL_NAME = "Estado clínico"


def build_distribution_figures(
    dataset: pd.DataFrame,
    feature_name: str,
    scatter_features: list[str],
):
    histogram = px.histogram(
        dataset,
        x=feature_name,
        color=TARGET_LABEL_COLUMN,
        barmode="overlay",
        color_discrete_sequence=COLOR_SEQUENCE[:2],
        title=f"Distribución de {feature_name}",
        labels={feature_name: feature_name, TARGET_LABEL_COLUMN: TARGET_LABEL_NAME},
    )
    histogram.update_layout(legend_title_text=TARGET_LABEL_NAME)

    boxplot = px.box(
        dataset,
        x=TARGET_LABEL_COLUMN,
        y=feature_name,
        color=TARGET_LABEL_COLUMN,
        color_discrete_sequence=COLOR_SEQUENCE[:2],
        title=f"Boxplot de {feature_name} por estado clínico",
        labels={TARGET_LABEL_COLUMN: TARGET_LABEL_NAME, feature_name: feature_name},
    )

    scatter_matrix = px.scatter_matrix(
        dataset,
        dimensions=scatter_features,
        color=TARGET_LABEL_COLUMN,
        title="Scatter matrix de variables cardiometabólicas",
        labels={feature: feature for feature in scatter_features},
        color_discrete_sequence=COLOR_SEQUENCE[:2],
    )
    scatter_matrix.update_traces(diagonal_visible=False, showupperhalf=False)
    return histogram, boxplot, scatter_matrix


def build_hypothesis_boxplot(dataset: pd.DataFrame):
    figure = px.box(
        dataset,
        x=TARGET_LABEL_COLUMN,
        y="thalach",
        color=TARGET_LABEL_COLUMN,
        color_discrete_sequence=COLOR_SEQUENCE[:2],
        title="Frecuencia cardíaca máxima (`thalach`) por presencia de enfermedad",
        labels={TARGET_LABEL_COLUMN: TARGET_LABEL_NAME, "thalach": "thalach"},
        points="all",
    )
    figure.update_layout(showlegend=False)
    return figure


def build_experiment_heatmap(
    experiments: pd.DataFrame,
    metric_name: str,
):
    base_metric = metric_name.removesuffix("_test")
    labels = experiments.apply(
        lambda row: f"{row['experiment_family']} | {row['config_label']}",
        axis=1,
    )
    values = pd.DataFrame(
        {
            "Experimento": labels,
            "Train": experiments.get(f"{base_metric}_train", experiments[metric_name]),
            "Validación cruzada": experiments.get(
                f"cv_{base_metric}_mean",
                experiments[metric_name],
            ),
            "Test": experiments[metric_name],
        }
    ).set_index("Experimento")

    heatmap = px.imshow(
        values,
        text_auto=".3f",
        aspect="auto",
        color_continuous_scale="YlOrBr",
        title=f"Comparación de {metric_name} por configuración",
        labels={"x": "Etapa", "y": "Configuración", "color": metric_name},
    )
    return heatmap


def build_model_figures(bundle: dict[str, object], experiment_id: str):
    roc_data = bundle["roc_curve"]
    pr_data = bundle["pr_curve"]
    confusion_data = bundle["confusion_matrix"]
    experiments = bundle["experiments"]
    intervals = bundle["metric_confidence_intervals"]

    selected_roc = roc_data.loc[roc_data["experiment_id"] == experiment_id]
    selected_pr = pr_data.loc[pr_data["experiment_id"] == experiment_id]
    selected_confusion = confusion_data.loc[
        confusion_data["experiment_id"] == experiment_id
    ]
    selected_experiment = experiments.loc[
        experiments["experiment_id"] == experiment_id
    ].iloc[0]

    roc_figure = px.line(
        selected_roc,
        x="fpr",
        y="tpr",
        title="Curva ROC",
        labels={"fpr": "Tasa de falsos positivos", "tpr": "Tasa de verdaderos positivos"},
    )
    roc_figure.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"dash": "dash", "color": "#999"},
    )

    pr_figure = px.line(
        selected_pr,
        x="recall",
        y="precision",
        title="Curva Precision-Recall",
        labels={"recall": "Recall", "precision": "Precision"},
    )

    confusion_matrix = selected_confusion.pivot(
        index="real",
        columns="predicho",
        values="valor",
    )
    confusion_figure = px.imshow(
        confusion_matrix,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Matriz de confusión",
        labels={"x": "Predicho", "y": "Real", "color": "Casos"},
    )

    metric_rows = [
        {"Métrica": "Familia", "Valor": selected_experiment["experiment_family"]},
        {"Métrica": "Configuración", "Valor": selected_experiment["config_label"]},
        {"Métrica": "ROC AUC test", "Valor": round(selected_experiment["roc_auc_test"], 4)},
        {"Métrica": "Accuracy test", "Valor": round(selected_experiment["accuracy_test"], 4)},
        {"Métrica": "Precision test", "Valor": round(selected_experiment["precision_test"], 4)},
        {"Métrica": "Recall test", "Valor": round(selected_experiment["recall_test"], 4)},
        {"Métrica": "Specificity test", "Valor": round(selected_experiment["specificity_test"], 4)},
        {"Métrica": "F1 test", "Valor": round(selected_experiment["f1_test"], 4)},
        {"Métrica": "PR AUC test", "Valor": round(selected_experiment["pr_auc_test"], 4)},
        {
            "Métrica": "ROC AUC validación cruzada",
            "Valor": round(selected_experiment["cv_roc_auc_mean"], 4),
        },
    ]

    if experiment_id == bundle["summary"]["best_experiment_id"]:
        for row in intervals.to_dict("records"):
            metric_rows.append(
                {
                    "Métrica": f"{row['metric']} IC 95%",
                    "Valor": f"{row['ci_low']:.3f} a {row['ci_high']:.3f}",
                }
            )

    return roc_figure, pr_figure, confusion_figure, metric_rows


def build_model_comparison_rows(
    experiments: pd.DataFrame,
    baseline_experiment_id: str,
    best_experiment_id: str,
):
    selected = experiments.loc[
        experiments["experiment_id"].isin([baseline_experiment_id, best_experiment_id]),
        [
            "experiment_id",
            "model_name",
            "experiment_family",
            "accuracy_test",
            "precision_test",
            "recall_test",
            "specificity_test",
            "f1_test",
            "roc_auc_test",
            "pr_auc_test",
        ],
    ].copy()
    selected["rol"] = selected["experiment_id"].map(
        {
            baseline_experiment_id: "Baseline tarea4",
            best_experiment_id: "Modelo final",
        }
    )
    selected = selected.drop(columns=["experiment_id"])
    metric_columns = [
        "accuracy_test",
        "precision_test",
        "recall_test",
        "specificity_test",
        "f1_test",
        "roc_auc_test",
        "pr_auc_test",
    ]
    selected[metric_columns] = selected[metric_columns].round(4)
    return selected.to_dict("records")


def register_callbacks(app, bundle: dict[str, object]) -> None:
    @app.callback(
        Output("histogram-graph", "figure"),
        Output("boxplot-graph", "figure"),
        Output("scatter-matrix-graph", "figure"),
        Input("feature-dropdown", "value"),
        Input("scatter-features-dropdown", "value"),
    )
    def update_distribution_section(feature_name: str, scatter_features: list[str]):
        return build_distribution_figures(
            bundle["dataset"],
            feature_name,
            scatter_features,
        )

    @app.callback(
        Output("roc-graph", "figure"),
        Output("pr-graph", "figure"),
        Output("confusion-graph", "figure"),
        Output("metrics-table", "data"),
        Output("metrics-table", "columns"),
        Input("experiment-dropdown", "value"),
    )
    def update_model_section(experiment_id: str):
        roc_figure, pr_figure, confusion_figure, metric_rows = build_model_figures(
            bundle,
            experiment_id,
        )
        return (
            roc_figure,
            pr_figure,
            confusion_figure,
            metric_rows,
            [
                {"name": "Métrica", "id": "Métrica"},
                {"name": "Valor", "id": "Valor"},
            ],
        )

    @app.callback(
        Output("experiment-heatmap", "figure"),
        Input("metric-dropdown", "value"),
    )
    def update_experiment_heatmap(metric_name: str):
        return build_experiment_heatmap(bundle["experiments"], metric_name)


METRIC_TABLE = dash_table.DataTable
