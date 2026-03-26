from __future__ import annotations

from dash import dcc, html

from dashboard.callbacks.main import (
    METRIC_TABLE,
    build_distribution_figures,
    build_experiment_heatmap,
    build_hypothesis_boxplot,
    build_model_comparison_rows,
    build_model_figures,
)


def create_layout(bundle: dict[str, object]):
    dataset = bundle["dataset"]
    experiments = bundle["experiments"]
    summary = bundle["summary"]
    feature_name = summary["default_feature"]
    experiment_id = summary["best_experiment_id"]

    histogram, boxplot, scatter_matrix = build_distribution_figures(
        dataset,
        feature_name,
        summary["scatter_features"],
    )
    roc_figure, pr_figure, confusion_figure, metric_rows = build_model_figures(
        bundle,
        experiment_id,
    )
    heatmap = build_experiment_heatmap(experiments, "roc_auc_test")
    thalach_boxplot = build_hypothesis_boxplot(dataset)

    hypothesis_rows = bundle["hypothesis_tests"].round(4).to_dict("records")
    normality_rows = bundle["normality_checks"].round(4).to_dict("records")
    comparison_rows = build_model_comparison_rows(
        experiments,
        summary["antecedent_logit_experiment_id"],
        experiment_id,
    )
    importance_rows = (
        bundle["feature_importance"]
        .loc[
            bundle["feature_importance"]["experiment_id"] == experiment_id,
            ["feature", "importance"],
        ]
        .head(10)
        .round(4)
        .to_dict("records")
    )

    return html.Div(
        className="page",
        children=[
            html.Header(
                className="hero",
                children=[
                    html.P("Actividad final de ciencia de datos", className="eyebrow"),
                    html.H1("Dashboard de riesgo de enfermedad cardíaca"),
                    html.P(summary["problem_statement"], className="hero-copy"),
                ],
            ),
            html.Section(
                className="kpi-grid",
                children=[
                    _build_kpi_card("Filas originales", str(summary["raw_n_observations"])),
                    _build_kpi_card(
                        "Filas únicas para modelado",
                        str(summary["modeling_n_observations"]),
                    ),
                    _build_kpi_card(
                        "Duplicados removidos",
                        str(summary["duplicate_rows_removed"]),
                    ),
                    _build_kpi_card(
                        "Prevalencia target=1",
                        f"{summary['positive_class_rate']:.3f}",
                    ),
                    _build_kpi_card("Mejor modelo", summary["best_model_name"]),
                    _build_kpi_card(
                        "Mejor ROC AUC",
                        f"{summary['best_roc_auc_test']:.3f}",
                    ),
                    _build_kpi_card(
                        "F1 final",
                        f"{summary['best_f1_test']:.3f}",
                    ),
                    _build_kpi_card(
                        "Accuracy baseline",
                        f"{summary['antecedent_logit_accuracy']:.3f}",
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.H2("Resumen ejecutivo"),
                    html.P(summary["main_conclusion"]),
                    html.Div(
                        className="chart-grid",
                        children=[
                            html.Div(
                                className="kpi-card",
                                children=[
                                    html.Span("Antecedente práctica2", className="kpi-label"),
                                    html.P(
                                        "El notebook previo mostró que las regresiones simples "
                                        "sobre `age` explicaban poco `chol` y `thalach`, así que "
                                        "no alcanzaban para clasificar riesgo."
                                    ),
                                ],
                            ),
                            html.Div(
                                className="kpi-card",
                                children=[
                                    html.Span("Antecedente tarea4", className="kpi-label"),
                                    html.P(
                                        "Se retoma el baseline logístico con `thalach + exang` "
                                        f"y accuracy {summary['antecedent_logit_accuracy']:.3f}, "
                                        "que ahora se mejora con más predictores y validación cruzada."
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.Div(
                        className="controls",
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Variable para exploración"),
                                    dcc.Dropdown(
                                        id="feature-dropdown",
                                        options=[
                                            {"label": column, "value": column}
                                            for column in dataset.columns
                                            if column not in {"target", "target_label"}
                                        ],
                                        value=feature_name,
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Variables del scatter matrix"),
                                    dcc.Dropdown(
                                        id="scatter-features-dropdown",
                                        options=[
                                            {"label": column, "value": column}
                                            for column in dataset.columns
                                            if column not in {"target", "target_label"}
                                        ],
                                        value=summary["scatter_features"],
                                        multi=True,
                                        clearable=False,
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.H2("Análisis exploratorio"),
                    html.Div(
                        className="chart-grid",
                        children=[
                            dcc.Loading(dcc.Graph(id="histogram-graph", figure=histogram)),
                            dcc.Loading(dcc.Graph(id="boxplot-graph", figure=boxplot)),
                        ],
                    ),
                    dcc.Loading(
                        dcc.Graph(id="scatter-matrix-graph", figure=scatter_matrix)
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.H2("Contraste de hipótesis"),
                    html.P(
                        "Se recupera el trabajo de `tarea4`: normalidad por grupo, "
                        "Mann-Whitney U sobre `thalach`, bootstrap del cambio de mediana "
                        "y chi-cuadrado entre `exang` y `target`."
                    ),
                    html.Div(
                        className="chart-grid",
                        children=[
                            dcc.Loading(
                                dcc.Graph(id="thalach-hypothesis-graph", figure=thalach_boxplot)
                            ),
                            METRIC_TABLE(
                                id="normality-table",
                                data=normality_rows,
                                columns=[
                                    {"name": column, "id": column}
                                    for column in bundle["normality_checks"].columns
                                ],
                                page_size=6,
                                style_table={"overflowX": "auto"},
                            ),
                        ],
                    ),
                    METRIC_TABLE(
                        id="hypothesis-table",
                        data=hypothesis_rows,
                        columns=[
                            {"name": column, "id": column}
                            for column in bundle["hypothesis_tests"].columns
                        ],
                        page_size=8,
                        style_table={"overflowX": "auto"},
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.H2("Antecedente lineal y comparación de modelos"),
                    html.P(
                        "La regresión lineal múltiple de `tarea4` explica `thalach` con "
                        f"R² {summary['antecedent_linear_regression']['r2_test']:.3f} "
                        f"y MSE {summary['antecedent_linear_regression']['mse_test']:.3f}. "
                        "Ese análisis se usa como antecedente explicativo, no como modelo final "
                        "de clasificación."
                    ),
                    METRIC_TABLE(
                        id="comparison-table",
                        data=comparison_rows,
                        columns=[
                            {"name": "Rol", "id": "rol"},
                            {"name": "Modelo", "id": "model_name"},
                            {"name": "Familia", "id": "experiment_family"},
                            {"name": "Accuracy", "id": "accuracy_test"},
                            {"name": "Precision", "id": "precision_test"},
                            {"name": "Recall", "id": "recall_test"},
                            {"name": "Specificity", "id": "specificity_test"},
                            {"name": "F1", "id": "f1_test"},
                            {"name": "ROC AUC", "id": "roc_auc_test"},
                            {"name": "PR AUC", "id": "pr_auc_test"},
                        ],
                        page_size=5,
                        style_table={"overflowX": "auto"},
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.Div(
                        className="controls",
                        children=[
                            html.Div(
                                children=[
                                    html.Label("Experimento / modelo"),
                                    dcc.Dropdown(
                                        id="experiment-dropdown",
                                        options=[
                                            {
                                                "label": (
                                                    f"{row['experiment_family']} | "
                                                    f"{row['config_label']}"
                                                ),
                                                "value": row["experiment_id"],
                                            }
                                            for _, row in experiments.iterrows()
                                        ],
                                        value=experiment_id,
                                        clearable=False,
                                    ),
                                ]
                            ),
                            html.Div(
                                children=[
                                    html.Label("Métrica comparativa"),
                                    dcc.Dropdown(
                                        id="metric-dropdown",
                                        options=[
                                            {
                                                "label": "ROC AUC test",
                                                "value": "roc_auc_test",
                                            },
                                            {
                                                "label": "Accuracy test",
                                                "value": "accuracy_test",
                                            },
                                            {"label": "F1 test", "value": "f1_test"},
                                            {"label": "PR AUC test", "value": "pr_auc_test"},
                                        ],
                                        value="roc_auc_test",
                                        clearable=False,
                                    ),
                                ]
                            ),
                        ],
                    ),
                    html.H2("Rendimiento del modelo"),
                    html.Div(
                        className="chart-grid",
                        children=[
                            dcc.Loading(dcc.Graph(id="roc-graph", figure=roc_figure)),
                            dcc.Loading(dcc.Graph(id="pr-graph", figure=pr_figure)),
                        ],
                    ),
                    html.Div(
                        className="chart-grid",
                        children=[
                            dcc.Loading(
                                dcc.Graph(
                                    id="confusion-graph",
                                    figure=confusion_figure,
                                )
                            ),
                            METRIC_TABLE(
                                id="metrics-table",
                                data=metric_rows,
                                columns=[
                                    {"name": "Métrica", "id": "Métrica"},
                                    {"name": "Valor", "id": "Valor"},
                                ],
                                page_size=10,
                            ),
                        ],
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.H2("Experimentos con hiperparámetros"),
                    dcc.Loading(
                        dcc.Graph(id="experiment-heatmap", figure=heatmap)
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.H2("Variables más influyentes"),
                    METRIC_TABLE(
                        id="importance-table",
                        data=importance_rows,
                        columns=[
                            {"name": "Variable", "id": "feature"},
                            {"name": "Importancia", "id": "importance"},
                        ],
                        page_size=10,
                    ),
                ],
            ),
            html.Section(
                className="section",
                children=[
                    html.H2("Conclusiones"),
                    html.Ul(
                        children=[
                            html.Li(summary["main_conclusion"]),
                            html.Li(
                                "El antecedente de `Práctica2` justificó abandonar la regresión "
                                "simple como enfoque principal."
                            ),
                            html.Li(
                                "El baseline de `tarea4` se reproduce y sirve como punto de "
                                "comparación formal frente al modelo multivariable final."
                            ),
                            html.Li(
                                "Modelar sobre 302 filas únicas evita inflar métricas y mantiene "
                                "consistencia con los contrastes de hipótesis del curso."
                            ),
                        ]
                    ),
                ],
            ),
        ],
    )


def _build_kpi_card(title: str, value: str) -> html.Div:
    return html.Div(
        className="kpi-card",
        children=[html.Span(title, className="kpi-label"), html.Strong(value)],
    )
