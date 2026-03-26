from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from proyecto_ciencia_datos.config import RANDOM_STATE


def confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> tuple[float, float]:
    sample = np.asarray(values, dtype=float)
    if sample.size == 0:
        return (np.nan, np.nan)
    if sample.size == 1 or np.isclose(np.std(sample, ddof=1), 0.0):
        mean_value = float(np.mean(sample))
        return (mean_value, mean_value)

    mean_value = float(np.mean(sample))
    standard_error = stats.sem(sample, nan_policy="omit")
    interval = stats.t.interval(
        confidence,
        df=sample.size - 1,
        loc=mean_value,
        scale=standard_error,
    )
    return (float(interval[0]), float(interval[1]))


def bootstrap_difference_interval(
    first_group: np.ndarray,
    second_group: np.ndarray,
    reducer: Callable[[np.ndarray], float],
    n_iterations: int = 1500,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    differences = []
    for _ in range(n_iterations):
        sampled_first = rng.choice(first_group, size=first_group.size, replace=True)
        sampled_second = rng.choice(second_group, size=second_group.size, replace=True)
        differences.append(float(reducer(sampled_first) - reducer(sampled_second)))

    point_estimate = float(reducer(first_group) - reducer(second_group))
    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    return (point_estimate, float(ci_low), float(ci_high))


def compute_simple_regression(
    frame: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> dict[str, float | str]:
    x_values = frame[x_column].to_numpy(dtype=float)
    y_values = frame[y_column].to_numpy(dtype=float)
    slope, intercept = np.polyfit(x_values, y_values, 1)
    predictions = slope * x_values + intercept
    residuals = y_values - predictions
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    sst = np.sum((y_values - np.mean(y_values)) ** 2)
    r_squared = float(1 - np.sum(residuals**2) / sst)
    correlation = float(np.corrcoef(x_values, y_values)[0, 1])

    return {
        "source": "practica2_notebook",
        "x_column": x_column,
        "y_column": y_column,
        "slope": float(slope),
        "intercept": float(intercept),
        "r_squared": r_squared,
        "correlation": correlation,
        "mse": mse,
        "mae": mae,
        "n_observations": int(frame.shape[0]),
    }


def run_hypothesis_tests(
    frame: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    group_zero = frame.loc[frame[target_column] == 0, "thalach"].to_numpy()
    group_one = frame.loc[frame[target_column] == 1, "thalach"].to_numpy()
    oldpeak_zero = frame.loc[frame[target_column] == 0, "oldpeak"].to_numpy()
    oldpeak_one = frame.loc[frame[target_column] == 1, "oldpeak"].to_numpy()

    normality_rows = []
    for target_value, sample in ((0, group_zero), (1, group_one)):
        normality_statistic, normality_p_value = stats.normaltest(sample)
        normality_rows.append(
            {
                "group": f"target = {target_value}",
                "estadistico_normalidad": float(normality_statistic),
                "p_valor_normalidad": float(normality_p_value),
            }
        )
    normality_checks = pd.DataFrame(normality_rows)

    mann_whitney_statistic, mann_whitney_p_value = stats.mannwhitneyu(
        group_zero,
        group_one,
        alternative="two-sided",
    )
    median_diff, ci_low, ci_high = bootstrap_difference_interval(
        group_one,
        group_zero,
        np.median,
    )

    contingency_table = pd.crosstab(frame["exang"], frame[target_column])
    chi_square_statistic, chi_square_p_value, _, _ = stats.chi2_contingency(
        contingency_table
    )
    oldpeak_statistic, oldpeak_p_value = stats.mannwhitneyu(
        oldpeak_zero,
        oldpeak_one,
        alternative="two-sided",
    )
    oldpeak_median_diff, oldpeak_ci_low, oldpeak_ci_high = bootstrap_difference_interval(
        oldpeak_one,
        oldpeak_zero,
        np.median,
    )
    cp_contingency_table = pd.crosstab(frame["cp"], frame[target_column])
    cp_chi_square_statistic, cp_chi_square_p_value, _, _ = stats.chi2_contingency(
        cp_contingency_table
    )

    hypothesis_tests = pd.DataFrame(
        [
            {
                "source": "tarea4_antecedente",
                "test_name": "mann_whitney_u",
                "feature": "thalach",
                "group_a": "target = 0",
                "group_b": "target = 1",
                "statistic": float(mann_whitney_statistic),
                "p_value": float(mann_whitney_p_value),
                "effect_value": median_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "significant": bool(mann_whitney_p_value < 0.05),
            },
            {
                "source": "tarea6_extension",
                "test_name": "mann_whitney_u",
                "feature": "oldpeak",
                "group_a": "target = 0",
                "group_b": "target = 1",
                "statistic": float(oldpeak_statistic),
                "p_value": float(oldpeak_p_value),
                "effect_value": oldpeak_median_diff,
                "ci_low": oldpeak_ci_low,
                "ci_high": oldpeak_ci_high,
                "significant": bool(oldpeak_p_value < 0.05),
            },
            {
                "source": "tarea6_extension",
                "test_name": "chi_square",
                "feature": "exang_vs_target",
                "group_a": "exang",
                "group_b": "target",
                "statistic": float(chi_square_statistic),
                "p_value": float(chi_square_p_value),
                "effect_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "significant": bool(chi_square_p_value < 0.05),
            },
            {
                "source": "tarea6_extension",
                "test_name": "chi_square",
                "feature": "cp_vs_target",
                "group_a": "cp",
                "group_b": "target",
                "statistic": float(cp_chi_square_statistic),
                "p_value": float(cp_chi_square_p_value),
                "effect_value": np.nan,
                "ci_low": np.nan,
                "ci_high": np.nan,
                "significant": bool(cp_chi_square_p_value < 0.05),
            },
        ]
    )

    summary = {
        "antecedent_mann_whitney_statistic": float(mann_whitney_statistic),
        "antecedent_mann_whitney_p_value": float(mann_whitney_p_value),
        "oldpeak_mann_whitney_statistic": float(oldpeak_statistic),
        "oldpeak_mann_whitney_p_value": float(oldpeak_p_value),
        "chi_square_exang_target_statistic": float(chi_square_statistic),
        "chi_square_exang_target_p_value": float(chi_square_p_value),
        "chi_square_cp_target_statistic": float(cp_chi_square_statistic),
        "chi_square_cp_target_p_value": float(cp_chi_square_p_value),
    }
    return hypothesis_tests, normality_checks, summary
