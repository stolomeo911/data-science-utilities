from typing import Literal
# from .sample_sizing_t_test import calculate_t_test_sample_size
# from .sample_sizing_prop_test import calculate_prop_test_sample_size
import pandas as pd
from statsmodels.stats.power import tt_ind_solve_power
from math import sqrt
import scipy.stats as st
import warnings


# https://www.statsmodels.org/stable/generated/statsmodels.stats.power.tt_ind_solve_power.html


def calculate_sample_size(
        input_data: pd.DataFrame,
        alpha: float,
        beta: float,
        alternative: Literal["two-sided", "one-sided"],
        expected_uplift: float,
        metric: str,
        metric_type: str,
        n_variant: int = 1
):
    """
    This method allow the user to compute the minimum sample size for a t-test with 1 or more variants.
    :param alpha: significance of the test (float in [0,1])
    :param beta: type II error (float in [0,1])
    :param alternative: "two-sided" if we want to asses a bilateral test, "smaller" or "larger" if we want to assess a one-sided test
    :param metric: name of the metric to test
    :param expected_uplift: expected increase of the CR in the test group
    :param n_variant: number of variant we would like to test
    :param filters: dictionary with all the fields to filter when extracting the data
    """
    # cleaning metric string
    metric = metric.lower()

    # Parameters validation
    for variable_name in ["alpha", "beta"]:
        value = eval(variable_name)
        if (value < 0) or (value > 1):
            raise Exception(f"{variable_name} must be between 0 and 1!")

    # Bonferroni correction for multiple variants
    alpha = alpha / n_variant

    # Choose the test
    if metric_type == 'continuous':
        mean_baseline = input_data[metric].mean()
        sd_baseline = input_data[metric].std()
        one_week_pop = input_data[metric].count()
        if alternative == 'one-sided':
            if expected_uplift >= 0:
                alternative = 'larger'
            else:
                alternative = 'smaller'
        result = calculate_t_test_sample_size(mean_baseline, sd_baseline, alpha, beta, alternative, one_week_pop,
                                              expected_uplift, metric, n_variant)
    elif metric_type == 'discrete':
        num = input_data['nr_numerator'].sum()
        den = input_data['nr_denominator'].sum()
        P1 = num / den
        one_week_pop = input_data['nr_denominator'].sum()
        if alternative == 'one-sided':
            bilateral = 0
        else:
            bilateral = 1
        result = calculate_prop_test_sample_size(P1, bilateral, alpha, beta, one_week_pop, expected_uplift, metric,
                                                 n_variant)
    else:
        raise Exception(f"{metric} has not type. Check configuration!")
    return result


def calculate_t_test_sample_size(
        mean_baseline: float,
        sd_baseline: float,
        alpha: float,
        beta: float,
        alternative: Literal["two-sided", "smaller", "larger"],
        one_week_pop: int,
        expected_uplift: float,
        metric: str,
        n_variant: int = 1,
):
    """
    This method allow the user to compute the minimum sample size for a t-test with 1 or more variants.
    :param mean_baseline: observed mean
    :param sd_baseline: observed standard deviation
    :param alpha: significance of the test (float in [0,1])
    :param beta: type II error (float in [0,1])
    :param alternative: "two-sided" if we want to asses a bilateral test, "smaller" or "larger" if we want to assess a one-sided test
    :param one_week_pop: average number of users observed in a week
    :param expected_uplift: expected increase of the CR in the test group
    :param metric: name of the metric to test
    :param n_variant: number of variant we would like to test
    """
    # Parameters validation
    for variable_name in ["alpha", "beta"]:
        value = eval(variable_name)
        if (value < 0) or (value > 1):
            raise Exception(f"{variable_name} must be between 0 and 1!")

    # Bonferroni correction for multiple variants
    alpha = alpha / n_variant

    mean_treatment = mean_baseline * (1 + expected_uplift)
    mean_diff = mean_treatment - mean_baseline

    std_effect_size = mean_diff / sd_baseline

    results = pd.DataFrame(
        columns=["share test group", "share control group", "weeks", "total_sample", "size test group",
                 "size control group", f"{metric} test", f"{metric} control"]
    )
    for x in range(1, 11):
        s2 = x / 20  # small group, test
        s1 = 1 - s2  # big group, control
        ratio = s2 / s1

        n1 = tt_ind_solve_power(
            effect_size=std_effect_size,
            alpha=alpha,
            power=(1 - beta),
            ratio=ratio,
            alternative=alternative,
        )
        n2 = n1 * ratio

        weeks_run = (n1 + n2) / one_week_pop
        row = {
            "share test group": s2,
            "share control group": s1,
            "weeks": round(weeks_run, 1),
            "total_sample": int(n1 + n2),
            "size test group": int(n2),
            "size control group": int(n1),
            f"{metric} test": mean_treatment,
            f"{metric} control": mean_baseline
        }
        results = results.append(row, ignore_index=True)
    return results


def calculate_prop_test_sample_size(
        P1: float,
        bilateral: bool,
        alpha: float,
        beta: float,
        one_week_pop: int,
        expected_uplift: float,
        metric: str,
        n_variant: int = 1,
):
    """
    This method allow the user to compute the minimum sample size for an proportion test with 1 or more variants.
    :param P1: baseline conversion rate (float in [0,1])
    :param bilateral: true if we want to estimate sample size for a two-sided test
    :param alpha: significance of the test (float in [0,1])
    :param beta: type II error (float in [0,1])
    :param one_week_pop: average number of users observed in a week
    :param expected_uplift: expected increase of the CR in the test group
    :param metric: name of the metric to test
    :param n_variant: number of variant we would like to test
    """
    # Parameters validation
    if (P1 < 0) or (P1 > 1):
        raise Exception(f"P1 must be between 0 and 1!")

    P2 = max(min(P1 * (1 + expected_uplift), 1), 0)  # setting a cap above to 100% conversion and a cap below to 0%

    if P2 == 1:
        warnings.warn(
            f"WARNING: the CR for test group was capped at 100%. The maximum expected uplift is {P2 / P1 - 1:.4f}"
        )

    if bilateral:
        xa = st.norm.ppf(1 - alpha / 2)
    else:
        xa = st.norm.ppf(1 - alpha)
    xb = st.norm.ppf(beta)

    # Sample size estimation
    results = pd.DataFrame(
        columns=["share test group", "share control group", "weeks", "total_sample", "size test group",
                 "size control group", f"{metric} test", f"{metric} control"]
    )
    for x in range(1, 11):
        s2 = x / 20  # small group, test
        s1 = 1 - s2  # big group, control
        sqrt_arg1 = P1 * (1 - P1) / s1 + P2 * (1 - P2) / s2
        sqrt_arg2 = (s1 * P1 + s2 * P2) * (1 - s1 * P1 - s2 * P2) / (s1 * s2)

        n = ((sqrt(sqrt_arg1) * xb + sqrt(sqrt_arg2) * xa) ** 2) / (P1 - P2) ** 2

        weeks_run = n / one_week_pop
        row = {
            "share test group": s2,
            "share control group": s1,
            "weeks": round(weeks_run, 1),
            "total_sample": round(n),
            "size test group": round(n * s2),
            "size control group": round(n * s1),
            f"{metric} test": P2,
            f"{metric} control": P1,
        }
        results = results.append(row, ignore_index=True)
    return results