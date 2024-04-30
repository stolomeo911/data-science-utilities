"""Monitor the evolution of metrics adopted in AB tests"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.proportion import proportions_chisquare
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.multitest import multipletests


def ab_test_check(input_data: pd.DataFrame,
                  metric: str,
                  group_name: str,
                  alpha: float,
                  metrics_dict: dict):
    """
    Perform hypothesis testing, deciding which kind of method to adopt depending on the type of KPI

:param input_data: input data to be used for test analysis. It should have a column specifying the test group
                       a customer belongs to, and different columns representing the KPIs we want to test.
    :type input_data: pd.DataFrame
    :param metric: name of the KPIs we want to analyze for hypothesis testing
    :type metric: str
    :param group_name: name of the column labelling the test group a customer belongs to
    :type group_name: str
    :param alpha: significance level
    :type alpha: float
    :param metrics_dict: a dictionary containing information about the KPIs under analysis
    :type metrics_dict: dict
    :return: a frame collecting all the basic information about the test, like baseline KPI value,
             treatment impact and its significance
    """
    if (alpha >= 1) or (alpha <= 0):
        raise ValueError('alpha parameter must be between 0 and 1')
    try:
        metric_info = metrics_dict[metric]
    except:
        raise Exception(f"{metric} is not available!")

    if metric_info['type'] == 'continuous':
        return t_test_check(input_data,
                            metric,
                            group_name,
                            alpha)
    elif metric_info['type'] == 'discrete':
        return proportion_test_check(input_data,
                                     metric,
                                     group_name,
                                     alpha)
    else:
        raise Exception(f"{metric} has not type. Check configuration!")


def proportion_test_check(input_data: pd.DataFrame,
                          metric: str,
                          group_name: str,
                          alpha: float):
    """
    Perform a chi2-test for proportions KPIs

    :param input_data: input data to be used for test analysis. It should have a column specifying the test group
                       a customer belongs to, and different columns representing the KPIs we want to test.
    :type input_data: pd.DataFrame
    :param metric: name of the KPIs we want to analyze for hypothesis testing
    :type metric: str
    :param group_name: name of the column labelling the test group a customer belongs to
    :type group_name: str
    :param alpha: significance level
    :type alpha: float
    :return: a frame collecting all the basic information about the test, like baseline KPI value,
             treatment impact and its significance
    """

    out_result = pd.DataFrame(columns=['impact',
                                       'impact_std_err',
                                       'uplift',
                                       'alpha',
                                       'pvalue',
                                       'significant'])

    relevant_cols = ['{}_numerator'.format(metric),
                     '{}_denominator'.format(metric)]

    input_data.rename(columns={'nr_numerator': f'{metric}_numerator', 'nr_denominator': f'{metric}_denominator'},
                      inplace=True)

    ctr_col = np.sort(pd.unique(input_data[group_name]))[0]
    treat_cols = np.sort(pd.unique(input_data[group_name]))[1:]

    ctr_grouped = input_data.loc[input_data[group_name] == ctr_col][relevant_cols].sum(axis=0)
    n_ctr = ctr_grouped.loc['{}_denominator'.format(metric)]
    p_ctr = ctr_grouped.loc['{}_numerator'.format(metric)] / n_ctr
    out_result.loc['ctr', 'impact'] = p_ctr
    out_result.loc['ctr', 'impact_std_err'] = np.sqrt(p_ctr * (1 - p_ctr) / n_ctr)
    out_result.loc['ctr', 'uplift'] = 0
    out_result.loc['ctr', 'pvalue'] = proportions_ztest(p_ctr * n_ctr, n_ctr, 0)[1]

    for col in treat_cols:
        treat_grouped = input_data.loc[input_data[group_name] == col][relevant_cols].sum(axis=0)
        n_treat = treat_grouped.loc['{}_denominator'.format(metric)]
        p_treat = treat_grouped.loc['{}_numerator'.format(metric)] / n_treat
        out_result.loc[col, 'impact'] = p_treat - p_ctr
        out_result.loc[col, 'uplift'] = (p_treat - p_ctr) / p_ctr
        out_result.loc[col, 'impact_std_err'] = np.sqrt(p_ctr * (1 - p_ctr) / n_ctr +
                                                        p_treat * (1 - p_treat) / n_treat)
        out_result.loc[col, 'pvalue'] = proportions_chisquare([p_ctr * n_ctr, p_treat * n_treat], [n_ctr, n_treat])[1]

    # apply correction for multiple tests
    # if len(treat_cols) > 1:
    #    pvals = out_result.loc[treat_cols, 'pvalue']
    #   out_result.loc[treat_cols, 'pvalue'] = multipletests(pvals, alpha)[1]

    out_result['alpha'] = alpha
    out_result['significant'] = 'No'
    out_result.loc[out_result['pvalue'] < alpha, 'significant'] = 'Yes'

    return out_result


def t_test_check(input_data: pd.DataFrame,
                 metric: str,
                 group_name: str,
                 alpha: float):
    """
    Perform a t-test for continuous KPIs

    :param input_data: input data to be used for test analysis. It should have a column specifying the test group
                       a customer belongs to, and different columns representing the KPIs we want to test.
    :type input_data: pd.DataFrame
    :param metric: name of the KPIs we want to analyze for hypothesis testing
    :type metric: str
    :param group_name: name of the column labelling the test group a customer belongs to
    :type group_name: str
    :param alpha: significance level
    :type alpha: float
    :return: a frame collecting all the basic information about the test, like baseline KPI value,
             treatment impact and its significance
    """

    y = input_data[metric].values

    X = pd.get_dummies(input_data[group_name], drop_first=True)
    X = sm.add_constant(X)

    mod = sm.OLS(y, X)
    res = mod.fit()

    out_result = pd.DataFrame(columns=['impact',
                                       'impact_std_err',
                                       'uplift',
                                       'alpha',
                                       'pvalue',
                                       'significant'])
    for idx, column in enumerate(np.sort(X.columns)):
        out_result.loc[column, 'impact'] = res.params[idx]
        out_result.loc[column, 'impact_std_err'] = res.bse[idx]
        out_result.loc[column, 'pvalue'] = res.pvalues[idx]

    out_result['alpha'] = alpha
    out_result['significant'] = 'No'
    out_result.loc[out_result['pvalue'] < out_result['alpha'], 'significant'] = 'Yes'

    out_result.index = ['ctr'] + out_result.index[1:].tolist()
    out_result.index.names = ['group type']

    base_val = out_result.loc['ctr', 'impact']
    out_result['uplift'] = out_result['impact'] / base_val
    out_result.loc['ctr', 'uplift'] = 0

    return out_result