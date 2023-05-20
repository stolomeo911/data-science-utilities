import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor


def drop_column_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    a = df.isnull().sum() / len(df) * 100
    df_clean = df.drop(columns=a[a > 20].keys())
    return df_clean


def calc_iv(df: pd.DataFrame, feature: str, target: str, bins=5, pr=False) -> pd.DataFrame:
    """
    Set pr=True to enable printing of output.

    Output:
      * iv: float,
      * data: pandas.DataFrame
    """

    lst = []

    # df[feature] = df[feature].fillna("NULL")

    for i in range(1, bins + 1):
        upper_bound = df[feature].quantile(i / bins)
        lower_bound = df[feature].quantile((i - 1) / bins)
        lst.append([feature,  # Variable
                    str(round(lower_bound, 2)) + "-" + str(round(upper_bound)),  # Value
                    df[(df[feature] < upper_bound) & (df[feature] >= lower_bound)].count()[feature],  # All
                    df[(df[feature] < upper_bound) & (df[feature] >= lower_bound) & (df[target] == 0)].count()[feature],
                    # Good (think: Fraud == 0)
                    df[(df[feature] < upper_bound) & (df[feature] >= lower_bound) & (df[target] == 1)].count()[
                        feature]])  # Bad (think: Fraud == 1)

    data = pd.DataFrame(lst, columns=['Variable', 'Value', 'All', 'Good', 'Bad'])

    data['Share'] = data['All'] / data['All'].sum()
    data['Bad Rate'] = data['Bad'] / data['All']
    data['Distribution Good'] = (data['All'] - data['Bad']) / (data['All'].sum() - data['Bad'].sum())
    data['Distribution Bad'] = data['Bad'] / data['Bad'].sum()
    data['WoE'] = np.log(data['Distribution Good'] / data['Distribution Bad'])

    data = data.replace({'WoE': {np.inf: 0, -np.inf: 0}})

    data['IV'] = data['WoE'] * (data['Distribution Good'] - data['Distribution Bad'])

    data = data.sort_values(by=['Variable', 'Value'], ascending=[True, True])
    data.index = range(len(data.index))

    if pr:
        print(data)
        print('IV = ', data['IV'].sum())

    iv = data['IV'].sum()
    # print(iv)

    return iv, data


def select_features_from_iv(df, columns_iv, plot=True, threshold=0.02):
    df_iv = pd.DataFrame(columns=["Variable", "IV"])
    for column in columns_iv:
        iv, data = calc_iv(df, column, "target")
        df_iv = df_iv.append({"Variable": column, "IV": iv}, ignore_index=True)
        df_iv = df_iv.sort_values("IV", ascending=False)
    if plot:
        df_iv.set_index("Variable").plot.barh(figsize=(20, 50), fontsize=16)
    feature_selected = df_iv[df_iv["IV"] > threshold].Variable
    return list(feature_selected.values)


def remove_highly_correlated_variables(df, threshold=0.7, plot=True):
    # Create correlation matrix
    corr_matrix = df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape),
                                      k=1).astype(np.bool_))

    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    df_clean = df.drop(columns=to_drop)

    if plot:
        fig, ax = plt.subplots(figsize=(20, 20))  # Sample figsize in inches

        sns.heatmap(df.corr(), xticklabels=df.corr().columns,
                    yticklabels=df.corr().columns, annot=True, ax=ax)
    return df_clean


def calculate_vif_(df, needed_features=[], thresh=5.0):
    variables = list(range(df.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(df.iloc[:, variables].values, ix)
               for ix in range(df.iloc[:, variables].shape[1])]
        maxloc = vif.index(max(vif))
        if max(vif) > thresh and df.iloc[:, variables].columns[maxloc] not in needed_features:
            print('dropping \'' + df.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(df.columns[variables])
    return df.iloc[:, variables]




