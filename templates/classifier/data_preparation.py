import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def set_target_variable(df, target_column):
    df["target"] = df[target_column]
    return df


def set_columns_models(df, column_models):
    return df[column_models]


def set_dates_columns(df, columns_dates, extract_features=True):
    for column_date in columns_dates:
        df[column_date] = pd.to_datetime(df[column_date])
        if extract_features:
            df[column_date + '_month'] = df[column_date].dt.month
            df[column_date + '_day'] = df[column_date].dt.day
    return df


def set_categorical_variable(df, categorical_columns):
    for c in categorical_columns:
        onehot = OneHotEncoder(
            drop="first", sparse=False, handle_unknown="ignore"
        ).fit(df[[c]])
        encoded_column = pd.DataFrame(
            onehot.transform(df[[c]]),
            columns=onehot.get_feature_names_out(),
        ).astype(bool)
        df = pd.concat(
            (df, encoded_column),
            axis=1,
        ).drop(columns=[c])
    return df


def set_numerical_variable(df, numerical_columns):
    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column])
    return df
