import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def set_target_variable(df, target_column):
    df["target"] = df[target_column]
    return df


def set_columns_models(df, column_models):
    return df[column_models]


def set_dates_columns(df, columns_dates, extract_features=True, drop_original_column=True):
    for column_date in columns_dates:
        df[column_date] = pd.to_datetime(df[column_date])
        if extract_features:
            df[column_date + '_month'] = df[column_date].dt.month
            df[column_date + '_day'] = df[column_date].dt.day
        if drop_original_column:
            df = df.drop(columns=[column_date])
    return df


def set_categorical_variable(df, categorical_columns):
    for c in categorical_columns:
        onehot = OneHotEncoder(
            drop="first", sparse=False, handle_unknown="ignore"
        ).fit(df[[c]])
        encoded_column = pd.DataFrame(
            onehot.transform(df[[c]]),
            columns=onehot.get_feature_names_out(),
        ).astype(int)
        df = pd.concat(
            (df, encoded_column),
            axis=1,
        ).drop(columns=[c])
    return df


def set_numerical_variable(df, numerical_columns):
    for column, impute_strategy in numerical_columns.items():
        df[column] = pd.to_numeric(df[column])
        if impute_strategy == '0':
            df[column] = df[column].fillna(0)
        elif impute_strategy == 'mean':
            df[column] = df[column].fillna(df[column].mean())
        elif impute_strategy == 'drop':
            df[column] = df[column].dropna()
        else:
            raise NotImplementedError
    return df


def prepare_dataset_for_train(df, config):

    target_column = config["target_column"]
    column_models = config["model_columns"]
    column_dates = config["columns_dates"]
    categorical_columns = config["categorical_columns"]
    numerical_columns = config["numerical_columns"]

    df = set_target_variable(df, target_column)
    df = set_dates_columns(df, column_dates)
    df = set_categorical_variable(df, categorical_columns)
    df = set_numerical_variable(df, numerical_columns)
    df = set_columns_models(df, column_models + ["is_train"])
    return df
