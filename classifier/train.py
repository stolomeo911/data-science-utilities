from typing import Any

import mlflow
import pandas as pd
from data_preparation import prepare_dataset_for_train
from xgboost import XGBClassifier


def fit_survival_model(
    train_df: pd.DataFrame,
    target_column: str,
    column_models: list[str],
    column_dates: list[str],
    numerical_columns: list[str],
    categorical_columns: list[str],
    columns_to_drop: list[str],
    model_options: dict[str, Any],
) -> XGBClassifier:
    onehot_encoders = prepare_dataset_for_train(train_df, target_column,
                                                column_models, column_dates,
                                                categorical_columns, numerical_columns)
    target_df = (
        train_df,
        numerical_columns,
        categorical_columns,
        onehot_encoders,
        columns_to_drop,
    )
    model = XGBClassifier(**model_options)
    model.fit(target_df)
    return model
