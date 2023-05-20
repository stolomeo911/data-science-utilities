from typing import Any

import mlflow
import pandas as pd
from data_preparation import prepare_dataset_for_train
from xgboost import XGBClassifier
from model import model_candidates


mlflow.set_tracking_uri("https://localhost:8000/")


def fit_survival_model(
    train_df: pd.DataFrame,
    target_column: str,
    column_models: list[str],
    column_dates: list[str],
    numerical_columns: list[str],
    categorical_columns: list[str],
    columns_to_drop: list[str],
    model_options: dict[str, Any],
    model
):

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
    models = model_candidates()
    train_model = models[model](**model_options)
    train_model.fit(target_df)
    return train_model
