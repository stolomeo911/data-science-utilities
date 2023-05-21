from typing import Any

import mlflow
import pandas as pd
from data_preparation import prepare_dataset_for_train
from xgboost import XGBClassifier
from model import model_candidates
from metrics import get_classification_report


def fit_model(
    df: pd.DataFrame,
    model: str,
    model_options=None,
    output_results=True
):
    train_df = df[df["is_train"] == 'train']
    X_train = train_df.drop(columns=["is_train", 'target'])
    y_train = train_df['target']
    test_df = df[df["is_train"] == 'test']
    x_test = test_df.drop(columns=["is_train", 'target'])
    y_test = test_df['target']
    models = model_candidates()
    if model_options:
        train_model = models[model](**model_options)
    else:
        train_model = models[model]
    train_model.fit(X_train, y_train)
    if output_results:
        preds = train_model.predict(x_test)
        results = get_classification_report(y_test, preds)
        return train_model, results
    else:
        return train_model


def model_predict(train_model, X):
    prediction = train_model.predict(X)


