from pathlib import Path

import pandas as pd
import typer
import mlflow
import json

from train import fit_model
from localdb.data_import import sql_to_df
from data_preparation import prepare_dataset_for_train
import yaml
from yaml import Loader
import pickle
from fastapi import FastAPI, HTTPException

PATH_SQL_QUERY = '/go/sql/dataset_classifier.sql'

RAW_DATASET_PATH = Path(__file__).parent / "artifacts" / "raw_dataset.parquet"
PREPROCESSED_DATASET_PATH = (
    Path(__file__).parent / "artifacts" / "preprocessed_dataset.parquet"
)

CONFIG_PATH = Path(__file__).parent.parent.parent / "inventory_params.yaml"
DEFAULT_SAVE_PATH = Path(__file__).parent / "artifacts" / "model.joblib"


#app = typer.Typer()


mlflow.set_tracking_uri("file:///tmp/my_tracking")
#mlflow.sklearn.autolog()


def setup_config():
    dict_settings = yaml.load(
        open("C:/Users/S.Tolomeo/Downloads/personal/data-science-utilities/classifier/data/params.yml"), Loader=Loader
    )
    return dict_settings


#@app.command()
def generate_dataset(path=typer.Option(RAW_DATASET_PATH)) -> None:
    df = sql_to_df(PATH_SQL_QUERY)
    df.to_parquet(RAW_DATASET_PATH)
    print("Reading configuration")
    #config = dataset_config(dvc.api.params_show("params.yaml")["dataset"])
    config = setup_config()
    print("Preprocessing dataset...")
    df = prepare_dataset_for_train(df, config)
    print("Saving preprocessed dataset")
    df.to_parquet(PREPROCESSED_DATASET_PATH)
    return df

#@app.command()
def train(
    dataset_path: str = typer.Option(PREPROCESSED_DATASET_PATH),
    model_path: str = typer.Option(DEFAULT_SAVE_PATH),
) -> None:
    print("Reading database")
    df = sql_to_df(PATH_SQL_QUERY)
    df.to_parquet(RAW_DATASET_PATH)
    print("Reading configuration")
    #config = dataset_config(dvc.api.params_show("params.yaml")["dataset"])
    config = setup_config()
    print("Preprocessing dataset...")
    df = prepare_dataset_for_train(df, config)
    print("Saving preprocessed dataset")
    df.to_parquet(PREPROCESSED_DATASET_PATH)

    mlflow.set_experiment(f"classifier-model/train")
    mlflow.log_params(config)

    print("Fitting model")
    model, results = fit_model(df, config["model"])

    for key in ['0', '1']:
        for metric, value in results[key].items():
           mlflow.log_metric(metric + '_' + key, value)

    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    mlflow.log_artifact('model.pkl')
    print("Model calibrated")
    #model.save(model_path)
    # results logging
    #save_coefficients_plot(model)
    #handle_metrics(dataset, model)


def run_predictions():
    df = pd.read_parquet(PREPROCESSED_DATASET_PATH)
    model = pickle.load(open('model.pkl', 'rb'))
    preds = model.predict(df.drop(columns=["is_train", "target"]))
    response = {"pred_" + str(x): int(preds[x]) for x in range(len(preds))}
    response = json.dumps(response)
    return response



app = FastAPI(title="Classifier API")


@app.post("/predict/", status_code=200)
async def predict():
    prediction = run_predictions()
    return prediction


#train()
run_predictions()