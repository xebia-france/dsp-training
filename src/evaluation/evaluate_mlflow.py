import os

import mlflow
from joblib import load
import pandas as pd
import logging

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import src.constants.files as files
import src.constants.columns as c
from src.constants import models
from sklearn.metrics import f1_score


def evaluate_mlflow():
    test_df = pd.read_csv(os.path.join(files.INTERIM_DATA, files.TEST))
    current_run_id = mlflow.active_run().info.run_id

    logging.info("Loading preprocessing pipeline")

    preprocessing_pipeline, pp_run_id = load_latest_preprocessing_pipeline(
        models.PREPROCESSING_PIPELINE, files.MLFLOW_EXPERIMENT_NAME)

    if pp_run_id != current_run_id:
        logging.info("Using preprocessing pipeline saved in a previous run")
        mlflow.log_param("preprocessing_pipeline_run_id", pp_run_id)
    preprocessed_test = preprocessing_pipeline.transform(test_df)
    y_test = test_df[c.Loans.Loan_Status].values

    logging.info("Loading trained model")
    logistic_reg = mlflow.sklearn.load_model(f"runs:/{str(current_run_id)}/{models.MODEL_NAME}")

    logging.info("Computing performance metrics")
    y_pred = logistic_reg.predict(preprocessed_test)

    logging.info("Saving predictions")
    test_with_predictions_df = test_df.copy()
    test_with_predictions_df["Predicted_Loan_Status"] = y_pred
    test_with_predictions_df.to_csv(os.path.join(files.OUTPUT_DATA, files.TEST_WITH_PREDICTIONS), index=False)

    score = round(f1_score(y_test, y_pred, pos_label="Y"), 2)

    mlflow.log_metric("f1_score", score)


def load_latest_preprocessing_pipeline(preprocessing_pipeline_name, experiment_name):
    experiment_id = get_experiment_id(experiment_name)

    runs = MlflowClient().search_runs(
        experiment_ids=str(experiment_id),
        run_view_type=ViewType.ACTIVE_ONLY,
        order_by=["tag.start_time DESC"]
    )

    for run in runs:
        run_id = run.info.run_id
        artifact_uri = run.info.artifact_uri
        try:
            return mlflow.sklearn.load_model(f"{artifact_uri}/{preprocessing_pipeline_name}"), run_id
        except:
            continue

    raise Exception(
        f"Could not find a preprocessing pipeline named {preprocessing_pipeline_name} in current and previous runs"
    )


def get_experiment_id(experiment_name):
    for experiment in MlflowClient().list_experiments():
        if experiment.name == experiment_name:
            return experiment.experiment_id
    raise Exception(f"Could not find and experiment id for experiment {experiment_name}")
