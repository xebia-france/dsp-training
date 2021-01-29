import os

import mlflow
from joblib import load
import pandas as pd
import logging

import src.constants.files as files
import src.constants.columns as c
from src.constants import models
from sklearn.metrics import f1_score


def evaluate_mlflow():
    test_df = pd.read_csv(os.path.join(files.INTERIM_DATA, files.TEST))
    run_id = mlflow.active_run().info.run_id

    logging.info("Loading preprocessing pipeline")
    preprocessing_pipeline = mlflow.sklearn.load_model(f"runs:/{str(run_id)}/{models.PREPROCESSING_PIPELINE}")
    preprocessed_test = preprocessing_pipeline.transform(test_df)
    y_test = test_df[c.Loans.Loan_Status].values

    logging.info("Loading trained model")
    logistic_reg = mlflow.sklearn.load_model(f"runs:/{str(run_id)}/{models.MODEL_NAME}")

    logging.info("Computing performance metrics")
    y_pred = logistic_reg.predict(preprocessed_test)

    score = round(f1_score(y_test, y_pred, pos_label="Y"), 2)

    mlflow.log_metric("f1_score", score)

