import os
import pandas as pd
import logging

import src.constants.files as files
import src.constants.columns as c
import src.constants.models as m
from sklearn.metrics import f1_score


def evaluate():
    model_name = m.LOGISTIC_REG_MODEL_NAME.replace('.joblib', '')

    prediction_df = pd.read_csv(os.path.join(files.OUTPUT_DATA, f"{model_name}_{files.PREDICTIONS_TEST}"))

    logging.info(f"Evaluating {model_name}")

    y_test = prediction_df[c.Loans.target()].values
    y_pred = prediction_df["prediction"].values

    score = round(f1_score(y_test, y_pred, pos_label="Y"), 2)

    logging.info(f"F1 score for model {model_name} is {score}")
