import pandas as pd
import logging

import src.constants.columns as c
import src.constants.models as m
from sklearn.metrics import f1_score


def evaluate(prediction_file_path):
    """
    Evaluate the performance of the model on test data.

    :param prediction_file_path: path to prediction data.

    :return: None
    """
    # TODO: compute f1 score

    score = round(f1_score(y_test, y_pred, pos_label="Y"), 2)

    logging.info(f"F1 score for model {m.LOGISTIC_REG_MODEL_NAME} is {score}")
