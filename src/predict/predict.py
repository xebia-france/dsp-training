import os
from joblib import load
import pandas as pd
import logging

import src.constants.files as files
import src.constants.columns as c
from src.logistic_reg.logistic_reg_train import LOGISTIC_REG_MODELS_PATH
from sklearn.metrics import f1_score


def predict():
    test_df = pd.read_csv(os.path.join(files.INTERIM_DATA, files.TEST))

    preprocessing_pipeline = load(os.path.join(files.PIPELINES, files.PREPROCESSING_PIPELINE))
    preprocessed_test = preprocessing_pipeline.transform(test_df)

    logistic_reg_model_name = "logistic_regression.joblib"

    logging.info(f"Make predictions with {logistic_reg_model_name}")
    logistic_reg = load(os.path.join(LOGISTIC_REG_MODELS_PATH, logistic_reg_model_name))
    y_pred = logistic_reg.predict(preprocessed_test)

    preprocessed_test["prediction"] = y_pred
    preprocessed_test.to_csv(os.path.join(files.OUTPUT_DATA, files.PREDICTIONS_TEST))




