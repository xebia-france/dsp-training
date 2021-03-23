import os
from joblib import load
import pandas as pd
import logging

import src.constants.files as files
import src.constants.models as m
from src.logistic_reg.logistic_reg_train import LOGISTIC_REG_MODELS_PATH


def predict():
    test_df = pd.read_csv(os.path.join(files.INTERIM_DATA, files.TEST))

    preprocessing_pipeline = load(os.path.join(files.PIPELINES, files.PREPROCESSING_PIPELINE))
    preprocessed_test = preprocessing_pipeline.transform(test_df)

    logging.info(f"Make predictions with {m.LOGISTIC_REG_MODEL_NAME}")
    logistic_reg = load(os.path.join(LOGISTIC_REG_MODELS_PATH, m.LOGISTIC_REG_MODEL_NAME))
    y_pred = logistic_reg.predict(preprocessed_test)

    preprocessed_test["prediction"] = y_pred

    model_name = m.LOGISTIC_REG_MODEL_NAME.replace('.joblib', '')
    preprocessed_test.to_csv(os.path.join(files.OUTPUT_DATA, f"{model_name}_{files.PREDICTIONS_TEST}"))




