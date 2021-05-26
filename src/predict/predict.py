from joblib import load
import pandas as pd
import logging

import src.constants.models as m


def predict(test_file_path, preprocessing_pipeline_path, logistic_reg_model_path, prediction_file_path):
    """
    Make prediction on test data with saved model.

    :param test_file_path: path to test data.
    :param preprocessing_pipeline_path: path to saved preprocessing pipeline.
    :param logistic_reg_model_path: path to saved logistic reg model.
    :param prediction_file_path: path to prediction data.

    :return: None
    """
    test_df = pd.read_csv(test_file_path)

    preprocessing_pipeline = load(preprocessing_pipeline_path)
    # TODO: preprocess test data
    preprocessed_test = None

    logging.info(f"Make predictions with {m.LOGISTIC_REG_MODEL_NAME}")
    # TODO: load model and compute prediction

    test_df["prediction"] = y_pred

    test_df.to_csv(prediction_file_path, index=False)




