import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from joblib import dump
import logging
from datetime import date
import mlflow

import src.constants.columns as c
import src.constants.files as files
import src.constants.models as models


def logistic_reg_train():
    # TODO: documentation
    """

    
    :return: None
    """
    train_df = pd.read_csv(os.path.join(files.INTERIM_DATA, files.PREPROCESSED_TRAIN))

    logistic_reg = LogisticRegression()
    
    logistic_reg.fit(
        train_df.drop(c.Loans.Loan_Status, axis=1).values,
        train_df[c.Loans.Loan_Status].values)

    logging.info("Saving model")

    mlflow.sklearn.log_model(logistic_reg, models.MODEL_NAME)
