import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from joblib import dump
import logging

import src.constants.files as files
import src.constants.columns as c


LOGISTIC_REG_MODELS_PATH = files.create_folder(os.path.join(files.MODELS, "logistic_reg"))


def logistic_reg_train():
    """
    Read preprocessed train data, instatiate model and fit on train data
    Save model
    :return: None
    """
    train_df = pd.read_csv(os.path.join(files.INTERIM_DATA, files.PREPROCESSED_TRAIN))

    logistic_reg = LogisticRegression()
    
    logistic_reg.fit(
        train_df.drop(c.Loans.Loan_Status, axis=1).values,
        train_df[c.Loans.Loan_Status].values)

    logging.info("Saving model")
    dump(logistic_reg, os.path.join(LOGISTIC_REG_MODELS_PATH, "logistic_regression.joblib"))
