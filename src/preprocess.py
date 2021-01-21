import logging
import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import src.constants.columns as c
import src.constants.files as files


def preprocess(training_file_path, preprocessed_train_destination, preprocessing_pipeline_destination):
    """
    Take training_file_path as input and write preprocessed data into preprocessed_train_destination.

    :return: None.
    """
    logging.info("Preprocessing raw data")
    train_df = pd.read_csv(training_file_path)

    num_features = c.Loans.num_features()
    cat_features = c.Loans.cat_features()

    pipeline = fit_preprocessing_pipeline(train_df, num_features, cat_features)

    logging.info("Transforming the column_transformer")
    preprocessed_train = pipeline.transform(train_df)

    one_hot_cols = retrieve_one_hot_columns(pipeline, cat_features)
    preprocessed_train_df = pd.DataFrame(preprocessed_train, columns=num_features + one_hot_cols)

    preprocessed_train_df[c.Loans.Loan_Status] = train_df[c.Loans.Loan_Status]

    logging.info("Saving the preprocessed train dataframe")
    preprocessed_train_df.to_csv(preprocessed_train_destination, index=False)

    logging.info("Saving the preprocessing pipeline")
    dump(pipeline, preprocessing_pipeline_destination)


def fit_preprocessing_pipeline(train_df, num_features, cat_features):
    pipeline = ColumnTransformer([
        (
            "num_pipeline",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            num_features
        ),
        (
            "cat_pipeline",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(drop="if_binary"))
            ]),
            cat_features
        )
    ])

    pipeline.fit(train_df)
    return pipeline


def retrieve_one_hot_columns(pipeline, cat_features):
    raw_one_hot_cols = pipeline.named_transformers_["cat_pipeline"].named_steps["one_hot_encoder"].get_feature_names()
    one_hot_cols = []
    for i in range(len(raw_one_hot_cols)):
        one_hot_col_name = cat_features[int(raw_one_hot_cols[i][1])] + raw_one_hot_cols[i][2:]
        one_hot_cols.append(one_hot_col_name)

    return one_hot_cols


def split_train_test(test_size=0.2, random_state=1):
    loans_df = pd.read_csv(os.path.join(files.RAW_DATA, files.LOANS))

    train_df, test_df = train_test_split(loans_df, test_size=test_size, random_state=random_state)

    train_df.to_csv(os.path.join(files.INTERIM_DATA, files.TRAIN), index=False)
    test_df.to_csv(os.path.join(files.INTERIM_DATA, files.TEST), index=False)
