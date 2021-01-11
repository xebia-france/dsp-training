import pandas as pd
import os

import src.constants.files as files
import src.constants.columns as c
import src.constants.models as md

import logging


REAL_DATA_PATH = files.create_folder(os.path.join(files.PROJECT_ROOT_PATH, "data"))
REAL_RAW_DATA_PATH = files.create_folder(os.path.join(REAL_DATA_PATH, "raw"))
TEST_RAW_DATA_PATH = os.path.join(files.PROJECT_ROOT_PATH, "tests", "free_integration_test", "data", "raw")


def prepare_raw_test_data(force_recompute=False):
    real_raw_csv_files = [file for file in os.listdir(REAL_RAW_DATA_PATH) if file.endswith(".csv")]
    test_raw_csv_files = [file for file in os.listdir(TEST_RAW_DATA_PATH) if file.endswith(".csv")]

    files_to_copy_in_test = [file for file in real_raw_csv_files if file not in test_raw_csv_files]

    if force_recompute:
        files_to_copy_in_test = real_raw_csv_files

    for file in files_to_copy_in_test:
        logging.info(f"Truncating {file} and writing truncated version into test raw data folder.")
        df = pd.read_csv(os.path.join(REAL_RAW_DATA_PATH, file))

        # Keep 10 years before end train year and 5 years after.
        df_trunc = df[(df[c.EnergyConsumptionGDP.YEAR] >= md.END_TRAIN_YEAR - 10)
                      & (df[c.EnergyConsumptionGDP.YEAR] <= md.END_TRAIN_YEAR + 5)]

        df_trunc.to_csv(os.path.join(TEST_RAW_DATA_PATH, file), index=False)
