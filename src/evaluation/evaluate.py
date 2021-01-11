import os
from joblib import load
import pandas as pd
import logging

import src.constants.files as files
import src.constants.columns as c
from src.lin_reg.lin_reg_core import split_train_test
from src.lin_reg.lin_reg_train import LIN_REG_MODELS_PATH
from src.evaluation.plots import plot_world_gdp_energy_lin_reg
from src.utils import mean_absolute_percentage_error


def evaluate():
    world_gdp_energy = pd.read_csv(os.path.join(files.INTERIM_DATA, files.GDP_ENERGY_DATA_CSV))

    train_df, test_df = split_train_test(world_gdp_energy)

    lin_reg_model_names = [file for file in os.listdir(LIN_REG_MODELS_PATH) if "joblib" in file]

    for lin_reg_model_name in lin_reg_model_names:
        logging.info(f"Evaluating {lin_reg_model_name}")
        lin_reg = load(os.path.join(LIN_REG_MODELS_PATH, lin_reg_model_name))
        predictions = lin_reg.predict(test_df[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD].values.reshape(-1, 1))

        mape = mean_absolute_percentage_error(test_df[c.EnergyConsumptionGDP.WORLD_ENERGY_CONSUMPTION], predictions)

        plot_world_gdp_energy_lin_reg(world_gdp_energy, predictions, test_df, mape, lin_reg, lin_reg_model_name)
        