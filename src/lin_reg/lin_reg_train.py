import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from joblib import dump
import logging

import src.constants.files as files
import src.constants.columns as c

from src.lin_reg.lin_reg_core import split_train_test

LIN_REG_MODELS_PATH = files.create_folder(os.path.join(files.MODELS, "lin_reg"))


def lin_reg_train():
    """
    Train linear regression with World GDP as input and World energy consumption as label to predict.
    
    :return: None
    """
    world_gdp_energy = pd.read_csv(os.path.join(files.INTERIM_DATA, files.GDP_ENERGY_DATA_CSV))
    
    train_df, test_df = split_train_test(world_gdp_energy)
    min_year = train_df[c.EnergyConsumptionGDP.YEAR].min()
    max_year = train_df[c.EnergyConsumptionGDP.YEAR].max()
    
    logging.info(f"Training linear regression on {min_year} to {max_year} world GDP and energy data.")
    lin_reg = LinearRegression()
    
    lin_reg.fit(
        train_df[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD].values.reshape(-1, 1),
        train_df[c.EnergyConsumptionGDP.WORLD_ENERGY_CONSUMPTION])
    
    logging.info("Saving model.")
    dump(lin_reg, os.path.join(LIN_REG_MODELS_PATH, f"world_gdp_energy_lin_reg_{min_year}_to_{max_year}.joblib"))
