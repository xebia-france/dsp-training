# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# add src to python path
import sys
sys.path.append("..")

from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from joblib import load

import src.constants.columns as c
import src.constants.files as files
import src.constants.models as md

from src.lin_reg.lin_reg_core import split_train_test
from src.lin_reg.lin_reg_train import LIN_REG_MODELS_PATH
from src.utils import mean_absolute_percentage_error


LIN_REG_PLOTS = files.create_folder(os.path.join(files.PLOTS, "lin_reg"))

# %matplotlib inline

# +
world_gdp_energy = pd.read_csv(os.path.join(files.INTERIM_DATA, files.GDP_ENERGY_DATA_CSV))

train_df, test_df = split_train_test(world_gdp_energy)
    
lin_reg = load(os.path.join(LIN_REG_MODELS_PATH, "world_gdp_fuel_lin_reg_1965_to_2000.joblib"))
predictions = lin_reg.predict(test_df[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD].values.reshape(-1, 1))

mape = mean_absolute_percentage_error(test_df[c.EnergyConsumptionGDP.WORLD_ENERGY_CONSUMPTION], predictions)
# -

world_gdp_energy.tail()

# +
matplotlib.rcParams.update({'font.size': 22})

plt.figure(1, figsize=(25, 12))

plt.scatter(world_gdp_energy[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD],
         world_gdp_energy[c.EnergyConsumptionGDP.WORLD_ENERGY_CONSUMPTION], label="valeurs réelles")

plt.scatter(test_df[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD], predictions,
            label="prédictions", s=300, color="red", marker="x")

plt.plot(world_gdp_energy[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD],
         lin_reg.predict(world_gdp_energy[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD].values.reshape(-1, 1)),
        label="régression linéaire", linewidth=3, color="green")

plt.xlabel("PIB mondial en milliards de dollars de 2016")
plt.ylabel("Consommation d’énergie primaire mondiale\nen millions de tonnes de pétrole")

plt.title("Régression linéaire de la consommation d’énergie primaire mondiale en fonction du PIB mondial\n"
         "Écart absolu relatif moyen des prédictions: " + str(round(100*mape, 1)) + "%",
         fontsize=30)
plt.legend()
# -

plt.plot(world_gdp_fuel[c.FuelConsumptionGDP.YEAR], world_gdp_fuel[c.FuelConsumptionGDP.WORLD_FUEL_CONSUMPTION])
plt.show()

plt.show()


