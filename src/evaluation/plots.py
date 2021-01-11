import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
from joblib import load

import src.constants.columns as c
import src.constants.files as files
import src.constants.models as md

from src.lin_reg.lin_reg_core import split_train_test
from src.lin_reg.lin_reg_train import LIN_REG_MODELS_PATH

register_matplotlib_converters()

LIN_REG_PLOTS = files.create_folder(os.path.join(files.PLOTS, "lin_reg"))


def plot_world_gdp_energy_lin_reg(world_gdp_energy, predictions, test_df, mape, lin_reg, lin_reg_model_name):
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
              "Écart absolu relatif moyen des prédictions: " + str(round(100 * mape, 1)) + "%",
              fontsize=30)
    plt.legend()
    plt.savefig(os.path.join(LIN_REG_PLOTS, lin_reg_model_name.replace("joblib", "png")))

    plt.close()
