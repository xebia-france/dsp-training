import pandas as pd
import logging
import os

import src.constants.files as files
import src.constants.columns as c


def preprocess():
    """
    Take raw data as input and write preprocessed data into data/interim.
    
    :return: None.
    """
    # TODO: utiliser un pipeline sklearn
    # TODO: sauvegarder les transformers sklearn
    logging.info("Preprocessing raw data.")
    loans_df = pd.read_csv(os.path.join(files.RAW_DATA, files.LOANS_CSV))
    
    loans_df = convert_world_GDP_in_billion_usd(loans_df)
    
    loans_df.to_csv(os.path.join(files.INTERIM_DATA, files.LOANS_CSV), index=False)
    

def convert_world_GDP_in_billion_usd(world_gdp_energy):
    """
    Convert World GDP in billion USD for better readability.
    
    :param world_gdp_energy: Dataframe of evolution of world GDP and energy consumption between 1965 and 2008.
    :return: 
    """
    world_gdp_energy[c.EnergyConsumptionGDP.WORLD_GDP_BILLION_USD] = (
        world_gdp_energy[c.EnergyConsumptionGDP.WORLD_GDP_USD] / 10 ** 9).apply(int)
    
    return world_gdp_energy
