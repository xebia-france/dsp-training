import src.constants.columns as c
import src.constants.models as md


def split_train_test(world_gdp_energy):
    train_df = world_gdp_energy[world_gdp_energy[c.EnergyConsumptionGDP.YEAR] <= md.END_TRAIN_YEAR]
    test_df = world_gdp_energy[world_gdp_energy[c.EnergyConsumptionGDP.YEAR] > md.END_TRAIN_YEAR]
    
    return train_df, test_df
