class EnergyConsumptionGDP():
    """
    Dataframe of evolution of world GDP and energy consumption between 1965 and 2008.
    """
    YEAR = "year"

    WORLD_ENERGY_CONSUMPTION = "world energy consumption in million tons oil equivalent"
    """
    Total world consumption of commercially traded energys only, expressed in million tons oil equivalent. This
    comprises oil, natural gaz, coal, nuclear and hydro electric energy, and excludes wood, peat, animal waste,
    wind, geothermal and solar power.
    Source: BP statistical review of World energy, 2009 https://openei.org/wiki/BP_Statistical_Review_of_World_Energy.
    """

    WORLD_GDP_USD = "world GDP in 2016 USD"
    """
    World GDP in 2016 USD.
    Source: https://pkgstore.datahub.io/core/gdp/gdp_csv/data/0048bc8f6228d0393d41cac4b663b90f/gdp_csv.csv
    """
    
    WORLD_GDP_BILLION_USD = "world GDP in billion 2016 USD"
