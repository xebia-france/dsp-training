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

from src.logistic_reg.logistic_reg_core import split_train_test
from src.logistic_reg.logistic_reg_train import LOGISTIC_REG_MODELS_PATH

register_matplotlib_converters()

LIN_REG_PLOTS = files.create_folder(os.path.join(files.PLOTS, "logistic_reg"))

