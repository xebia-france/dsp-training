import os


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",)
DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "data"))

RAW_DATA = create_folder(os.path.join(DATA_PATH, "raw"))
INTERIM_DATA = create_folder(os.path.join(DATA_PATH, "interim"))
OUTPUT_DATA = create_folder(os.path.join(DATA_PATH, "output"))
MODELS = create_folder(os.path.join(DATA_PATH, "models"))
PIPELINES = create_folder(os.path.join(DATA_PATH, "pipelines"))

# TODO: put google-cloud bucket url
LOANS_DATA_URL = "https://gitlab.com/VincentVillet/cookiecutter-data-fr/-/raw/master/world_gdp_and_energy_consumption.csv"

LOANS = "loans.csv"
TRAIN = "train.csv"
TEST = "test.csv"
PREPROCESSED_TRAIN = "preprocessed_train.csv"
PREPROCESSING_PIPELINE = "pipeline.joblib"
PREDICTIONS_TEST = "predictions.csv"