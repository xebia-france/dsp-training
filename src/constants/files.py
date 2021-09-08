import os
import src.constants.models as m


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
LOGISTIC_REG_MODELS_PATH = create_folder(os.path.join(MODELS, "logistic_reg"))

LOANS_DATA_URL = "https://ps-dsp-training.s3.eu-west-1.amazonaws.com/loans.csv"

LOANS = os.path.join(RAW_DATA, "loans.csv")

TRAIN = os.path.join(INTERIM_DATA, "train.csv")
TEST = os.path.join(INTERIM_DATA, "test.csv")
PREPROCESSED_TRAIN = os.path.join(INTERIM_DATA, "preprocessed_train.csv")

PREPROCESSING_PIPELINE = os.path.join(PIPELINES, "pipeline.joblib")

LOGISTIC_REG_MODEL = os.path.join(LOGISTIC_REG_MODELS_PATH, m.LOGISTIC_REG_MODEL_NAME + ".joblib")

PREDICTIONS_TEST = os.path.join(OUTPUT_DATA, f"{m.LOGISTIC_REG_MODEL_NAME}_predictions.csv")