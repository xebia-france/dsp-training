import os
import logging
from datetime import datetime


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


# equals True when running test with pytest because of environment variable specified in pytest.ini
is_running_test = eval(os.getenv("IS_RUNNING_TEST", "False"))

# TODO: améliorer avec pathlib ?
PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",)
# data path will be /data for project and /tests/integration_test/data for tests
if not is_running_test:
    DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "data"))
else:
    DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "tests", "integration_test", "data_test"))

# TODO: revoir la gestion des logs
# Set up writing of logs to today’s log file
LOG_PATH = create_folder(os.path.join(DATA_PATH, "logs"))

today_str = datetime.now().strftime("%Y-%m-%d")
LOG_FILE = os.path.join(LOG_PATH, today_str + ".log")
logging_format = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format=logging_format)

# Set up logging to console
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter(logging_format))
logging.getLogger('').addHandler(console)

RAW_DATA = create_folder(os.path.join(DATA_PATH, "raw"))
INTERIM_DATA = create_folder(os.path.join(DATA_PATH, "interim"))
OUTPUT_DATA = create_folder(os.path.join(DATA_PATH, "output"))

PLOTS = create_folder(os.path.join(OUTPUT_DATA, "plots"))

GDP_ENERGY_DATA_URL = "https://gitlab.com/VincentVillet/cookiecutter-data-fr/-/raw/master/world_gdp_and_energy_consumption.csv"

LOANS = "loans.csv"
TRAIN = "train.csv"
TEST = "test.csv"
TEST_WITH_PREDICTIONS = "test_with_predictions.csv"
PREPROCESSED_TRAIN = "preprocessed_train.csv"
PREPROCESSING_PIPELINE = "pipeline.joblib"

MLFLOW_EXPERIMENT_NAME = "mlflow-experiment"