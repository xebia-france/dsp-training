import os


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


# equals True when running test with pytest because of environment variable specified in pytest.ini
is_running_test = eval(os.getenv("IS_RUNNING_TEST", "False"))

PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",)
# data path will be /data for project and /tests/integration_test/data for tests
if not is_running_test:
    DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "data"))
else:
    DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "tests", "integration_test", "data_test"))

RAW_DATA = create_folder(os.path.join(DATA_PATH, "raw"))
INTERIM_DATA = create_folder(os.path.join(DATA_PATH, "interim"))
OUTPUT_DATA = create_folder(os.path.join(DATA_PATH, "output"))

PLOTS = create_folder(os.path.join(OUTPUT_DATA, "plots"))

if not is_running_test:
    LOANS_DATA_URL = "https://storage.googleapis.com/formation-dsp-data/loans.csv"
else:
    LOANS_DATA_URL = "https://storage.googleapis.com/formation-dsp-data/loans_test.csv"

LOANS = "loans.csv"
TRAIN = "train.csv"
TEST = "test.csv"
TEST_WITH_PREDICTIONS = "test_with_predictions.csv"
PREPROCESSED_TRAIN = "preprocessed_train.csv"
PREPROCESSING_PIPELINE = "pipeline.joblib"

MLFLOW_EXPERIMENT_NAME = "mlflow-experiment"