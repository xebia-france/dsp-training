import os
import logging
from datetime import datetime


def create_folder(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


# equals True when running test with pytest because of environment variable specified in pytest.ini
is_running_test = eval(os.getenv("IS_RUNNING_TEST", "False")) 

PROJECT_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",)
# data path will be /data for project and /tests/free_integration_test/data for tests
if not is_running_test:
    DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "data"))
else:
    DATA_PATH = create_folder(os.path.join(PROJECT_ROOT_PATH, "tests", "free_integration_test", "data"))

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
MODELS = create_folder(os.path.join(DATA_PATH, "models"))

PLOTS = create_folder(os.path.join(OUTPUT_DATA, "plots"))

GDP_ENERGY_DATA_URL = "https://gitlab.com/VincentVillet/cookiecutter-data-fr/-/raw/master/world_gdp_and_energy_consumption.csv"

GDP_ENERGY_DATA_CSV = "world_gdp_and_energy_consumption.csv"