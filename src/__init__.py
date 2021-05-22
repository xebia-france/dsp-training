import logging
from datetime import datetime
import os

from src.constants.files import create_folder, DATA_PATH


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
