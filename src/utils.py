from tqdm import tqdm
import os
import logging
import requests
import numpy as np

import src.constants.files as files


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def download_file_from_url(url, file_path, overwrite=False):
    if not overwrite and os.path.exists(file_path):
        logging.info(f"File '{file_path.replace(files.PROJECT_ROOT_PATH, '')}' already exists, we do not overwrite it.")
        return

    logging.info(f"Download file at url {url} to '{file_path.replace(files.PROJECT_ROOT_PATH, '')}'")
    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in tqdm(r.iter_content(10 ** 5)):
            f.write(chunk)
