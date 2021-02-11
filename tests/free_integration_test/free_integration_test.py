import shutil

from main import main
import mlflow
import os
from tests.free_integration_test.prepare_raw_test_data import prepare_raw_test_data


LOCAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def test_main():
    mlruns_path = f"{LOCAT_ROOT}/mlruns_test"
    setup_mlruns(mlruns_path)

    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True,
                 "evaluate_mlflow": False}
    prepare_raw_test_data()
    main(bool_dict)


def setup_mlruns(mlruns_path):
    if os.path.exists(mlruns_path):
        shutil.rmtree(mlruns_path)

    mlflow.set_tracking_uri(mlruns_path)
