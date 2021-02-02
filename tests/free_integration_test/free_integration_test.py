from main import main
import mlflow
import os
from tests.free_integration_test.prepare_raw_test_data import prepare_raw_test_data


LOCAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def test_main():
    mlflow.set_tracking_uri(f"{LOCAT_ROOT}/mlruns_test")
    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True,
                 "evaluate_mlflow": False}
    prepare_raw_test_data()
    main(bool_dict)
