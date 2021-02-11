import shutil

from main import main
import mlflow
import os
import pandas as pd

from src.constants import files

LOCAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def setup_mlruns(mlruns_path):
    if os.path.exists(mlruns_path):
        shutil.rmtree(mlruns_path)

    mlflow.set_tracking_uri(mlruns_path)


def test_main_runs():
    mlruns_path = f"{LOCAT_ROOT}/mlruns_test"
    setup_mlruns(mlruns_path)

    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True,
                 "evaluate_mlflow": True}

    main(bool_dict)


def test_main():
    mlruns_path = f"{LOCAT_ROOT}/mlruns_test"
    setup_mlruns(mlruns_path)

    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True,
                 "evaluate_mlflow": True}

    main(bool_dict)

    # Then
    expected = pd.read_csv(os.path.join(LOCAT_ROOT, "expected_predictions.csv"))
    # Read result from csv to avoid problems with nan
    result = pd.read_csv(os.path.join(files.OUTPUT_DATA, files.TEST_WITH_PREDICTIONS))

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)


def test_main_runs_with_preprocess_false():
    mlruns_path = f"{LOCAT_ROOT}/mlruns_test"
    setup_mlruns(mlruns_path)

    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True,
                 "evaluate_mlflow": True}

    main(bool_dict)

    bool_dict["preprocess"] = False

    main(bool_dict)