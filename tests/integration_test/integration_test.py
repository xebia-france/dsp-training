from main import main
import os
import pandas as pd
from unittest import TestCase
import shutil
import mlflow

from src.constants import files

LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
MLRUNS_PATH = files.create_folder(os.path.join("file:", LOCAL_ROOT, "mlruns_test").replace("C:", ""))


class IntegrationTest(TestCase):
    def setUp(self) -> None:
        # Is executed at the beginning of each test run
        mlflow.set_tracking_uri(MLRUNS_PATH)
        
    def tearDown(self) -> None:
        # Is executed at the end of each test run
        for i in range(10):
            mlruns_path_i = os.path.join(MLRUNS_PATH, str(i))
            if os.path.exists(mlruns_path_i):
                shutil.rmtree(mlruns_path_i)

    @staticmethod
    def test_main_runs():
        bool_dict = {"load_and_split": True,
                     "preprocess": True,
                     "logistic_reg_train": True,
                     "predict": True,
                     "evaluate": True}

        main(bool_dict)

    @staticmethod
    def test_main():
        bool_dict = {"load_and_split": True,
                     "preprocess": True,
                     "logistic_reg_train": True,
                     "predict": True,
                     "evaluate": True}

        main(bool_dict)

        # Then
        expected = pd.read_csv(os.path.join(LOCAL_ROOT, "expected_predictions.csv"))
        # Read result from csv to avoid problems with nan
        # TODO 5 : charger le dataframe contenant les prédictions.
        result = NotImplementedError()

        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    @staticmethod
    def test_main_runs_with_preprocess_false():
        bool_dict = {"load_and_split": True,
                     "preprocess": True,
                     "logistic_reg_train": True,
                     "predict": True,
                     "evaluate": True}

        main(bool_dict)

        bool_dict["preprocess"] = False

        main(bool_dict)
