import os
import pandas as pd
import mlflow

from src.preprocess import preprocess


LOCAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def test_preprocess():
    mlflow.set_tracking_uri(f"{LOCAT_ROOT}/mlruns_test")
    # Given
    preprocess_train_destination = os.path.join(LOCAT_ROOT, "result_test.csv")

    # When
    preprocess(
        training_file_path=os.path.join(LOCAT_ROOT, "loans_test.csv"),
        preprocessed_train_destination=preprocess_train_destination
    )

    # Then
    expected = pd.read_csv(os.path.join(LOCAT_ROOT, "expected.csv"))
    # Read result from csv to avoid problems with nan
    result = pd.read_csv(preprocess_train_destination)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
