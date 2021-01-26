import os
import pandas as pd

from src.preprocess import preprocess


LOCAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def test_preprocess():
    # Given
    pipeline_destination = os.path.join(LOCAT_ROOT, "pipeline_test.joblib")
    preprocess_train_destination = os.path.join(LOCAT_ROOT, "result_test.csv")

    # When
    preprocess(
        training_file_path=os.path.join(LOCAT_ROOT, "loans_test.csv"),
        preprocessed_train_destination=preprocess_train_destination,
        preprocessing_pipeline_destination=pipeline_destination
    )

    # Then
    expected = pd.read_csv(os.path.join(LOCAT_ROOT, "expected.csv"))
    # Read result from csv to avoid problems with nan
    result = pd.read_csv(preprocess_train_destination)

    assert os.path.exists(pipeline_destination), "The preprocessing pipeline should have been created"
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
