import os
import pandas as pd
import mlflow

from src.constants import models
from src.preprocess import preprocess
from src.constants import files


LOCAL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def test_preprocess():
    mlflow.set_tracking_uri(os.path.join("file:", LOCAL_ROOT, "mlruns_test").replace("C:", ""))
    mlflow.set_experiment(files.MLFLOW_EXPERIMENT_NAME)
    # Given
    preprocess_train_destination = os.path.join(LOCAL_ROOT, "result_test.csv")

    # When
    preprocess(
        training_file_path=os.path.join(LOCAL_ROOT, "loans_test.csv"),
        preprocessed_train_destination=preprocess_train_destination
    )

    # Then
    expected = pd.read_csv(os.path.join(LOCAL_ROOT, "expected.csv"))
    # Read result from csv to avoid problems with nan
    result = pd.read_csv(preprocess_train_destination)

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    try:
        artifact_uri = mlflow.active_run().info.artifact_uri
        mlflow.sklearn.load_model(f"{artifact_uri}/{models.PREPROCESSING_PIPELINE}")
    except IOError:
        raise AssertionError("The preprocessing pipeline has not been saved with mlflow")
