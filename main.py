import os

from src.evaluation.evaluate_mlflow import evaluate_mlflow
from src.utils import download_file_from_url
from src.preprocess import preprocess, split_train_test
from src.logistic_reg.logistic_reg_train import logistic_reg_train
from src.evaluation.evaluate import evaluate
import src.constants.files as files


def main(bool_dict):
    """
    Launch all project steps.

    :param bool_dict: Dictionnary with step names as keys and boolean as values allowing to bypass steps. This can be
    useful to re-run all steps but model training steps if they are already done for example.
    :return:
    """
    download_file_from_url(files.GDP_ENERGY_DATA_URL, os.path.join(files.RAW_DATA, files.LOANS))

    if bool_dict["split"]:
        split_train_test()

    if bool_dict["preprocess"]:
        preprocess(
            training_file_path=os.path.join(files.INTERIM_DATA, files.TRAIN),
            preprocessed_train_destination=os.path.join(files.INTERIM_DATA, files.PREPROCESSED_TRAIN),
            preprocessing_pipeline_destination=os.path.join(files.PIPELINES, files.PREPROCESSING_PIPELINE)
        )

    if bool_dict["logistic_reg_train"]:
        logistic_reg_train()

    if bool_dict["evaluate"]:
        evaluate()

    if bool_dict["evaluate_mlflow"]:
        evaluate_mlflow()


if __name__ == "__main__":
    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True,
                 "evaluate_mlflow": True}
    main(bool_dict)
