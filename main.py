import logging
import os
import mlflow
from datetime import datetime

from src.evaluation.evaluate_mlflow import evaluate_mlflow
from src.utils import download_file_from_url
from src.preprocess import preprocess, split_train_test
from src.logistic_reg.logistic_reg_train import logistic_reg_train
import src.constants.files as files


def main(bool_dict):
    """
    Launch all project steps.

    :param bool_dict: Dictionnary with step names as keys and boolean as values allowing to bypass steps. This can be
    useful to re-run all steps but model training steps if they are already done for example.
    :return:
    """
    download_file_from_url(files.GDP_ENERGY_DATA_URL, os.path.join(files.RAW_DATA, files.LOANS))

    today_str = str(datetime.date(datetime.now()))
    with mlflow.start_run(run_name=today_str):
        if bool_dict["split"]:
            split_train_test()

        if bool_dict["preprocess"]:
            preprocess(
                training_file_path=os.path.join(files.INTERIM_DATA, files.TRAIN),
                preprocessed_train_destination=os.path.join(files.INTERIM_DATA, files.PREPROCESSED_TRAIN)
            )

        if bool_dict["logistic_reg_train"]:
            logistic_reg_train()

        if bool_dict["evaluate_mlflow"]:
            evaluate_mlflow()


if __name__ == "__main__":
    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate_mlflow": True}

    main(bool_dict)

    # TODO: model registry.
    # TODO: Nettoyer les tests.
    # TODO: Rajouter des tests notamment sur le mécanisme de récupération du dernier preprocessing_pipeline.
    # TODO: revoir les README
