import os

from src.utils import download_file_from_url
from src.preprocess.preprocess import preprocess, load_and_split_data
from src.logistic_reg.logistic_reg_train import logistic_reg_train
from src.evaluation.evaluate import evaluate
from src.predict.predict import predict
import src.constants.files as files


def main(bool_dict):
    """
    Launch all project steps.

    :param bool_dict: Dictionnary with step names as keys and boolean as values allowing to bypass steps. This can be
    useful to re-run all steps but model training steps if they are already done for example.
    :return:
    """
    download_file_from_url(files.LOANS_DATA_URL, os.path.join(files.RAW_DATA, files.LOANS))

    if bool_dict["load_and_split"]:
        load_and_split_data()

    if bool_dict["preprocess"]:
        preprocess(
            training_file_path=os.path.join(files.INTERIM_DATA, files.TRAIN),
            preprocessed_train_destination=os.path.join(files.INTERIM_DATA, files.PREPROCESSED_TRAIN),
            preprocessing_pipeline_destination=os.path.join(files.PIPELINES, files.PREPROCESSING_PIPELINE)
        )

    if bool_dict["logistic_reg_train"]:
        logistic_reg_train()

    if bool_dict["predict"]:
        predict()

    if bool_dict["evaluate"]:
        evaluate()


if __name__ == "__main__":
    bool_dict = {"load_and_split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "predict":True,
                 "evaluate": True
                 }
    main(bool_dict)
