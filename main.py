import os

from src.utils import download_file_from_url
from src.preprocess import preprocess, split_train_test
from src.lin_reg.lin_reg_train import logistic_reg_train
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
        preprocess()

    if bool_dict["logistic_reg_train"]:
        logistic_reg_train()

    if bool_dict["evaluate"]:
        evaluate()


if __name__ == "__main__":
    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True}
    main(bool_dict)
