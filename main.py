import os

from src.utils import download_file_from_url
from src.preprocess import preprocess
from src.lin_reg.lin_reg_train import lin_reg_train
from src.evaluation.evaluate import evaluate
import src.constants.files as files


def main(bool_dict):
    """
    Launch all project steps.

    :param bool_dict: Dictionnary with step names as keys and boolean as values allowing to bypass steps. This can be
    useful to re-run all steps but model training steps if they are already done for example.
    :return:
    """
    download_file_from_url(files.GDP_ENERGY_DATA_URL, os.path.join(files.RAW_DATA, files.GDP_ENERGY_DATA_CSV))
    
    if bool_dict["preprocess"]:
        preprocess()

    if bool_dict["lin_reg_train"]:
        lin_reg_train()

    if bool_dict["evaluate"]:
        evaluate()


if __name__ == "__main__":
    bool_dict = {"preprocess": True,
                 "lin_reg_train": True,
                 "evaluate": True}
    main(bool_dict)
