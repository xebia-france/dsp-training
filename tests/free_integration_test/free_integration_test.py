from main import main
from tests.free_integration_test.prepare_raw_test_data import prepare_raw_test_data


def test_main():
    bool_dict = {"split": True,
                 "preprocess": True,
                 "logistic_reg_train": True,
                 "evaluate": True}
    prepare_raw_test_data()
    main(bool_dict)
