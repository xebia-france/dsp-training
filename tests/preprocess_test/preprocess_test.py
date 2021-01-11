import os
import pandas as pd
import logging

from src.preprocess import convert_world_GDP_in_billion_usd


LOCAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)))


def test_preprocess():
    input = pd.read_csv(os.path.join(LOCAT_ROOT, "input - world_gdp_energy.csv"))

    result = convert_world_GDP_in_billion_usd(input)

    logging.info("writing results csv")
    result.to_csv(os.path.join(LOCAT_ROOT, "result.csv"), index=False)

    expected = pd.read_csv(os.path.join(LOCAT_ROOT, "expected.csv"))
    # Read result from csv to avoid problems with nan
    result = pd.read_csv(os.path.join(LOCAT_ROOT, "result.csv"))

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
