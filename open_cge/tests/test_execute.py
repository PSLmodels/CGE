import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal
from open_cge import execute


def test_runner():
    """
    Test of model solution from execute.runner
    """

    dict = {
        "AGR": 115.999989,
        "OIL": 149.000054,
        "IND": 1431.999758,
        "SER": 1803.000214,
    }
    expectedQ = pd.DataFrame.from_dict(dict, orient="index")
    expectedQ.columns = [None]
    testQ = execute.runner()
    assert_series_equal(expectedQ[None], testQ, check_dtype=False)


def test_check_square():
    """
    Test of check_square function
    """
    testdata = [[1, 2, 3], [2, 3, 4], [3, 1, 4]]
    mat = pd.DataFrame(data=testdata, index=None)
    mat.to_numpy(dtype=None, copy=True)
    if not mat.shape[0] == mat.shape[1]:
        raise ValueError(
            f"SAM is not square. It has {mat.shape[0]} rows and {mat.shape[1]} columns"
        )


def test_row_total():
    """
    Test of row_total function
    """
    rowdict = {
        "AGR": 118,
        "OIL": 1015.435052,
        "IND": 1738,
        "SER": 1837,
        "LAB": 551,
        "CAP": 1051,
        "LAND": 24,
        "NTR": 890.4350515,
        "DTX": 948,
        "IDT": 21,
        "ACT": 9,
        "HOH": 2616.435052,
        "GOV": 978,
        "INV": 485,
        "EXT": 1504.435052,
    }
    expected_rowtotal = pd.DataFrame.from_dict(rowdict, orient="index")
    expected_rowtotal.columns = [None]
    test_rowtotal = execute.row_total()
    assert_series_equal(
        expected_rowtotal[None], test_rowtotal, check_dtype=False
    )


def test_col_total():
    """
    Test of col_total function
    """
    coldict = {
        "AGR": 118,
        "OIL": 1015.435052,
        "IND": 1738,
        "SER": 1837,
        "LAB": 551,
        "CAP": 1051,
        "LAND": 24,
        "NTR": 890.4350515,
        "DTX": 948,
        "IDT": 21,
        "ACT": 9,
        "HOH": 2616.435052,
        "GOV": 978,
        "INV": 485,
        "EXT": 1504.435052,
    }
    expected_coltotal = pd.DataFrame.from_dict(coldict, orient="index")
    expected_coltotal.columns = [None]
    test_coltotal = execute.col_total()
    assert_series_equal(
        expected_coltotal[None], test_coltotal, check_dtype=False
    )


def test_row_col_equal():
    """
    Test of row_col_equal function
    """
    data = {
        "row_1": [3, 2, 1, 0],
        "row_2": [2, 3, 4, 5],
        "row_3": [1, 4, 3, 6],
        "row_4": [0, 5, 6, 3],
    }
    test_df = pd.DataFrame.from_dict(data, orient="index")
    test_row_sum = test_df.sum(axis=0)
    test_col_sum = test_df.sum(axis=1)
    np.testing.assert_allclose(test_row_sum, test_col_sum)
