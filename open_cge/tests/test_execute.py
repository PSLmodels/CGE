import pytest;
import numpy as np;
import pandas as pd;
from pandas.testing import assert_series_equal;
from open_cge import execute;

def test_runner():
    '''
    Test of mdoel solution from execute.runner
    '''

    dict = {'AGR': 115.999989, 'OIL': 149.000054, 'IND': 1431.999758, 'SER': 1803.000214}
    expectedQ = pd.DataFrame.from_dict(dict, orient='index')
    expectedQ.columns = [None]
    testQ = execute.runner()
    assert_series_equal(expectedQ[None], testQ, check_dtype=False)
