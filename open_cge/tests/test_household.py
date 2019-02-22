import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from open_cge import household

def test_eqF():
    beta_vals = np.array([[0.3, 0.4, 0.5], [0.4, 0.4, 0.3], [0.4, 0.2, 0.2]])
    index = ['Row'+str(i) for i in range(1, beta_vals.shape[0]+ 1)]
    beta = pd.DataFrame(beta_vals, index=index)
    py = np.array([0.5, 2, 88])
    pf = np.array([4, 1, 5])
    Y = np.array([6, 3, 0.4])
    expected_F_vals = np.array([[0.225, 0.6, 4.4], [1.2, 2.4, 10.56],
                           [0.24,  0.24, 1.408]])
    expected_F = pd.DataFrame(expected_F_vals, index=index)
    test_F = household.eqF(beta, py, Y, pf)
    print('Type = ', type(test_F))
    print(beta)
    assert_frame_equal(
        expected_F,
        test_F,
        check_dtype=False)
