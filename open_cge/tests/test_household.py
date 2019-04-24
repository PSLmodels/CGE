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

def test_eqI():
    pf = np.array([4, 5, 10, 2])
    Ff = np.array([10, 4, 2, 10])
    Sp = 10
    Td = 10
    Fsh = 5
    expected_I = 75
    test_I = household.eqI(pf, Ff, Sp, Td, Fsh, Trf)
    print('Type = ', type(test_I))
    print(I)
    assert test_I == expected_I_vals

def test_eqXp():
    alpha_vals = np.array([0.2, 0.4, 0.4])
    assert sum(alpha_vals) == 1
    index = ['Row'+str(i) for i in range(1, alpha_vals.shape[0]+ 1)]
    alpha = pd.DataFrame(alpha_vals, index=index)
    I = 75
    pq = np.array([1, 3, 5])
    expected_Xp_vals = np.array([15, 10, 6])
    expected_Xp = pd.DataFrame(expected_Xp_vals, index=index)
    test_Xp = household.eqXp(alpha, I, pq)
    print('Type = ', type(test_Xp))
    print(alpha)
    assert_frame_equal(
        expected_Xp,
        test_Xp,
        check_dtype=False)
