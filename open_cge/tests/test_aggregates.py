## This file tests the functiuons in aggregates.py to check they return expected outputs

import pytest
import math
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from numpy.testing import assert_allclose
from open_cge import aggregates

# Total household saving
def test_eqSp():
    ssp = 0.1
    pf = np.array([5, 6, 3, 4])
    Ff = np.array([6, 4, 10, 5])
    Fsh = 5.0
    Trf = 1.0
    expected_Sp = 10.0
    test_Sp = aggregates.eqSp(ssp, pf, Ff, Fsh, Trf)
    assert expected_Sp == test_Sp


# Domestic capital holdings
def test_Kd():
    g = 0.03
    Sp = 15.0
    lam = np.array([0.3, 0.2, 0.4, 0.1])
    pq = np.array([3, 5, 6, 7])
    assert sum(lam) == 1
    expected_Kd = 100.0
    test_Kd = aggregates.eqKd(g, Sp, lam, pq)
    assert math.isclose(expected_Kd, test_Kd)

# Foreign holdings of domestically used capital
def test_eqKf():
    Kk = 110.0
    Kd = 100.0
    expected_Kf = 10.0
    test_Kf = aggregates.eqKf(Kk, Kd)
    assert expected_Kf == test_Kf

# Capital market clearing equation
#def test_eqKk():
#    pf = np.array([2, 4, 5, 1])
#    Ff = np.array([3, 4, 5, 7])
#    R = 0.02
#    lam = np.array([0.3, 0.2, 0.1, 0.4])
#    assert sum(lam) == 1
#    pq = np.array([5, 6, 8, 4])
#    expected_Kk = 529.41
#    test_Kk = aggregates.eqKk(pf, Ff, R, lam, pq)
#    assert_allclose(expected_Kk, test_Kk)

# Balance of payments
def test_eqbop():
    pWe = np.array( [2, 3, 6, 7])
    pWm = np.array([4, 6, 2, 5])
    E = np.array([5, 8, 10, 12])
    M = np.array([10, 6, 12, 15])
    Sf = 4.0
    Fsh = 10.0
    er = 2.0
    expected_bop_error = 0.0
    test_bop_error = aggregates.eqbop(pWe, pWm, E, M, Sf, Fsh, er)
    assert expected_bop_error == test_bop_error

# Net foreign investment/savings
def test_eqSf():
    g = 0.03
    Kf = 100
    lam = np.array([0.2, 0.3, 0.2, 0.3])
    pq = np.array([4, 6, 3, 5])
    assert sum(lam) == 1
    expected_Sf = 14.1
    test_Sf = aggregates.eqSf(g, lam, pq, Kf)
    print(test_Sf)
    assert_allclose(expected_Sf, test_Sf)

# Resource constraint
def test_eqpqerror():
    Q = np.array([30, 40, 50, 60])
    Xp = np.array([10, 15, 30, 35])
    Xg = np.array([4, 6, 7, 9])
    Xv = np.array([6, 4, 3, 6])
    X = np.array([[2, 4, 3, 1], [4, 4, 2, 5], [4, 2, 3, 1], [2, 1, 3, 4]])
    expected_pq_error = np.array([0, 0, 0, 0])
    test_pq_error = aggregates.eqpqerror(Q, Xp, Xg, Xv, X)
    assert_allclose(expected_pq_error, test_pq_error)



# Comparing labor supply from the model to that in the data
def test_eqpf():
    F = pd.DataFrame.from_dict({'LAB': [10, 40], 'CAP': [30, 20]},
    orient = 'index')
    Ff0 = pd.Series(50, index = ['LAB', 'CAP'])
    expected_pf_error = 0
    test_pf_error = aggregates.eqpf(F, Ff0)
    assert_allclose(expected_pf_error, test_pf_error)


# Comparing capital demand in the model and data
def test_eqpk():
    F = pd.DataFrame.from_dict({'LAB': [10, 40], 'CAP': [30, 20]},
    orient = 'index')
    Kk = 50
    Ff0 = Ff0 = pd.Series(50, index = ['LAB', 'CAP'])
    Kk0 = Ff0 = pd.Series(50, index = ['CAP'])
    expected_pk_error = 0
    test_pk_error = aggregates.eqpk(F, Kk, Ff0, Kk0)
    assert_allclose(expected_pk_error, test_pk_error)


# Total investment
def eqXXv():
    g = 0.03
    Kk = 100
    expected_eqXXv = 3
    test_eqXXv = aggregates.eqXXv(g, Kk)
    assert expected_eqXXv == test_eqXXv
