## This file tests the functiuons in aggregates.py to check they return expected outputs

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from open_cge import aggregates

# Total investment
def test_eqXg():
    g = 0.03
    kk = 100
    expected_XXv = 3
    test_XXv = aggregates.eqXg(g, kk)
    print('Type = ', type(test_XXv))
    assert expected_XXv == test_XXv


# Total household saving
def test_eqSp():
    ssp = 0.1
    pf = np.array([5, 6, 3, 4])
    Ff = np.array([6, 4, 10, 5])
    Fsh = 5
    Trf = 1
    expected_Sp = 10
    test_Sp = aggregates.eqSp(ssp, pf, Ff, Fsh, Trf)
    assert expected_Sp == test_Sp


# Domestic capital holdings
def test_Kd():
    g = 0.05
    Sp = 100
    lam = np.array([0.3, 0.2, 0.4, 0.1])
    pq = np.array([2, 4, 6, 8])
    assert sum(lam) == 1
    expected_Kd = 100
    test_Kd = aggregates.eqKd(g, Sp, lam, pq)
    assert expected_Kd == test_Kd

# Foreign holdings of domestically used capital
def test_eqKf():
    Kk = 110
    Kd = 100
    expected_Kf = 10
    test_Kf = aggregates.eqKf(Kk, Kd)
    assert expected_Kf == test_Kf

# Capital market clearing equation
def test_eqKk():
    pf = np.array([])
    Ff = np.array([])

# Balance of payments
def test_eqbop():


# Net foreign investment/savings
def test_eqSf():


# Resource constraint
def test_eqpqerror():


# Comparing labor supply from the model to that in the data
def test_eqpf():


# Comparing capital demand in the model and data
def test_eqpk():


# Total investment
def eqXXv():


#
