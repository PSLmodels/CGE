## This file tests the functions in government.py to check they produce the
## which are used in simpleCGE.py

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from open_cge import government
from numpy.testing import assert_allclose


# Direct tax revenue
def test_eqTd():
    taud = 0.10
    pf = np.array([2, 5, 3, 6])
    Ff = np.array([6, 2, 4, 3])
    expected_Td = 5.20
    test_Td = government.eqTd(taud, pf, Ff)
    assert_allclose(expected_Td, test_Td)

# Total transfers to households


# Production tax revenue from each commodity
def test_eqTz():
    tauz = np.array([0.02, 0.05, 0.08, 0.10])
    pz = np.array([8, 12, 15, 20])
    Z = np.array([100, 50, 60, 30])
    expected_Tz = np.array([16, 30, 72, 60])
    test_Tz = government.eqTz(tauz, pz, Z)
    assert_allclose(expected_Tz, test_Tz)

# Tariff revenue from each commodity
def test_eqTm():
    taum = np.array([0.05, 0.09, 0.10, 0.15])
    pm = np.array([10, 12, 15, 20])
    M = np.array([20, 15, 14, 10])
    expected_Tm = np.array([10, 16.2, 21, 30])
    test_Tm = government.eqTm(taum, pm, M)
    assert_allclose(expected_Tm, test_Tm)

# Government expenditures on commodity j
def test_eqXg():
    mu = np.array([0.2, 0.4, 0.3, 0.1])
    XXg = 100.0
    expected_Xg = np.array([20, 40, 30, 10])
    test_Xg = government.eqXg(mu, XXg)
    assert_allclose(expected_Xg, test_Xg)

# Total government savings
def test_eqSg():
    mu = np.array([0.2, 0.4, 0.3, 0.1])
    Td = 500.0
    Tz = np.array([50, 10, 40, 60])
    Tm = np.array([20, 30, 70, 90])
    XXg = 100.0
    Trf = 10.0
    pq = np.array([8, 10, 13, 15])
    expected_Sg = -240
    test_Sg = government.eqSg(mu, Td, Tz, Tm, XXg, Trf, pq)
    assert_allclose(expected_Sg, test_Sg)
