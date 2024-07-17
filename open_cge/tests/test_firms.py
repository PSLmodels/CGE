## This file tests the functions in firms.py to check they return expected outputs.

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from open_cge import firms
from numpy.testing import assert_allclose


# Test production function output
def test_eqpy():
    b = np.array([0.6, 0.4])
    assert sum(b) == 1
    F = np.array([[2, 5], [6, 3]])
    beta = np.array([[0.3, 0.7], [0.6, 0.4]])
    Y = np.array([3, 2])
    expected_py_error = np.array([0.835533, 0.084917])
    test_py_error = firms.eqpy(b, F, beta, Y)
    assert_allclose(expected_py_error, test_py_error, rtol=1e-5)


# Test demand for intermediate inputs
def test_eqX():
    ax = np.array([[0.3, 0.7], [0.6, 0.4]])
    Z = np.array([20, 40])
    Z.reshape(2, 1)
    expected_X = np.array([[6, 28], [12, 16]])
    test_X = firms.eqX(ax, Z)
    assert_allclose(expected_X, test_X)


# Test value added
def test_eqY():
    ay = np.array([0.1, 0.3])
    Z = np.array([20, 30])
    expected_Y = np.array([2, 9])
    test_Y = firms.eqY(ay, Z)
    assert_allclose(expected_Y, test_Y)


# Test output prices
def test_eqpz():
    ay = np.array([0.2, 0.8])
    ax = np.array([[0.4, 0.6], [0.6, 0.4]])
    py = np.array([5, 7])
    pq = np.array([12, 16])
    expected_pz = np.array([13, 21.6])
    test_pz = firms.eqpz(ay, ax, py, pq)
    assert_allclose(expected_pz, test_pz)


# Test investment demand for each good
def test_eqXv():
    lam = np.array([0.4, 0.6])
    XXv = 100
    expected_Xv = np.array([40, 60])
    test_Xv = firms.eqXv(lam, XXv)
    assert_allclose(expected_Xv, test_Xv)


# Test repatriated profits
def test_eqFsh():
    R = 0.02
    Kf = 100.0
    er = 2.0
    espected_Fsh = 4.0
    test_Fsh = firms.eqFsh(R, Kf, er)
    assert_allclose(espected_Fsh, test_Fsh)


# Test export prices
def test_eqpe():
    er = 2.0
    pWe = np.array([2, 6, 12, 15])
    expected_pe = np.array([4, 12, 24, 30])
    test_pe = firms.eqpe(er, pWe)
    assert_allclose(expected_pe, test_pe)


# Test import prices
def test_eqpm():
    er = 2.0
    pWm = np.array([3, 7, 13, 22])
    expected_pm = np.array([6, 14, 26, 44])
    test_pm = firms.eqpm(er, pWm)
    assert_allclose(expected_pm, test_pm)


# Test CES production for importing firms
def test_eqQ():
    gamma = np.array([0.8, 0.5])
    deltad = np.array([0.6, 0.4])
    deltam = np.array([0.3, 0.7])
    assert sum(deltad) == 1
    assert sum(deltam) == 1
    eta = np.array([-0.6, -0.2])
    M = np.array([20, 10])
    D = np.array([10, 20])
    expected_Q = np.array([])
    test_Q = np.array([])
    assert_allclose(expected_Q, test_Q)


# Test demand for imports
def test_eqM():
    gamma = np.array([0.8, 0.5])
    deltam = np.array([0.3, 0.7])
    assert sum(deltam) == 1
    eta = np.array([-0.5, -0.2])
    Q = np.array([20.0, 40.0])
    pq = np.array([10.0, 6.0])
    pm = np.array([12.0, 8.0])
    taum = np.array([0.10, 0.05])
    expected_M = np.array([8.023519, 25.198421])
    test_M = firms.eqM(gamma, deltam, eta, Q, pq, pm, taum)
    assert_allclose(expected_M, test_M)


# Test demand for domestically produced goods from importers
def test_eqD():
    gamma = np.array([0.8, 0.5])
    deltad = np.array([0.6, 0.4])
    eta = np.array([-0.5, -0.2])
    Q = np.array([20.0, 20.0])
    pq = np.array([8.0, 5.0])
    pd = np.array([9.0, 6.0])
    expected_D = np.array([9, 6])
    test_D = firms.eqD(gamma, deltad, eta, Q, pq, pd)
    assert_allclose(expected_D, test_D)


# Test price of domestically produced goods from importers
def test_eqpd():
    gamma = np.array([0.8, 0.5])
    deltad = np.array([0.6, 0.4])
    eta = np.array([-0.5, -0.2])
    Q = np.array([20.0, 40.0])
    pq = np.array([8.0, 5.0])
    D = np.array([18.0, 30.0])
    expected_pd = np.array([6.285394, 3.24461])
    test_pd = firms.eqpd(gamma, deltad, eta, Q, pq, D)
    assert_allclose(expected_pd, test_pd)


# Test exporting firm production function
def test_eqZ():
    theta = np.array([0.8, 0.5])
    xie = np.array([0.2, 0.3])
    xid = np.array([0.2, 0.8])
    phi = np.array([-0.6, 0.8])
    E = np.array([20, 30])
    D = np.array([75, 150])
    expected_Z = np.array([125.572867, 64.176968])
    test_Z = firms.eqZ(theta, xie, xid, phi, E, D)
    assert_allclose(expected_Z, test_Z)


# Test supply of exports
def test_eqE():
    theta = np.array([0.8, 0.5])
    xie = np.array([0.2, 0.3])
    tauz = np.array([0.09, 0.06])
    phi = np.array([-0.4, -0.8])
    pz = np.array([10, 15])
    pe = np.array([20, 30])
    Z = np.array([125, 64])
    expected_E = np.array([27.35559, 31.354712])
    test_E = firms.eqE(theta, xie, tauz, phi, pz, pe, Z)
    assert_allclose(expected_E, test_E)


# Test demand for domestic goods by exporters
def test_eqDex():
    theta = np.array([0.8, 0.5])
    xid = np.array([0.3, 0.2])
    tauz = np.array([0.09, 0.06])
    phi = np.array([-0.4, -0.8])
    pz = np.array([10, 15])
    pd = np.array([12, 13])
    Z = np.array([50, 100])
    expected_Dex = np.array([21.054694, 62.238608])
    test_Dex = firms.eqDex(theta, xid, tauz, phi, pz, pd, Z)
    assert_allclose(expected_Dex, test_Dex)
