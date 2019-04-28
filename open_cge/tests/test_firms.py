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
    assert_allclose(expected_py_error, test_py_error, rtol = 1e-5)

# Test demand for intermediate inputs
def test_eqX(ax, Z):
    ax = np.array([[0.3, 0.7], [0.6, 0.4]])
    Z = np.array([20, 40])
    expected_X = np.array([6, 2])
    test_X = firms.eqX(ax, Z)
    assert_allclose(expected_X, test_X)

# Test value added
def test_eqY(ay, Z):
    ay = np.array([0.1, 0.3])
    Z = np.array([20, 30])
    expected_Y = np.array([2, 9])
    test_Y = firms.eqY(ay, Z)
    assert_allclose(expected_Y, test_Y)

# Test output prices
def test_eqpz():
    ay =  np.array([0.2, 0.8])
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


# Test export prices


# Test import prices


# Test CES production for importing firms


# Test demand for domestically produced goods from importers


# Test exporting firm production function


# Test supply of exports
