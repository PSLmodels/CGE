## This file tests the functions in firms.py to check they return expected outputs.

import pytest
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from open_cge import firms

# Test production function output
def test_eqpy():
    b = np.array([0.3, 0.3, 0.4])
    F = np.array([], [], [])
    beta = np.array([], [], [])
    Y = np.array([])

# Test demand for intermediate inputs
def test_eqX(ax, Z):
    ax = np.array([], [], [])


# Test value added


# Test output prices


# Test investment demand for each good


# Test repatriated profits


# Test export prices


# Test import prices


# Test CES production for importing firms


# Test demand for domestically produced goods from importers


# Test exporting firm production function


# Test supply of exports
