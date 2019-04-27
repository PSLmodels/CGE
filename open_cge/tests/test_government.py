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


# Tariff revenue from each commodity


# Government expenditures on commodity j


# Total government savings
