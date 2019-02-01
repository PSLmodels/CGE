# import packages
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
current_path = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.insert(0, current_path)
import equations as eq
import calibrate

# load social accounting matrix
sam_path = os.path.join(current_path, 'SAM.xlsx')
sam = pd.read_excel(sam_path)

# declare sets
u = ('AGR', 'OIL', 'IND', 'SER', 'LAB', 'CAP', 'LAND', 'NTR',
     'DTX', 'IDT', 'ACT', 'HOH', 'GOV', 'INV', 'EXT')
ind = ('AGR', 'OIL', 'IND', 'SER')
h = ('LAB', 'CAP', 'LAND', 'NTR')
w = ('LAB', 'LAND', 'NTR')


def io_system(pvec, args):
    '''
    This function solves the system of equations that represents the
    input output pricing model.

    Args:
        pvec (Numpy array): Vector of prices
        args (tuple): Tuple of arguments for equations

    Returns:
        p_error (Numpy array): Errors from IO equations
    '''
    (p, d, ind, h, er, pf) = args
	
    pq = pvec[0:len(ind)]
    pq = Series(pq, index=list(ind))
    pm = eq.eqpm(er, d.pWm)
    py = eq.eqpy(pf, p.beta)
    pz = eq.eqpz(p.ay, p.ax, py, pq)
    pq_error = eq.eqpq(p.deltam, p.taum, p.tauz, pm, pz, pq)
    pq_error = pq_error.values
	
    return pq_error


# solve io_system
dist = 10
tpi_iter = 0
tpi_max_iter = 30
tpi_tol = 1e-10
xi = 0.1

# pvec = pvec_init
pvec = np.ones(len(ind))


# Load data and parameters classes
d = calibrate.model_data(sam, h, u, ind)
p = calibrate.parameters(d, ind)

# Exogenous variables
XXg = d.XXg0
er = 1
pf = np.ones(len(h))
pf = Series(pf, index=list(h))
pf = pf * 1


'''
#checking calibration of model
io_args = [p, d, ind, h, er, pf]
errors = io_system(pvec, io_args)
#---------------------------------------------
'''



while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
    tpi_iter += 1
    io_args = [p, d, ind, h, er, pf]

    results = opt.root(io_system, pvec, args=io_args, method='lm',
                       tol=1e-5)
    pprime = results.x
    pqprime = pprime[0:len(ind)]
    pqprime = Series(pqprime, index=list(ind))
	
    pvec = pprime
    pq_error = io_system(pvec, io_args)

    dist = (((pq_error) ** 2 ) ** (1 / 2)).sum()
    print('Distance at iteration ', tpi_iter, ' is ', dist)


pm = eq.eqpm(er, d.pWm)	
py = eq.eqpy(pf, p.beta)
pz = eq.eqpz(p.ay, p.ax, py, pqprime)
pqbar = eq.eqpqbar(p.deltam, p.taum, p.tauz, pm, pz)
   
print('Model solved, pq = ', pqbar)

