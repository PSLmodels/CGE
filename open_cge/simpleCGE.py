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


def cge_system(pvec, args):
    '''
    This function solves the system of equations that represents the
    CGE model.

    Args:
        pvec (Numpy array): Vector of prices
        args (tuple): Tuple of arguments for equations

    Returns:
        p_error (Numpy array): Errors from CGE equations
    '''
    (p, d, ind, h, Z, Q, Kd, pd, Ff, R, er, Sg) = args

    py = pvec[0:len(ind)]
    pf = pvec[len(ind): len(ind) + len(h)]
    py = Series(py, index=list(ind))
    pf = Series(pf, index=list(h))

    pe = eq.eqpe(er, d.pWe)
    pm = eq.eqpm(er, d.pWm)
    pq = eq.eqpq(pm, pd, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
    pz = eq.eqpz(p.ay, p.ax, py, pq)
    Kk = eq.eqKk(pf, Ff, R, p.lam, pq)
    XXv = eq.eqXXv(d.g, Kk)
    Xv = eq.eqXv(p.lam, XXv)
    Kf = eq.eqKf(Kk, Kd)
    Fsh = eq.eqFsh(R, Kf, er)
    Td = eq.eqTd(p.taud, pf, Ff)
    Trf = eq.eqTrf(p.tautr, pf, Ff)
    Tz = eq.eqTz(p.tauz, pz, Z)
    X = eq.eqX(p.ax, Z)
    Y = eq.eqY(p.ay, Z)
    F = eq.eqF(p.beta, py, Y, pf)
    Sp = eq.eqSp(p.ssp, pf, Ff, Fsh, Trf)
    Xp = eq.eqXp(p.alpha, pf, Ff, Sp, Td, Fsh, Trf, pq)
    E = eq.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Z)
    D = eq.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pd, Z)
    M = eq.eqM(p.gamma, p.deltam, p.deltad, p.eta, Q, pq, pm, p.taum)
    Tm = eq.eqTm(p.taum, pm, M)
    XXg = eq.eqSg(p.mu, Td, Tz, Tm, Sg, Trf, pq)
    Xg = eq.eqXg(p.mu, XXg)

    pf_error = eq.eqpf(F, Ff)
    pk_error = eq.eqpk(F, Kk, d.Kk0, d.Ff0)
    py_error = eq.eqpy(p.b, F, p.beta, Y)

    pf_error = pf_error.append(pk_error)
    pf_error = DataFrame(pf_error)
    pf_error = pf_error.T
    pf_error = DataFrame(pf_error, columns=list(h))
    pf_error = pf_error.iloc[0]

    py_error = py_error.values
    pf_error = pf_error.values

    p_error = np.append(py_error, pf_error)

    return p_error


# solve cge_system
dist = 10
tpi_iter = 0
tpi_max_iter = 500
tpi_tol = 1e-5
xi = 0.15

# pvec = pvec_init
pvec = np.ones(len(ind) + len(h))

# Load data and parameters classes
d = calibrate.model_data(sam, h, u, ind)
p = calibrate.parameters(d, ind)


R = d.R0
Sg = d.Sg0
er = 1
p.taum = p.taum * 1.1

Zbar = d.Z0 * 1
Ffbar = d.Ff0 * 1
Kdbar = d.Kd0 * 1
Qbar = d.Q0 * 1
pdbar = pvec[0:len(ind)]


'''
#checking calibration of model
#cge_args = [p, d, ind, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er, Sg]
#errors = cge_system(pvec, cge_args)
#---------------------------------------------

py = pvec[0:len(ind)]
pf = pvec[len(ind): len(ind) + len(h)]
py = Series(py, index=list(ind))
pf = Series(pf, index=list(h))
pe = eq.eqpe(er, d.pWe)
pm = eq.eqpm(er, d.pWm)
pq = eq.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
pz = eq.eqpz(p.ay, p.ax, py, pq)
Kk = eq.eqKk(pf, Ffbar, R, p.lam, pq)
XXv = eq.eqXXv(d.g, Kk)
Xv = eq.eqXv(p.lam, XXv)
Kf = eq.eqKf(Kk, Kdbar)
Fsh = eq.eqFsh(R, Kf, er)
Td = eq.eqTd(p.taud, pf, Ffbar)
Trf = eq.eqTrf(p.tautr, pf, Ffbar)
Tz = eq.eqTz(p.tauz, pz, Zbar)
X = eq.eqX(p.ax, Zbar)
Y = eq.eqY(p.ay, Zbar)
F = eq.eqF(p.beta, py, Y, pf)
Sp = eq.eqSp(p.ssp, pf, Ffbar, Fsh, Trf)
Xp = eq.eqXp(p.alpha, pf, Ffbar, Sp, Td, Fsh, Trf, pq)
E = eq.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar)
D = eq.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar)
M = eq.eqM(p.gamma, p.deltam, p.deltad, p.eta, Qbar, pq, pm, p.taum)
Tm = eq.eqTm(p.taum, pm, M)
XXg = eq.eqSg(p.mu, Td, Tz, Tm, Sg, Trf, pq)
Xg = eq.eqXg(p.mu, XXg)

'''

while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
	tpi_iter += 1
	cge_args = [p, d, ind, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er, Sg]

	results = opt.root(cge_system, pvec, args=cge_args, method='lm',
					   tol=1e-5)
	pprime = results.x
	pyprime = pprime[0:len(ind)]
	pfprime = pprime[len(ind):len(ind) + len(h)]
	pyprime = Series(pyprime, index=list(ind))
	pfprime = Series(pfprime, index=list(h))

	pvec = pprime

	temp = cge_system(pvec, cge_args)

	pe = eq.eqpe(er, d.pWe)
	pm = eq.eqpm(er, d.pWm)
	pq = eq.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
	pz = eq.eqpz(p.ay, p.ax, pyprime, pq)
	Kk = eq.eqKk(pfprime, Ffbar, R, p.lam, pq)
	Td = eq.eqTd(p.taud, pfprime, Ffbar)
	Trf = eq.eqTrf(p.tautr, pfprime, Ffbar)
	Tz = eq.eqTz(p.tauz, pz, Zbar)
	Kf = eq.eqKf(Kk, Kdbar)
	Fsh = eq.eqFsh(R, Kf, er)
	Sf = eq.eqSf(d.g, p.lam, pq, Kf)
	Sp = eq.eqSp(p.ssp, pfprime, Ffbar, Fsh, Trf)
	Xp = eq.eqXp(p.alpha, pfprime, Ffbar, Sp, Td, Fsh, Trf, pq)
	E = eq.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar)
	D = eq.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar)
	M = eq.eqM(p.gamma, p.deltam, p.deltad, p.eta, Qbar, pq, pm, p.taum)
	X = eq.eqX(p.ax, Zbar)
	Tm = eq.eqTm(p.taum, pm, M)
	XXg = eq.eqSg(p.mu, Td, Tz, Tm, Sg, Trf, pq)
	Xg = eq.eqXg(p.mu, XXg)
	XXv = eq.eqXXv(d.g, Kk)
	Xv = eq.eqXv(p.lam, XXv)
	Qprime = eq.eqpqerror(Xp, Xg, Xv, X)
	pdprime = eq.eqpd(p.gamma, p.deltam, p.deltad, p.eta, Qprime, pq, D)
	Zprime = eq.eqZ(p.theta, p.xie, p.xid, p.phi, E, D)
	Kdprime = eq.eqKd(d.g, Sp, p.lam, pq)
	Ffprime = Ffbar
	Ffprime['CAP'] = R * Kk * (p.lam * pq).sum() / pfprime[1]

	dist = (((Qbar - Qprime) ** 2 ) ** (1 / 2) + ((pdbar - pdprime) ** 2 ) ** (1 / 2)).sum()
	print('Distance at iteration ', tpi_iter, ' is ', dist)

	pdbar = xi * pdprime + (1 - xi) * pdbar
	Zbar = xi * Zprime + (1 - xi) * Zbar
	Kdbar = xi * Kdprime + (1 - xi) * Kdbar
	Qbar = xi * Qprime + (1 - xi) * Qbar
	Ffbar = xi * Ffprime + (1 - xi) * Ffbar



bop_error = eq.eqbop(d.pWe, d.pWm, E, M, Sf, Fsh, er)


print('Model solved, Q = ', Qprime)

