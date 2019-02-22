# import packages
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os
current_path = os.path.abspath(os.path.dirname(__file__))
import sys
sys.path.insert(0, current_path)
# import equations as eq
import government as gov
import household as hh
import aggregates as agg
import firms
import calibrate


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
    (p, d, ind, h, Z, Q, Kd, pd, Ff, R, er, XXg) = args

    py = pvec[0:len(ind)]
    pf = pvec[len(ind): len(ind) + len(h)]
    py = Series(py, index=list(ind))
    pf = Series(pf, index=list(h))

    pe = firms.eqpe(er, d.pWe)
    pm = firms.eqpm(er, d.pWm)
    pq = firms.eqpq(pm, pd, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
    pz = firms.eqpz(p.ay, p.ax, py, pq)
    Kk = agg.eqKk(pf, Ff, R, p.lam, pq)
    XXv = agg.eqXXv(d.g, Kk)
    Xv = firms.eqXv(p.lam, XXv)
    Xg = agg.eqXg(p.mu, XXg)
    Kf = agg.eqKf(Kk, Kd)
    Fsh = firms.eqFsh(R, Kf, er)
    Td =gov.eqTd(p.taud, pf, Ff)
    Trf = gov.eqTrf(p.tautr, pf, Ff)
    Tz = gov.eqTz(p.tauz, pz, Z)
    X = firms.eqX(p.ax, Z)
    Y = firms.eqY(p.ay, Z)
    F = hh.eqF(p.beta, py, Y, pf)
    Sp = agg.eqSp(p.ssp, pf, Ff, Fsh, Trf)
    Xp = hh.eqXp(p.alpha, pf, Ff, Sp, Td, Fsh, Trf, pq)
    E = firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Z)
    D = firms.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pd, Z)
    M = firms.eqM(p.gamma, p.deltam, p.deltad, p.eta, Q, pq, pm, p.taum)
    Tm = gov.eqTm(p.taum, pm, M)


    pf_error = agg.eqpf(F, d.Ff0)
    pk_error = agg.eqpk(F, Kk, d.Kk0, d.Ff0)
    py_error = firms.eqpy(p.b, F, p.beta, Y)

    pf_error = pf_error.append(pk_error)
    pf_error = DataFrame(pf_error)
    pf_error = pf_error.T
    pf_error = DataFrame(pf_error, columns=list(h))
    pf_error = pf_error.iloc[0]

    py_error = py_error.values
    pf_error = pf_error.values

    p_error = np.append(py_error, pf_error)

    return p_error
