# import packages
import numpy as np
import pandas
from pandas import Series, DataFrame
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms


def cge_system(pvec, args):
    """
    This function solves the system of equations that represents the
    CGE model.

    Args:
        pvec (Numpy array): Vector of prices
        args (tuple): Tuple of arguments for equations

    Returns:
        p_error (Numpy array): Errors from CGE equations
    """
    (p, d, ind, h, Z, Q, Kd, pd, Ff, R, er) = args

    py = pvec[0 : len(ind)]
    pf = pvec[len(ind) : len(ind) + len(h)]
    py = Series(py, index=list(ind))
    pf = Series(pf, index=list(h))

    pm = firms.eqpm(er, d.pWm)
    pq = firms.eqpq(pm, pd, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
    Kk = agg.eqKk(pf, Ff, R, p.lam, pq)
    Kf = agg.eqKf(Kk, Kd)
    Fsh = firms.eqFsh(R, Kf, er)
    Td = gov.eqTd(p.taud, pf, Ff)
    Trf = gov.eqTrf(p.tautr, pf, Ff)
    Y = firms.eqY(p.ay, Z)
    F = hh.eqF(p.beta, py, Y, pf)
    Sp = agg.eqSp(p.ssp, pf, Ff, Fsh, Trf)
    I = hh.eqI(pf, Ff, Sp, Td, Fsh, Trf)

    pf_error = agg.eqpf(F, d.Ff0)
    pk_error = agg.eqpk(F, Kk, d.Kk0, d.Ff0)
    py_error = firms.eqpy(p.b, F, p.beta, Y)

    pf_error = pandas.concat((pf_error, pk_error))
    pf_error = DataFrame(pf_error)
    pf_error = pf_error.T
    pf_error = DataFrame(pf_error, columns=list(h))
    pf_error = pf_error.iloc[0]

    py_error = py_error.values
    pf_error = pf_error.values

    p_error = np.append(py_error, pf_error)

    return p_error
