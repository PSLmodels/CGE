#import packages
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#load social accounting matrix
sam = pd.read_excel('SAM.xlsx')

#declare sets
u = ('AGR', 'OIL', 'IND', 'SER', 'LAB', 'CAP', 'LAND', 'NTR',
     'DTX', 'IDT', 'ACT', 'HOH', 'GOV', 'INV', 'EXT' )
ind = ('AGR', 'OIL', 'IND', 'SER')
h = ('LAB', 'CAP', 'LAND', 'NTR')
w = ('LAB', 'LAND', 'NTR')






def cge_system(pvec, *args):
    '''
    This function solves the system of equations that represents the
    CGE model.

    Args:
        pvec (Numpy array): Vector of prices
        args (tuple): Tuple of arguments for equations

    Returns:
        p_error (Numpy array): Errors from CGE equations
    '''
    (eta, phi, alpha, beta, b, ax, ay, mu, lam, deltam, deltad, gamma, xid, xie,
     theta, ssp, taud, tautr, tauz, taum, g, pWe, pWm, Kk0, Ff0, XXg, R, ind, h, er, Z, Q, Kd, pd, Ff) = args

    py = pvec[0:len(ind)]
    pq = pvec[len(ind):len(ind)*2]
    pf = pvec[len(ind)*2:12]

    py = Series(py, index=list(ind))
    pq = Series(pq, index=list(ind))
    pf = Series(pf, index=list(h))

    pe = eqpe(er, pWe)
    pm = eqpm(er, pWm)

    pz = eqpz(ay, ax, py, pq)

    Kk = eqKk(pf, Ff, R, lam, pq)
    XXv = eqXXv(g, Kk)
    Xv = eqXv(lam, XXv)

    Xg = eqXg(mu, XXg)

    Kf = eqKf(Kk, Kd)
    Fsh = eqFsh(R, Kf, er)
    Sf = eqSf(g, lam, pq, Kf)

    Td = eqTd(taud, pf, Ff)
    Trf = eqTrf(tautr, pf, Ff)
    Tz = eqTz(tauz, pz, Z)

    X = eqX(ax, Z)
    Y = eqY(ay, Z)
    F = eqF(beta, py, Y, pf)

    Sp = eqSp(ssp, pf, Ff, Fsh, Trf)
    Xp = eqXp(alpha, pf, Ff, Sp, Td, Fsh, Trf, pq)

    E = eqE(theta, xie , tauz, phi, pz, pe, Z)
    D = eqD(theta, xid , tauz, phi, pz, pd, Z)

    M = eqM(gamma, deltam, deltad, eta, Q, pq, pm, taum)
    Tm = eqTm(taum, pm, M)
    Sg = eqSg(mu, Td, Tz, Tm, XXg, Trf, pq)

    pq_error = eqpq(Q, Xp, Xg, Xv, X)
    pf_error = eqpf(F, Ff0)
    pk_error = eqpk(F, Kk, Kk0, Ff0)
    py_error = eqpy(b, F, beta, Y)

    pf_error = pf_error.append(pk_error)
    pf_error = DataFrame(pf_error)
    pf_error = pf_error.T
    pf_error = DataFrame(pf_error, columns = list(h))
    pf_error = pf_error.iloc[0]

    py_error = py_error.values
    pf_error = pf_error.values
    pq_error = pq_error.values

    p_error = np.append(py_error, pq_error)
    p_error = np.append(p_error, pf_error)

    return p_error



#solve cge_system

dist = 10
tpi_iter = 0
tpi_max_iter = 30
tpi_tol = 1e-10
xi = 0.1

#pvec = pvec_init
pvec = np.ones(12)

XXg = XXg0
R = R0
er = 1

Zbar = Z0
Ffbar = Ff0
Kdbar = Kd0
Qbar = Q0
pdbar = pvec[0:len(ind)]

'''
#checking system
py = pvec[0:len(ind)]
pq = pvec[len(ind):len(ind)*2]
pf = pvec[len(ind)*2:12]

py = Series(py, index=list(ind))
pq = Series(pq, index=list(ind))
pf = Series(pf, index=list(h))

pe = eqpe(er, pWe)
pm = eqpm(er, pWm)

pz = eqpz(ay, ax, py, pq)

Kk = eqKk(pf, Ffbar, R, lam, pq)
XXv = eqXXv(g, Kk)
Xv = eqXv(lam, XXv)

Xg = eqXg(mu, XXg)

Kf = eqKf(Kk, Kdbar)
Fsh = eqFsh(R, Kf, er)
Sf = eqSf(g, lam, pq, Kf)

Td = eqTd(taud, pf, Ffbar)
Trf = eqTrf(tautr, pf, Ffbar)
Tz = eqTz(tauz, pz, Zbar)

X = eqX(ax, Zbar)
Y = eqY(ay, Zbar)
F = eqF(beta, py, Y, pf)

Sp = eqSp(ssp, pf, Ffbar, Fsh, Trf)
Xp = eqXp(alpha, pf, Ffbar, Sp, Td, Fsh, Trf, pq)

E = eqE(theta, xie , tauz, phi, pz, pe, Zbar)
D = eqD(theta, xid , tauz, phi, pz, pdbar, Zbar)

M = eqM(gamma, deltam, deltad, eta, Qbar, pq, pm, taum)
Tm = eqTm(taum, pm, M)
Sg = eqSg(mu, Td, Tz, Tm, XXg, Trf, pq)

pq_error = eqpq(Qbar, Xp, Xg, Xv, X)
pf_error = eqpf(F, Ff0)
pk_error = eqpk(F, Kk, Kk0, Ff0)
py_error = eqpy(b, F, beta, Y)


pf_error = pf_error.append(pk_error)
pf_error = DataFrame(pf_error)
pf_error = pf_error.T
pf_error = DataFrame(pf_error, columns = list(h))
pf_error = pf_error.iloc[0]
#pf_error.columns = list(ind)

py_error = py_error.values
pf_error = pf_error.values
pq_error = pq_error.values

p_error = np.append(py_error, pq_error)
p_error = np.append(p_error, pf_error)
#p_error = py_error.append([pq_error, pf_error])
#p_error = p_error.values
#---------------------------------------------
'''

while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
    tpi_iter += 1

    cge_args = (eta, phi, alpha, beta, b, ax, ay, mu, lam, deltam, deltad, gamma, xid, xie,
         theta, ssp, taud, tautr, tauz, taum, g, pWe, pWm, Kk0, Ff0, XXg, R, ind, h, er,
         Zbar, Qbar, Kdbar, pdbar, Ffbar)

    results = opt.root(cge_system, pvec, args=(cge_args),
                       method='lm', tol=1e-5)
    pprime = results.x


    pyprime = pprime[0:len(ind)]
    pqprime = pprime[len(ind):len(ind)*2]
    pfprime= pprime[len(ind)*2:12]

    pvec = pprime

    temp = cge_system(pvec,cge_args)

    Qprime = eqQ(gamma, deltam, deltad, eta, M, D)
    pdprime = eqpd(gamma, deltam, deltad, eta, Qprime, pqprime, D)
    Zprime = eqZ(theta, xie, xid, phi, E, D)
    Zprime = Zprime.iloc[0]
    Kdprime = eqKd(g, Sp, lam, pqprime)
    Ffprime = Ff0
    Ffprime['CAP'] = R * Kk * (lam * pq).sum() / pf[1]
'''
    dist = (((Zbar - Zprime) ** 2 ) ** (1 / 2)).sum()
    print('Distance at iteration ', tpi_iter, ' is ', dist)
    pdbar = xi * pdprime + (1 - xi) * pdbar
    Zbar = xi * Zprime + (1 - xi) * Zbar
    Kdbar = xi * Kdprime + (1 - xi) * Kdbar
    Qbar = xi * Qprime + (1 - xi) * Qbar
    Ffbar = xi * Ffprime + (1 - xi) * Ffbar
'''

'''
    bop_error = eqbop(pWe, pWm, E, M, Sf, Fsh, er)

    pd = eqpd(gamma, deltam, deltad, eta, Q, pq, D)
    Z = eqZ(theta, xie, xid, phi, E, D)
    Kd = eqKd(g, Sp, lam, pq)
    Q = eqQ(gamma, deltam, deltad, eta, M, D)
'''
