import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series
import os
from open_cge import government as gov
from open_cge import household as hh
from open_cge import aggregates as agg
from open_cge import firms
from open_cge import calibrate
from open_cge import simpleCGE as cge

# load social accounting matrix
current_path = os.path.abspath(os.path.dirname(__file__))
sam_path = os.path.join(current_path, 'SAM.xlsx')
sam = pd.read_excel(sam_path, index_col=0, header=0)

# declare sets
u = ('AGR', 'OIL', 'IND', 'SER', 'LAB', 'CAP', 'LAND', 'NTR',
     'DTX', 'IDT', 'ACT', 'HOH', 'GOV', 'INV', 'EXT')
ind = ('AGR', 'OIL', 'IND', 'SER')
h = ('LAB', 'CAP', 'LAND', 'NTR')
w = ('LAB', 'LAND', 'NTR')


def runner():
    '''
    this function runs the cge model
    '''

    # solve cge_system
    dist = 10
    tpi_iter = 0
    tpi_max_iter = 1000
    tpi_tol = 1e-10
    xi = 0.1

    # pvec = pvec_init
    pvec = np.ones(len(ind) + len(h))

    # Load data and parameters classes
    d = calibrate.model_data(sam, h, u, ind)
    p = calibrate.parameters(d, ind)

    R = d.R0
    er = 1

    Zbar = d.Z0
    Ffbar = d.Ff0
    Kdbar = d.Kd0
    Qbar = d.Q0
    pdbar = pvec[0:len(ind)]

    pm = firms.eqpm(er, d.pWm)

    while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
        tpi_iter += 1
        cge_args = [p, d, ind, h, Zbar, Qbar, Kdbar, pdbar, Ffbar, R, er]

        print("initial guess = ", pvec)
        results = opt.root(cge.cge_system, pvec, args=cge_args, method='lm',
                           tol=1e-5)
        pprime = results.x
        pyprime = pprime[0:len(ind)]
        pfprime = pprime[len(ind):len(ind) + len(h)]
        pyprime = Series(pyprime, index=list(ind))
        pfprime = Series(pfprime, index=list(h))

        pvec = pprime

        pe = firms.eqpe(er, d.pWe)
        pm = firms.eqpm(er, d.pWm)
        pq = firms.eqpq(pm, pdbar, p.taum, p.eta, p.deltam, p.deltad, p.gamma)
        pz = firms.eqpz(p.ay, p.ax, pyprime, pq)
        Kk = agg.eqKk(pfprime, Ffbar, R, p.lam, pq)
        Td = gov.eqTd(p.taud, pfprime, Ffbar)
        Trf = gov.eqTrf(p.tautr, pfprime, Ffbar)
        Kf = agg.eqKf(Kk, Kdbar)
        Fsh = firms.eqFsh(R, Kf, er)
        Sp = agg.eqSp(p.ssp, pfprime, Ffbar, Fsh, Trf)
        I = hh.eqI(pfprime, Ffbar, Sp, Td, Fsh, Trf)
        E = firms.eqE(p.theta, p.xie, p.tauz, p.phi, pz, pe, Zbar)
        D = firms.eqDex(p.theta, p.xid, p.tauz, p.phi, pz, pdbar, Zbar)
        M = firms.eqM(p.gamma, p.deltam, p.eta, Qbar, pq, pm, p.taum)
        Qprime = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)
        pdprime = firms.eqpd(p.gamma, p.deltam, p.eta, Qprime, pq, D)
        Zprime = firms.eqZ(p.theta, p.xie, p.xid, p.phi, E, D)
        #    Zprime = Zprime.iloc[0]
        Kdprime = agg.eqKd(d.g, Sp, p.lam, pq)
        Ffprime = d.Ff0
        # Ffprime['CAP'] = R * d.Kk * (p.lam * pq).sum() / pf[1]
        Ffprime['CAP'] = R * Kk * (p.lam * pq).sum() / pfprime[1]

        dist = (((Zbar - Zprime) ** 2) ** (1 / 2)).sum()
        print('Distance at iteration ', tpi_iter, ' is ', dist)
        pdbar = xi * pdprime + (1 - xi) * pdbar
        Zbar = xi * Zprime + (1 - xi) * Zbar
        Kdbar = xi * Kdprime + (1 - xi) * Kdbar
        Qbar = xi * Qprime + (1 - xi) * Qbar
        Ffbar = xi * Ffprime + (1 - xi) * Ffbar

        Q = firms.eqQ(p.gamma, p.deltam, p.deltad, p.eta, M, D)

    print('Model solved, Q = ', Q)

    return Q


if __name__ == "__main__":
    runner()
