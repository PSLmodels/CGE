'''
This module defines the equations that characterize the CGE model.
'''


def eqpy(b, F, beta, Y):
    '''
    Production function constraint.

    Args:
        b (float):
        F (float):
        beta (float):
        Y (float): Output

    Returns:
        py_error (float): The difference between Y and the production
            function evaluated at F.
    '''
    py_error = Y - b * (F ** beta).prod(axis=0)
    return py_error


def eqF(beta, py, Y, pf):
    '''

    Args:

    Returns:

    '''
    F = beta.div(pf, axis=0) * Y * py
    return F


def eqX(ax, Z):
    '''

    Args:

    Returns:

    '''
    X = ax * Z
    return X


def eqY(ay, Z):
    '''

    Args:

    Returns:

    '''
    Y = ay * Z
    return Y


def eqpz(ay, ax, py, pq):
    '''

    Args:

    Returns:

    '''
    pz = ay * py + (ax * pq).sum(axis=0)
    return pz


def eqTd(taud, pf, Ff):
    '''

    Args:

    Returns:

    '''
    Td = taud * (pf * Ff).sum()
    return Td


def eqTrf(tautr, pf, Ff):
    '''

    Args:

    Returns:

    '''
    Trf = tautr * pf['LAB'] * Ff['LAB']
    return Trf


def eqTz(tauz, pz, Z):
    '''

    Args:

    Returns:

    '''
    Tz = tauz * pz * Z
    return Tz


def eqTm(taum, pm, M):
    '''

    Args:

    Returns:

    '''
    Tm = taum * pm * M
    return Tm


def eqXg(mu, XXg):
    '''

    Args:

    Returns:

    '''
    Xg = mu * XXg.values
    return Xg


def eqXv(lam, XXv):
    '''

    Args:

    Returns:

    '''
    Xv = lam * XXv.values
    return Xv


def eqXXv(g, Kk):
    '''

    Args:

    Returns:

    '''
    XXv = g * Kk
    return XXv


def eqSp(ssp, pf, Ff, Fsh, Trf):
    '''

    Args:

    Returns:

    '''
    Sp = ssp * ( (pf * Ff).sum() - Fsh + Trf )
    return Sp


def eqSg(mu, Td, Tz, Tm, XXg, Trf, pq):
    '''

    Args:

    Returns:

    '''
    Sg = Td + Tz.sum() + Tm.sum() - (Trf + XXg * (mu * pq).sum())
    return Sg


def eqFsh(R, Kf, er):
    '''

    Args:

    Returns:

    '''
    Fsh = R * Kf * er
    return Fsh


def eqKd(g, Sp, lam, pq):
    '''

    Args:

    Returns:

    '''
    Kd = Sp / (g * (lam * pq).sum())
    return Kd


def eqKf(Kk, Kd):
    '''

    Args:

    Returns:

    '''
    Kf = Kk - Kd
    return Kf


def eqKk(pf, Ff, R, lam, pq):
    '''

    Args:

    Returns:

    '''
#    R = ( (pf['CAP'] * Ff['CAP']) / Kk) / ((lam * pq).sum())
    Kk = (pf['CAP'] * Ff['CAP']) / (R * ( (lam * pq).sum() ))
    return Kk


def eqXp(alpha, pf, Ff, Sp, Td, Fsh, Trf, pq):
    '''

    Args:

    Returns:

    '''
    Xp = alpha * ((pf * Ff).sum() - Sp - Td - Fsh + Trf) / pq
    return Xp


def eqpe(er, pWe):
    '''

    Args:

    Returns:

    '''
    pe = er * pWe
    return pe


def eqpm(er, pWm):
    '''

    Args:

    Returns:

    '''
    pm = er * pWm
    return pm


def eqbop(pWe, pWm, E, M, Sf, Fsh, er):
    '''

    Args:

    Returns:

    '''
    bop_error = (pWe * E).sum() + Sf / er - ( (pWm * M).sum() + Fsh / er)
    return bop_error


def eqSf(g, lam, pq, Kf):
    '''

    Args:

    Returns:

    '''
    Sf = g * Kf * (lam * pq).sum()
    return Sf


def eqQ(gamma, deltam, deltad, eta, M, D):
    '''

    Args:

    Returns:

    '''
    Q = gamma * (deltam * M ** eta + deltad * D ** eta) ** (1 / eta)
    return Q


def eqM(gamma, deltam, deltad, eta, Q, pq, pm, taum):
    '''

    Args:

    Returns:

    '''
    M = (gamma ** eta * deltam * pq / ((1 + taum) * pm)) ** (1 / (1 - eta)) * Q
    return M


# def eqD(gamma, deltam, deltad, eta, Q, pq, pd):
#     '''
#
#     Args:
#
#     Returns:
#
#     '''
#     D = (gamma ** eta * deltad * pq / pd) ** (1 / (1 - eta)) * Q
#     return pd


def eqpd(gamma, deltam, deltad, eta, Q, pq, D):
    '''

    Args:

    Returns:

    '''
    pd = (gamma ** eta * deltad * pq) * (D / Q) ** (eta - 1)
    return pd


def eqZ(theta, xie, xid, phi, E, D):
    '''

    Args:

    Returns:

    '''
    Z = theta * (xie * E ** phi + xid * D ** phi) ** (1 / phi)
    return Z


def eqE(theta, xie , tauz, phi, pz, pe, Z):
    '''

    Args:

    Returns:

    '''
    E = (theta ** phi * xie * (1 + tauz) * pz / pe) ** (1 / (1 - phi)) * Z
    return E


def eqD(theta, xid , tauz, phi, pz, pd, Z):
    '''

    Args:

    Returns:

    '''
    D = (theta ** phi * xid * (1 + tauz) * pz / pd) ** (1 / (1 - phi)) * Z
    return D


def eqpq(Q, Xp, Xg, Xv, X):
    '''

    Args:

    Returns:

    '''
    pq_error = Q - (Xp + Xg + Xv + X.sum(axis=1))
    return pq_error


def eqpf(F, Ff0):
    '''

    Args:

    Returns:

    '''
    F1 = F.drop(['CAP'])
    Ff1 = Ff0.drop(['CAP'])
    pf_error = Ff1 - F1.sum(axis=1)
    return pf_error


def eqpk(F, Kk, Kk0, Ff0):
    '''

    Args:

    Returns:

    '''
    Fcap = F.loc[['CAP']]
    pk_error = Fcap.sum(axis=1) - Kk / Kk0 * Ff0['CAP']
    return pk_error
