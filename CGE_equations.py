'''
This module defines the equations that characterize the CGE model.
'''


def eqpy(b, F, beta, Y):
    '''
    Production function.

    .. math::
        Y_{j} = b_{j}\prod_{h}F_{h,j}^{\beta_{h,j}}

    Args:
        b (1D numpy array): Scale parameter for each good j
        F (2D numpy array): The use of factor h in the production of
            good j
        beta (2D numpy array): Cost share parameter for factor h in
            production of good j
        Y (1D numpy array): Value added for each good j

    Returns:
        py_error (1D numpy array): The difference between Y and the
            production function evaluated at F.
    '''
    py_error = Y - b * (F ** beta).prod(axis=0)
    return py_error


def eqF(beta, py, Y, pf):
    '''
    Factor demand.

    .. math::
        F_{h,j} = \beta_{h,j}\frac{py_{j}}{pf_{h}}Y_{j}

    Args:
        beta (2D numpy array): Cost share parameter for factor h in
            production of good j
        py (1D array): The price of value added for each good j
        Y (1D numpy array): Value added for each good j
        pf (1D array): Price of each factor h

    Returns:
        F (2D numpy array): The demand for factor h used in the
            production of good j
    '''
    F = beta.div(pf, axis=0) * Y * py
    return F


def eqX(ax, Z):
    '''
    Demand for intermediate inputs.

    .. math::
        X_{i,j} = ax_{i,j}Z_{j}

    Args:
        ax (2D numpy array): Fixed proportions of intermeidate input i
            used in production of good j (Leontif production function)
        Z (1D numpy array): Output of industry j

    Returns:
        X (2D numpy array): Demand for intermediate input i used in the
            production of good j
    '''
    X = ax * Z
    return X


def eqY(ay, Z):
    '''
    Value added.

    .. math::
        Y_{j} = ay_{j}Z_{j}

    Args:
        ay (1D numpy array): Leontif production parameter, share of
            output of industry j in value added of good j
        Z (1D numpy array): Output of industry j

    Returns:
        Y (1D numpy array): Value added of good j
    '''
    Y = ay * Z
    return Y


def eqpz(ay, ax, py, pq):
    '''
    Output prices.

    .. math::
        pz_{j} = ay_{j}py_{j} + \sum_{i}ax_{i,j}pq_{i}

    Args:
        ay (1D numpy array): Leontif production parameter, share of
            output of industry j in value added of good j
        ax (2D numpy array): Fixed proportions of intermeidate input i
            used in production of good j (Leontif production function)
        py (1D numpy array): The price of value added for each good j
        pq (1D numpy array): price of XXXX for each good i

    Returns:
        pz (1D numpy array): price of output good j
    '''
    pz = ay * py + (ax * pq).sum(axis=0)
    return pz


def eqTd(taud, pf, Ff):
    '''
    Direct tax revenue.

    .. math::
        Td = \tau d \sum_{h}pf_{h}FF_{h}

    Args:
        taud (float): Direct tax rate
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h

    Returns:
        Td (float): Total direct tax revenue.
    '''
    Td = taud * (pf * Ff).sum()
    return Td


def eqTrf(tautr, pf, Ff):
    '''
    Total transfers to households.

    .. math::
        Trf = \tau^{tr} \sum_{h}pf_{h}FF_{h}

    Args:
        tautr (float): Tranfer rate (??)
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h

    Returns:
        Trf (float): Total transfers to households
    '''
    Trf = tautr * pf['LAB'] * Ff['LAB']
    return Trf


def eqTz(tauz, pz, Z):
    '''
    Production tax revenue from each commodity.

    .. math::
        Tz_{j} = \tau^{z}_{j} pz_{j}Z_{j}

    Args:
        tauz (1D numpy array): Ad valorem tax rate on commodity j
        pz (1D numpy array): price of output good j
        Z (1D numpy array): Output of industry j

    Returns:
        Tz (1D numpy array): Production tax revenue for each commodity j
    '''
    Tz = tauz * pz * Z
    return Tz


def eqTm(taum, pm, M):
    '''
    Tariff revenue from each commodity.

    .. math::
        Tm_{j} = \tau^{m}_{j} pm_{j}M_{j}

    Args:
        taum (1D numpy array): Tariff rate on commodity j
        pm (1D numpy array): price of import good j
        M (1D numpy array): Imports of good j

    Returns:
        Tm (1D numpy array): Tariff revenue for each commodity j
    '''
    Tm = taum * pm * M
    return Tm


def eqXg(mu, XXg):
    '''
    Government expenditures on commodity j

    .. math::
        X^{g}_{j} = \mu_{j}XX_{g}

    Args:
        mu (1D numpy array): Government expenditure share parameters for
            each commodity j
        XXg (float??): Total government spending on goods/services (??)

    Returns:
        Xg (1D numpy array): Government expenditures on commodity j
    '''
    Xg = mu * XXg.values
    return Xg


def eqXv(lam, XXv):
    '''
    Investment demand for each good j

    .. math::
        Xv_{j} = \lambda_{j}XXv

    Args:
        lam (1D numpy array): Fixed shares of investment for each good j
        XXv (float??): Total investment

    Returns:
        Xv = (1D numpy array): Investment demand for each good j
    '''
    Xv = lam * XXv.values
    return Xv


def eqXXv(g, Kk):
    '''
    Total investment.

    .. math::
        XXv = g \cdot KK

    Args:
        g (float): Exogenous long run growth rate of the economy
        KK (float): Total capital stock

    Returns:
        XXv (float): Total investment.
    '''
    XXv = g * Kk
    return XXv


def eqSp(ssp, pf, Ff, Fsh, Trf):
    '''
    Total household savings.

    .. math::
        Sp = ssp \cdot \left(\sum_{h}pf_{h}FF_{h} \right)

    Args:
        ssp (float): Fixed household savings rate
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h
        Fsh (float): Repatriated profits
        Trf (float): Total transfers to households

    Returns:
        Sp (float): Total household savings
    '''
    Sp = ssp * ((pf * Ff).sum() - Fsh + Trfs)
    return Sp


def eqSg(mu, Td, Tz, Tm, XXg, Trf, pq):
    '''
    Total government savings.

    .. math::
        Sg = Td + \sum_{j}Tz_{j} + \sum_{j}Tm_{j} - (Trf + \sum_{j}Xg_{j})

    Args:
    mu (1D numpy array): Government expenditure share parameters for
        each commodity j
    Td (float): Total direct tax revenue.
    Tz (1D numpy array): Production tax revenue for each commodity j
    Tm (1D numpy array): Tariff revenue for each commodity j
    XXg (float??): Total government spending on goods/services (??)
    Trf (float): Total transfers to households
    pq (1D numpy array): price of XXXX for each good i

    Returns:
        Sg (float): Total government savings
    '''
    Sg = Td + Tz.sum() + Tm.sum() - (Trf + XXg * (mu * pq).sum())
    return Sg


def eqFsh(R, Kf, er):
    '''
    Domestic profits that are repatriated to foreign owners of capital.

    .. math::
        FSH = R \cdot KF

    Args:
        R (float): Real return on capital
        Kf (float): Foreign holdings of domestic capital (??)
        er (??):

    Returns:
        Fsh = Repatriated profits
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
