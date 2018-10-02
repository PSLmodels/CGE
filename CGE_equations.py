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
        Xv (1D numpy array): Investment demand for each good j
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
        Kk (float): Total capital stock

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
        Td (float): Total direct tax revenue
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
        FSH = R \cdot KF \cdot \varepsilon

    Args:
        R (float): Real return on capital
        Kf (float): Foreign holdings of domestic capital (??)
        er (float): The real exchange rate

    Returns:
        Fsh = Repatriated profits
    '''
    Fsh = R * Kf * er
    return Fsh


def eqKd(g, Sp, lam, pq):
    '''
    Domestic capital holdings.

    .. math::
        K^{d} = \frac{S^{p}}{g\sum_{j}\lambda_{j}pq_{j}}

    Args:
        g (float): Exogenous long run growth rate of the economy
        Sp (float): Total household savings
        lam (1D numpy array): Fixed shares of investment for each good j
        pq (1D numpy array): price of XXXX for each good i

    Returns:
        Kd (float): Domestically owned capital ??
    '''
    Kd = Sp / (g * (lam * pq).sum())
    return Kd


def eqKf(Kk, Kd):
    '''
    Foreign holdings of domestically used capital.

    .. math::
        K^{f} = KK - K^{d}

    Args:
        Kk (float): Total capital stock
        Kd (float): Domestically owned capital ??

    Returns:
        Kf (float): Foreign owned capital ??
    '''
    Kf = Kk - Kd
    return Kf


def eqKk(pf, Ff, R, lam, pq):
    '''
    Capital market clearing equation.

    .. math::
        KK = \frac{pf * FF}{R \sum_{j}\lambda_{j}pq_{j}}

    Args:
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h
        R (float): Real return on capital
        lam (1D numpy array): Fixed shares of investment for each good j
        pq (1D numpy array): price of XXXX for each good i

    Returns:
        Kk (float): Total capital stock
    '''
#    R = ( (pf['CAP'] * Ff['CAP']) / Kk) / ((lam * pq).sum())
    Kk = (pf['CAP'] * Ff['CAP']) / (R * ((lam * pq).sum()))
    return Kk


def eqXp(alpha, pf, Ff, Sp, Td, Fsh, Trf, pq):
    '''
    Demand for Xp.

    .. math::
        X^{p}_{i}= \frac{}\alpha_{i}}{pq_{i}}\left(\sum_{h}pf_{h}Ff_{h} - S^{p} - T^{d}- FSH - TRF\right)

    Args:
        alpha (1D numpy array): Budget share of good i
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h
        Sp (float): Total household savings
        Td (float): Total direct tax revenue
        Fsh = Repatriated profits
        Trf (float): Total transfers to households
        pq (1D numpy array): price of XXXX for each good i

    Returns:
        Xp (1D numpy array): Demand for production good i
    '''
    Xp = alpha * ((pf * Ff).sum() - Sp - Td - Fsh + Trf) / pq
    return Xp


def eqpe(er, pWe):
    '''
    Export prices.

    .. math::
        pe_{i} = \varepsilon \cdot pWe_{i}

    Args:
        er (float): The real exchange rate
        pWe (1D numpy array): The world price of commodity i in foreign currency

    Returns:
        pe (1D numpy array): Price of commodity i exports in domestic currency
    '''
    pe = er * pWe
    return pe


def eqpm(er, pWm):
    '''
    Import prices.

    .. math::
        pm_{i} = \varepsilon \cdot pWm_{i}

    Args:
        er (float): The real exchange rate
        pWm (1D numpy array): The world price of commodity i in foreign currency.

    Returns:
        pm (1D numpy array): The price of commodity i imports in domestic currency.
    '''
    pm = er * pWm
    return pm


def eqbop(pWe, pWm, E, M, Sf, Fsh, er):
    '''
    Balance of payments.

    .. math::
        \sum_{i}pWe_{i}E_{i} + \frac{Sf}{\varepsilon} = \sum_{i}pWm_{i}M_{i} + \frac{Fsh}{\varepsilon}

    Args:
        pWe (1D numpy array): The world price of commodity i in foreign currency
        pWm (1D numpy array): The world price of commodity i in foreign currency.
        E (1D numpy array): Exports of commodity i
        M (1D numpy array): Imports of commodity i
        Sf (float): Total foreign savings (??)
        Fsh = Repatriated profits
        er (float): The real exchange rate

    Returns:
        bop_error (float): Error in balance of payments equation.

    '''
    bop_error = (pWe * E).sum() + Sf / er - ((pWm * M).sum() + Fsh / er)
    return bop_error


def eqSf(g, lam, pq, Kf):
    '''
    Net foreign investment/savings.

    .. math::
        Sf = g Kf \sum_{j} \lambda_{j} pq_{j}

    Args:
        g (float): Exogenous long run growth rate of the economy
        lam (1D numpy array): Fixed shares of investment for each good j
        pq (1D numpy array): price of XXXX for each good i
        Kf (float): Foreign owned capital ??

    Returns:
        Sf (float): Total foreign savings (??)
    '''
    Sf = g * Kf * (lam * pq).sum()
    return Sf


def eqQ(gamma, deltam, deltad, eta, M, D):
    '''
    CES production function for the importing firm.

    .. math::
        Q_{i} = \gamma_{i}\left[\delta^{m}_{i}M^{\eta_{i}}_{i} + \delta^{d}_{i}D^{\eta_{i}}_{i}\right]^{\frac{1}{\eta_{i}}}

    Args:
        gamma (1D numpy array): ??
        deltam (1D numpy array): Share parameter for use of imports of good i in produciton Armington good i
        deltad (1D numpy array): Share parameter for use of domestically produced good i in produciton Armington good i
        eta (1D numpy array): The elasticity of substitution between imports and domestically supplied good i
        M (1D numpy array): The imports of good i
        D (1D numpy array): The domestic supply of good i from domestic production

    Returns:
        Q (1D numpy array): The domestic supply of good i, the Armington good
    '''
    Q = gamma * (deltam * M ** eta + deltad * D ** eta) ** (1 / eta)
    return Q


def eqM(gamma, deltam, deltad, eta, Q, pq, pm, taum):
    '''
    Demand for imports.

    .. math::
        M_{i} = \left(\gamma^{\eta_{i}}_{i}\delta^{m}_{i}\frac{pq_{i}}{(1+\tau^{m}_{i})pm_{i}}\right)^{\frac{1}{1-\eta_{i}}}Q_{i}

    Args:
        gamma (1D numpy array): ??
        deltam (1D numpy array): Share parameter for use of imports of good i in produciton Armington good i
        deltad (1D numpy array): Share parameter for use of domestically produced good i in produciton Armington good i
        eta (1D numpy array): The elasticity of substitution between imports and domestically supplied good i
        Q (1D numpy array): The domestic supply of good i, the Armington good
        pq (1D numpy array): price of XXXX for each good i
        pm (1D numpy array): The price of commodity i imports in domestic currency.
        taum (1D numpy array): Tariff rate on commodity i

    Returns:
        M (1D numpy array): Demand for imports of good i
    '''
    M = (gamma ** eta * deltam * pq / ((1 + taum) * pm)) ** (1 / (1 - eta)) * Q
    return M


def eqD(gamma, deltam, deltad, eta, Q, pq, pd):
    '''
    Demand for domestically produced goods from importers.

    .. math::
        D_{i} = \left(\gamma_{i}^{\eta_{i}}\delta^{d}_{i}\frac{pq_{i}}{pd_{i}}\right)^{\frac{1}{1-\eta_{i}}}Q_{i}

    Args:
        gamma (1D numpy array): ??
        deltam (1D numpy array): Share parameter for use of imports of good i in produciton Armington good i
        deltad (1D numpy array): Share parameter for use of domestically produced good i in produciton Armington good i
        eta (1D numpy array): The elasticity of substitution between imports and domestically supplied good i
        Q (1D numpy array): The domestic supply of good i, the Armington good
        pq (1D numpy array): price of XXXX for each good i
        pd (1D numpy array): price of domesically produced good i

    Returns:
        D (1D numpy array): Demand for domestically produced good i from importers
    '''
    D = (gamma ** eta * deltad * pq / pd) ** (1 / (1 - eta)) * Q
    return pd


def eqpd(gamma, deltam, deltad, eta, Q, pq, D):
    '''
    Price of domestically produced goods from importers.

    .. math::
        pd_{i} = \left(\gamma_{i}^{\eta_{i}}\delta^{d}_{i}pq_{i}\right)\left(\frac{D_{i}}{Q_{i}}\right)^{\eta_{i}-1}

    Args:
        gamma (1D numpy array): ??
        deltam (1D numpy array): Share parameter for use of imports of good i in produciton Armington good i
        deltad (1D numpy array): Share parameter for use of domestically produced good i in produciton Armington good i
        eta (1D numpy array): The elasticity of substitution between imports and domestically supplied good i
        Q (1D numpy array): The domestic supply of good i, the Armington good
        pq (1D numpy array): price of XXXX for each good i
        D (1D numpy array): Demand for domestically produced good i from importers

    Returns:
        pd (1D numpy array): price of domesically produced good i
    '''
    pd = (gamma ** eta * deltad * pq) * (D / Q) ** (eta - 1)
    return pd


def eqZ(theta, xie, xid, phi, E, D):
    '''
    Exporting firm production function.

    .. math::
        Z_{i} = \theta_{i}\left[\xi_{i}^{E}E_{i}^{\phi_{i}} + \xi_{i}^{D}D_{i}^{\phi_{i}}\right]^{\frac{1}{\phi_{i}}}

    Args:
        theta (1D numpy array):
        xie (1D numpy array): Share parameter for the share of exports of good i used by firms exporting good i
        xie (1D numpy array): Share parameter for the share of domestically produced good i used by firms exporting good i
        phi (1D numpy array): Elasticity of substitution between exports (??) and domestically produced goods by firms exporting good i
        E (1D numpy array): Exports of good i
        D (1D numpy array): Domestically produced good i

    Returns:
        Z (1D numpy array): Output from exporters CET production function
    '''
    Z = theta * (xie * E ** phi + xid * D ** phi) ** (1 / phi)
    return Z


def eqE(theta, xie, tauz, phi, pz, pe, Z):
    '''
    Supply of exports.

    .. math::
        E_{i} = \left(\theta_{i}^{\phi_{i}}\xi^{E}_{i}(1+\tau^{z}_{i}\frac{pz_{i}}{pe_{i}})\right)^{\frac{1}{1-\phi_{i}}}Z_{i}

    Args:
        theta (1D numpy array):
        xie (1D numpy array): Share parameter for the share of exports of good i used by firms exporting good i
        tauz (1D numpy array): Ad valorem tax rate on commodity i
        phi (1D numpy array): Elasticity of substitution between exports (??) and domestically produced goods by firms exporting good i
        pz (1D numpy array): price of output good i
        pe (1D numpy array): Price of commodity i exports in domestic currency
        Z (1D numpy array): Output from exporters CET production function

    Returns:
        E (1D numpy array): Exports of good i
    '''
    E = (theta ** phi * xie * (1 + tauz) * pz / pe) ** (1 / (1 - phi)) * Z
    return E


def eqD(theta, xid, tauz, phi, pz, pd, Z):
    '''
    Demand for domestic goods by exporters.

    .. math::
        D_{i} = \left(\theta_{i}^{\phi_{i}}\xi^{D}_{i}(1+\tau^{z}_{i}\frac{pz_{i}}{pd_{i}})\right)^{\frac{1}{1-\phi_{i}}}Z_{i}

    Args:
        theta (1D numpy array):
        xid (1D numpy array): Share parameter for the share of domestically produced good i used by firms exporting good i
        tauz (1D numpy array): Ad valorem tax rate on commodity i
        phi (1D numpy array): Elasticity of substitution between exports (??) and domestically produced goods by firms exporting good i
        pz (1D numpy array): price of output good i
        pd (1D numpy array): price of domesically produced good i
        Z (1D numpy array): Output from exporters CET production function

    Returns:
        D (1D numpy array): Demand for domestic good i by exporters.
    '''
    D = (theta ** phi * xid * (1 + tauz) * pz / pd) ** (1 / (1 - phi)) * Z
    return D


def eqpq(Q, Xp, Xg, Xv, X):
    '''
    Resource constraint.

    .. math::
        Q_{i} = X^{p}_{i} + X^{g}_{i} + X^{v}_{i} + \sum_{j}X_{i,j}
    Args:
        Q (1D numpy array): The domestic supply of good i, the Armington good
        Xp (1D numpy array): Demand for production good i
        Xg (1D numpy array): Government expenditures on commodity i
        Xv (1D numpy array): Investment demand for each good i
        X (2D numpy array): Demand for intermediate input i used in the
            production of good j

    Returns:
        pq_error (1D numpy array): Error in resource constraint for each good i
    '''
    pq_error = Q - (Xp + Xg + Xv + X.sum(axis=1))
    return pq_error


def eqpf(F, Ff0):
    '''
    ??

    .. math::

    Args:
        F (2D numpy array): The use of factor h in the production of
            good j
        Ff0 (??))

    Returns:
        pf_error ():
    '''
    F1 = F.drop(['CAP'])
    Ff1 = Ff0.drop(['CAP'])
    pf_error = Ff1 - F1.sum(axis=1)
    return pf_error


def eqpk(F, Kk, Kk0, Ff0):
    '''
    ??

    .. math::

    Args:
        F (2D numpy array): The use of factor h in the production of
            good j
        Kk (float): Total capital stock

    Returns:
        pk_error ():
    '''
    Fcap = F.loc[['CAP']]
    pk_error = Fcap.sum(axis=1) - Kk / Kk0 * Ff0['CAP']
    return pk_error
