def eqpy(b, F, beta, Y):
    '''
    Production function.

    .. math::
        Y_{j} = b_{j}\prod_{h}F_{h,j}^{\\beta_{h,j}}

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
        pq (1D numpy array): price of the Armington good (domestic + imports) for each good i

    Returns:
        pz (1D numpy array): price of output good j
    '''
    pz = ay * py + (ax * pq).sum(axis=0)
    return pz


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
    Xv = lam * XXv
    return Xv


def eqFsh(R, Kf, er):
    '''
    Domestic profits that are repatriated to foreign owners of capital.

    .. math::
        FSH = R \cdot KF \cdot \\varepsilon

    Args:
        R (float): Real return on capital
        Kf (float): Foreign holdings of domestic capital
        er (float): The real exchange rate

    Returns:
        Fsh = Repatriated profits
    '''
    Fsh = R * Kf * er
    return Fsh


def eqpe(er, pWe):
    '''
    Export prices.

    .. math::
        pe_{i} = \\varepsilon \cdot pWe_{i}

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
        pm_{i} = \\varepsilon \cdot pWm_{i}

    Args:
        er (float): The real exchange rate
        pWm (1D numpy array): The world price of commodity i in foreign currency.

    Returns:
        pm (1D numpy array): The price of commodity i imports in domestic currency.
    '''
    pm = er * pWm
    return pm


def eqQ(gamma, deltam, deltad, eta, M, D):
    '''
    CES production function for the importing firm.

    .. math::
        Q_{i} = \gamma_{i}\left[\delta^{m}_{i}M^{\eta_{i}}_{i} + \delta^{d}_{i}D^{\eta_{i}}_{i}\\right]^{\\frac{1}{\eta_{i}}}

    Args:
        gamma (1D numpy array): Scale parameter for CES production function
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


def eqM(gamma, deltam, eta, Q, pq, pm, taum):
    '''
    Demand for imports.

    .. math::
        M_{i} = \left(\gamma^{\eta_{i}}_{i}\delta^{m}_{i}\\frac{pq_{i}}{(1+\\tau^{m}_{i})pm_{i}}\\right)^{\\frac{1}{1-\eta_{i}}}Q_{i}

    Args:
        gamma (1D numpy array): Scale parameter for CES production function
        deltam (1D numpy array): Share parameter for use of imports of good i in produciton Armington good i
        eta (1D numpy array): The elasticity of substitution between imports and domestically supplied good i
        Q (1D numpy array): The domestic supply of good i, the Armington good
        pq (1D numpy array): price of the Armington good (domestic + imports) for each good i
        pm (1D numpy array): The price of commodity i imports in domestic currency.
        taum (1D numpy array): Tariff rate on commodity i

    Returns:
        M (1D numpy array): Demand for imports of good i
    '''
    M = (gamma ** eta * deltam * pq / ((1 + taum) * pm)) ** (1 / (1 - eta)) * Q
    return M


def eqD(gamma, deltad, eta, Q, pq, pd):
    '''
    Demand for domestically produced goods from importers.

    .. math::
        D_{i} = \left(\gamma_{i}^{\eta_{i}}\delta^{d}_{i}\\frac{pq_{i}}{pd_{i}}\\right)^{\\frac{1}{1-\eta_{i}}}Q_{i}

    Args:
        gamma (1D numpy array): Scale parameter for CES production function
        deltad (1D numpy array): Share parameter for use of domestically produced good i in produciton Armington good i
        eta (1D numpy array): The elasticity of substitution between imports and domestically supplied good i
        Q (1D numpy array): The domestic supply of good i, the Armington good
        pq (1D numpy array): price of the Armington good (domestic + imports) for each good i
        pd (1D numpy array): price of domesically produced good i

    Returns:
        D (1D numpy array): Demand for domestically produced good i from importers
    '''
    D = (gamma ** eta * deltad * pq / pd) ** (1 / (1 - eta)) * Q
    return pd


def eqpd(gamma, deltad, eta, Q, pq, D):
    '''
    Price of domestically produced goods from importers.

    .. math::
        pd_{i} = \left(\gamma_{i}^{\eta_{i}}\delta^{d}_{i}pq_{i}\\right)\left(\\frac{D_{i}}{Q_{i}}\\right)^{\eta_{i}-1}

    Args:
        gamma (1D numpy array): Scale parameter for CES production function
        deltad (1D numpy array): Share parameter for use of domestically produced good i in produciton Armington good i
        eta (1D numpy array): The elasticity of substitution between imports and domestically supplied good i
        Q (1D numpy array): The domestic supply of good i, the Armington good
        pq (1D numpy array): price of the Armington good (domestic + imports) for each good i
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
        Z_{i} = \\theta_{i}\left[\\xi_{i}^{E}E_{i}^{\phi_{i}} + \\xi_{i}^{D}D_{i}^{\phi_{i}}\\right]^{\\frac{1}{\phi_{i}}}

    Args:
        theta (1D numpy array): Scaling coefficient of the ith good transformation from domestic output to exports
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
        E_{i} = \left(\\theta_{i}^{\phi_{i}}\\xi^{E}_{i}(1+\\tau^{z}_{i}\\frac{pz_{i}}{pe_{i}})\\right)^{\\frac{1}{1-\phi_{i}}}Z_{i}

    Args:
        theta (1D numpy array): Scaling coefficient of the ith good transformation from domestic output to exports
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


def eqDex(theta, xid, tauz, phi, pz, pd, Z):
    '''
    Demand for domestic goods by exporters.

    .. math::
        D_{i} = \left(\\theta_{i}^{\phi_{i}}\\xi^{D}_{i}(1+\\tau^{z}_{i}\\frac{pz_{i}}{pd_{i}})\\right)^{\\frac{1}{1-\phi_{i}}}Z_{i}

    Args:
        theta (1D numpy array): Scaling coefficient of the ith good transformation from domestic output to exports
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


def eqpq(pm, pd, taum, eta, deltam, deltad, gamma):

    pq = (((pm * (1 + taum)) ** eta / (deltam * gamma ** eta)) ** (1 / (eta - 1))
			+ (pd ** eta / (deltad * gamma ** eta)) ** (1 / (eta - 1))) ** ((eta - 1) / eta)
    return pq
