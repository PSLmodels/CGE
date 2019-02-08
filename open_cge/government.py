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
        XXg (float): Total government spending on goods/services

    Returns:
        Xg (1D numpy array): Government expenditures on commodity j
    '''
    Xg = mu * XXg.values
    return Xg


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
        XXg (float): Total government spending on goods/services
        Trf (float): Total transfers to households
        pq (1D numpy array): price of the Armington good (domestic + imports) for each good i

    Returns:
        Sg (float): Total government savings
    '''
    Sg = Td + Tz.sum() + Tm.sum() - (Trf + XXg * (mu * pq).sum())
    return Sg
