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
    Production tax revenue from each good.

    .. math::
        Tz_{i} = \tau^{z}_{i} pz_{i}Z_{i}

    Args:
        tauz (1D numpy array): Ad valorem tax rate on good i
        pz (1D numpy array): Price of output good i
        Z (1D numpy array): Total output of good i

    Returns:
        Tz (1D numpy array): Production tax revenue for each good i
    '''
    Tz = tauz * pz * Z
    return Tz


def eqTm(taum, pm, M):
    '''
    Tariff revenue from each good i.

    .. math::
        Tm_{i} = \tau^{m}_{i} pm_{i}M_{i}

    Args:
        taum (1D numpy array): Tariff rate on good i
        pm (1D numpy array): price of import good i
        M (1D numpy array): Imports of good i

    Returns:
        Tm (1D numpy array): Tariff revenue for each good i
    '''
    Tm = taum * pm * M
    return Tm


def eqXg(mu, XXg):
    '''
    Government expenditures on good i

    .. math::
        X^{g}_{i} = \mu_{i}XX_{g}

    Args:
        mu (1D numpy array): Government expenditure share parameters for
            each good i
        XXg (float): Total government spending on goods/services

    Returns:
        Xg (1D numpy array): Government expenditures on good i
    '''
    Xg = mu * XXg
    return Xg


def eqSg(mu, Td, Tz, Tm, XXg, Trf, pq):
    '''
    Total government savings.

    .. math::
        Sg = Td + \sum_{i}Tz_{i} + \sum_{i}Tm_{i} - (Trf + \sum_{i}Xg_{i})

    Args:
        mu (1D numpy array): Government expenditure share parameters for
            each good i
        Td (float): Total direct tax revenue
        Tz (1D numpy array): Production tax revenue for each good i
        Tm (1D numpy array): Tariff revenue for each good i
        XXg (float): Total government spending on goods/services
        Trf (float): Total transfers to households
        pq (1D numpy array): price of the Armington good (domestic +
            imports) for each good i

    Returns:
        Sg (float): Total government savings
    '''
    Sg = Td + Tz.sum() + Tm.sum() - (Trf + XXg * (mu * pq).sum())
    return Sg
