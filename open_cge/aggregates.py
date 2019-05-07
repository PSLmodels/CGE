def eqSp(ssp, pf, Ff, Fsh, Trf):
    '''
    Total household savings.

    .. math::
        Sp = ssp \cdot \left(\sum_{h}pf_{h}Ff_{h} \\right)

    Args:
        ssp (float): Fixed household savings rate
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h
        Fsh (float): Repatriated profits
        Trf (float): Total transfers to households

    Returns:
        Sp (float): Total household savings
    '''
    Sp = ssp * ((pf * Ff).sum() - Fsh + Trf)
    return Sp


def eqKd(g, Sp, lam, pq):
    '''
    Domestic capital holdings.

    .. math::
        K^{d} = \\frac{S^{p}}{g\sum_{j}\lambda_{j}pq_{j}}

    Args:
        g (float): Exogenous long run growth rate of the economy
        Sp (float): Total household savings
        lam (1D numpy array): Fixed shares of investment for each good j
        pq (1D numpy array): price of the Armington good (domestic +
            imports) for each good j

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
        Kd (float): Domestically owned capital

    Returns:
        Kf (float): Foreign owned domestic capital
    '''
    Kf = Kk - Kd
    return Kf


def eqKk(pf, Ff, R, lam, pq):
    '''
    Capital market clearing equation.

    .. math::
        KK = \\frac{pf * FF}{R \sum_{j}\lambda_{j}pq_{j}}

    Args:
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h
        R (float): Real return on capital
        lam (1D numpy array): Fixed shares of investment for each good j
        pq (1D numpy array): price of the Armington good (domestic +
            imports) for each good j

    Returns:
        Kk (float): Total capital stock
    '''
#    R = ( (pf['CAP'] * Ff['CAP']) / Kk) / ((lam * pq).sum())
    print('kk inputs= ', pf['CAP'], Ff['CAP'], R, lam, pq)
    Kk = (pf['CAP'] * Ff['CAP']) / (R * ((lam * pq).sum()))
    return Kk


def eqbop(pWe, pWm, E, M, Sf, Fsh, er):
    '''
    Balance of payments.

    .. math::
        \sum_{i}pWe_{i}E_{i} + \\frac{Sf}{\\varepsilon} = \sum_{i}pWm_{i}M_{i} + \\frac{Fsh}{\\varepsilon}

    Args:
        pWe (1D numpy array): The world price of commodity i in foreign
            currency
        pWm (1D numpy array): The world price of commodity i in foreign
            currency.
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
        pq (1D numpy array): price of the Armington good (domestic +
            imports) for each good j
        Kf (float): Foreign owned domestic capital

    Returns:
        Sf (float): Total foreign savings (??)
    '''
    Sf = g * Kf * (lam * pq).sum()
    return Sf


def eqpqerror(Q, Xp, Xg, Xv, X):
    '''
    Resource constraint.

    .. math::
        Q_{i} = X^{p}_{j} + X^{g}_{j} + X^{v}_{j} + \sum_{j}X_{i,j}

    Args:
        Q (1D numpy array): The domestic supply of good j, the Armington good
        Xp (1D numpy array): Demand for production good j by consumers
        Xg (1D numpy array): Government expenditures on commodity i
        Xv (1D numpy array): Investment demand for each good j
        X (2D numpy array): Demand for intermediate input i used in the
            production of good j

    Returns:
        pq_error (1D numpy array): Error in resource constraint for each good j
    '''
    pq_error = Q - (Xp + Xg + Xv + X.sum(axis=1))
    return pq_error


def eqpf(F, Ff0):
    '''
    Comparing labor supply from the model to that in the data.

    Args:
        F (2D numpy array): The use of factor h in the production of
            good j
        Ff0 (float): Total labor demand from SAM

    Returns:
        pf_error ():
    '''
    F1 = F.drop(['CAP'])
    Ff1 = Ff0.drop(['CAP'])
    pf_error = Ff1 - F1.sum(axis=1)
    return pf_error


def eqpk(F, Kk, Kk0, Ff0):
    '''
    Comparing capital demand in the model and data.

    Args:
        F (2D numpy array): The use of factor h in the production of
            good j
        Kk (float): Total capital stock
        Kk0 (float): Total capital stock from SAM??
        Ff0 (float): Total labor demand from SAM??

    Returns:
        pk_error ():
    '''
    Fcap = F.loc[['CAP']]
    pk_error = Fcap.sum(axis=1) - Kk / Kk0 * Ff0['CAP']
    return pk_error


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
