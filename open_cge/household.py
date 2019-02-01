def eqF(beta, py, Y, pf):
    '''
    Factor demand.

    .. math::
        F_{h,j} = \\beta_{h,j}\\frac{py_{j}}{pf_{h}}Y_{j}

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


def eqXp(alpha, pf, Ff, Sp, Td, Fsh, Trf, pq):
    '''
    Demand for production good i by consumers.

    .. math::
        X^{p}_{i}= \\frac{}\\alpha_{i}}{pq_{i}}\left(\sum_{h}pf_{h}Ff_{h} - S^{p} - T^{d}- FSH - TRF\\right)

    Args:
        alpha (1D numpy array): Budget share of good i
        pf (1D numpy array): The price of factor h
        Ff (1D numpy array): Endowment of factor h
        Sp (float): Total household savings
        Td (float): Total direct tax revenue
        Fsh = Repatriated profits
        Trf (float): Total transfers to households
        pq (1D numpy array): price of the Armington good (domestic + imports) for each good i

    Returns:
        Xp (1D numpy array): Demand for production good i by consumers
    '''
    Xp = alpha * ((pf * Ff).sum() - Sp - Td - Fsh + Trf) / pq
    return Xp
