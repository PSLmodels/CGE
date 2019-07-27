# import packages
import numpy as np
from pandas import Series, DataFrame


class model_data(object):
    '''
    This function reads the SAM file and initializes variables using
    these data.

    Args:
        sam (DataFrame): DataFrame containing social and economic data

    Returns:
        model_data (data class): Data used in the CGE model
    '''

    def __init__(self, sam, h, u, ind):
        # foreign saving
        self.Sf0 = DataFrame(sam, index=['INV'], columns=['EXT'])
        # private saving
        self.Sp0 = DataFrame(sam, index=['INV'], columns=['HOH'])
        # government saving/budget balance
        self.Sg0 = DataFrame(sam, index=['INV'], columns=['GOV'])
        # repatriation of profits
        self.Fsh0 = DataFrame(sam, index=['EXT'], columns=['HOH'])
        # capital stock
        self.Kk0 = 10510
        # foreign-owned capital stock
        self.Kf0 = 6414.35
        # domestically-owned capital stock
        self.Kd0 = self.Kk0 - self.Kf0

        # direct tax
        self.Td0 = DataFrame(sam, index=['DTX'], columns=['HOH'])
        # transfers
        self.Trf0 = DataFrame(sam, index=['HOH'], columns=['GOV'])
        # production tax
        self.Tz0 = DataFrame(sam, index=['ACT'], columns=list(ind))
         # import tariff
        self.Tm0 = DataFrame(sam, index=['IDT'], columns=list(ind))

        # the h-th factor input by the j-th firm
        self.F0 = DataFrame(sam, index=list(h), columns=list(ind))
        # factor endowment of the h-th factor
        self.Ff0 = self.F0.sum(axis=1)
        # composite factor (value added)
        self.Y0 = self.F0.sum(axis=0)
        # intermediate input
        self.X0 = DataFrame(sam, index=list(ind), columns=list(ind))
        # total intermediate input by the j-th sector
        self.Xx0 = self.X0.sum(axis=0)
        # output of the i-th good
        self.Z0 = self.Y0 + self.Xx0

        # household consumption of the i-th good
        self.Xp0 = DataFrame(sam, index=list(ind), columns=['HOH'])
        # government consumption
        self.Xg0 = DataFrame(sam, index=list(ind), columns=['GOV'])
        # investment demand
        self.Xv0 = DataFrame(sam, index=list(ind), columns=['INV'])
        # exports
        self.E0 = DataFrame(sam, index=list(ind), columns=['EXT'])
        self.E0 = self.E0['EXT']
        # imports
        self.M0 = DataFrame(sam, index=['EXT'], columns=list(ind))
        self.M0 = self.M0.loc['EXT']

        # domestic supply/Armington composite good
        self.Q0 = (self.Xp0['HOH'] + self.Xg0['GOV'] + self.Xv0['INV']
                   + self.X0.sum(axis=1))
        # production tax rate
        tauz = self.Tz0 / self.Z0
        # domestic tax rate
        self.D0 = (1 + tauz.loc['ACT']) * self.Z0 - self.E0

        # Compute aggregates

        # aggregate output
        self.Yy0 = self.Y0.sum()
        # aggregate demand
        self.XXp0 = self.Xp0.sum()
        # aggregate investment
        self.XXv0 = self.Xv0.sum()
        # aggregate government spending
        self.XXg0 = self.Xg0.sum()
        # aggregate imports
        self.Mm0 = self.M0.sum()
        # aggregate exports
        self.Ee0 = self.E0.sum
        # aggregate gross domestic product
        self.Gdp0 = (self.XXp0 + self.XXv0 + self.XXg0 + self.Ee0 -
                     self.Mm0)
        # growth rate of capital stock
        self.g = self.XXv0 / self.Kk0
        # interest rate
        self.R0 = self.Ff0['CAP'] / self.Kk0

        # export price index
        self.pWe = np.ones(len(ind))
        self.pWe = Series(self.pWe, index=list(ind))
        # import price index
        self.pWm = np.ones(len(ind))
        self.pWm = Series(self.pWm, index=list(ind))


class parameters(object):
    '''
    This function sets the values of parameters used in the model.

    Args:

    Returns:
        parameters (parameters class): Class of parameters for use in
            CGE model.
    '''

    def __init__(self, d, ind):

        # elasticity of substitution
        self.sigma = ([3, 1.2, 3, 3])
        self.sigma = Series(self.sigma, index=list(ind))
        # substitution elasticity parameter
        self.eta = (self.sigma - 1) / self.sigma

        # elasticity of transformation
        self.psi = ([3, 1.2, 3, 3])
        self.psi = Series(self.psi, index=list(ind))
        # transformation elasticity parameter
        self.phi = (self.psi + 1) / self.psi

        # share parameter in utility function
        self.alpha = d.Xp0 / d.XXp0
        self.alpha = self.alpha['HOH']
        # share parameter in production function
        self.beta = d.F0 / d.Y0
        temp = d.F0 ** self.beta
        # scale parameter in production function
        self.b = d.Y0 / temp.prod(axis=0)

        # intermediate input requirement coefficient
        self.ax = d.X0 / d.Z0
        # composite factor input requirement coefficient
        self.ay = d.Y0 / d.Z0
        self.mu = d.Xg0 / d.XXg0
        # government consumption share
        self.mu = self.mu['GOV']
        self.lam = d.Xv0 / d.XXv0
        # investment demand share
        self.lam = self.lam['INV']

        # production tax rate
        self.tauz = d.Tz0 / d.Z0
        self.tauz = self.tauz.loc['ACT']
        # import tariff rate
        self.taum = d.Tm0 / d.M0
        self.taum = self.taum.loc['IDT']

        # share parameter in Armington function
        self.deltam = ((1 + self.taum) * d.M0 ** (1 - self.eta) /
                       ((1 + self.taum) * d.M0 ** (1 - self.eta) + d.D0
                        ** (1 - self.eta)))
        self.deltad = (d.D0 ** (1 - self.eta) /
                       ((1 + self.taum) * d.M0 ** (1 - self.eta) + d.D0
                        ** (1 - self.eta)))

        # scale parameter in Armington function
        self.gamma = (d.Q0 / (self.deltam * d.M0 ** self.eta +
                              self.deltad * d.D0 ** self.eta) **
                      (1 / self.eta))

        # share parameter in transformation function
        self.xie = (d.E0 ** (1 - self.phi) / (d.E0 ** (1 - self.phi) +
                                              d.D0 ** (1 - self.phi)))
        self.xid = (d.D0 ** (1 - self.phi) / (d.E0 ** (1 - self.phi) +
                                              d.D0 ** (1 - self.phi)))

        # scale parameter in transformation function
        self.theta = (d.Z0 / (self.xie * d.E0 ** self.phi + self.xid *
                              d.D0 ** self.phi) ** (1 / self.phi))

        # average propensity to save
        self.ssp = (d.Sp0.values / (d.Ff0.sum() - d.Fsh0.values +
                                    d.Trf0.values))
        self.ssp = np.asscalar(self.ssp)
        # direct tax rate
        self.taud = d.Td0.values / d.Ff0.sum()
        self.taud = np.asscalar(self.taud)
        # transfer rate
        self.tautr = d.Trf0.values / d.Ff0['LAB']
        self.tautr = np.asscalar(self.tautr)
        # government revenue
        self.ginc = d.Td0 + d.Tz0.sum() + d.Tm0.sum()
        # household income
        self.hinc = d.Ff0.sum()
