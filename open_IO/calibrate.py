# import packages
import numpy as np
from pandas import Series, DataFrame


class model_data(object):
    """
    This function reads the SAM file and initializes variables using
    these data.

    Args:
        sam (DataFrame): DataFrame containing social and economic data

    Returns:
        model_data (data class): Data used in the CGE model
    """

    def __init__(self, sam, h, u, ind):
        self.Sf0 = DataFrame(
            sam, index=["INV"], columns=["EXT"]
        )  # foreign saving
        self.Sp0 = DataFrame(
            sam, index=["INV"], columns=["HOH"]
        )  # private saving'
        self.Sg0 = DataFrame(
            sam, index=["INV"], columns=["GOV"]
        )  # government saving/budget balance

        self.Fsh0 = DataFrame(
            sam, index=["EXT"], columns=["HOH"]
        )  # repatriation of profits
        self.Kk0 = 10510  # capital stock
        self.Kf0 = 6414.35  # foreign-owned capital stock
        self.Kd0 = self.Kk0 - self.Kf0  # domestically-owned capital stock

        self.Td0 = DataFrame(sam, index=["DTX"], columns=["HOH"])  # direct tax
        self.Trf0 = DataFrame(sam, index=["HOH"], columns=["GOV"])  # transfers
        self.Tz0 = DataFrame(
            sam, index=["ACT"], columns=list(ind)
        )  # production tax
        self.Tm0 = DataFrame(
            sam, index=["IDT"], columns=list(ind)
        )  # import tariff

        self.F0 = DataFrame(
            sam, index=list(h), columns=list(ind)
        )  # the h-th factor input by the i-th firm
        self.Ff0 = self.F0.sum(axis=1)  # factor endowment of the h-th factor
        self.Y0 = self.F0.sum(axis=0)  # composite factor (value added)
        self.X0 = DataFrame(
            sam, index=list(ind), columns=list(ind)
        )  # intermediate input
        self.Xx0 = self.X0.sum(
            axis=0
        )  # total intermediate input by the j-th sector
        self.Z0 = self.Y0 + self.Xx0  # output of the j-th good

        self.Xp0 = DataFrame(
            sam, index=list(ind), columns=["HOH"]
        )  # household consumption of the i-th good
        self.Xg0 = DataFrame(
            sam, index=list(ind), columns=["GOV"]
        )  # government consumption
        self.Xv0 = DataFrame(
            sam, index=list(ind), columns=["INV"]
        )  # investment demand
        self.E0 = DataFrame(sam, index=list(ind), columns=["EXT"])  # exports
        self.E0 = self.E0["EXT"]
        self.M0 = DataFrame(sam, index=["EXT"], columns=list(ind))  # imports
        self.M0 = self.M0.loc["EXT"]

        self.Q0 = (
            self.Xp0["HOH"]
            + self.Xg0["GOV"]
            + self.Xv0["INV"]
            + self.X0.sum(axis=1)
        )  # domestic supply/Armington composite good
        tauz = self.Tz0 / self.Z0  # production tax rate
        self.D0 = (1 + tauz.loc["ACT"]) * self.Z0 - self.E0  # domestic
        # D0 = D0.loc['ACT']

        # Compute aggregates
        self.Yy0 = self.Y0.sum()
        self.XXp0 = self.Xp0.sum()
        self.XXv0 = self.Xv0.sum()
        self.XXg0 = self.Xg0.sum()
        self.Mm0 = self.M0.sum()
        self.Ee0 = self.E0.sum()
        self.Gdp0 = self.XXp0 + self.XXv0 + self.XXg0 + self.Ee0 - self.Mm0

        self.g = self.XXv0 / self.Kk0
        self.R0 = self.Ff0["CAP"] / self.Kk0

        self.pWe = np.ones(len(ind))  # export price index
        self.pWe = Series(self.pWe, index=list(ind))
        self.pWm = np.ones(len(ind))  # import price index
        self.pWm = Series(self.pWm, index=list(ind))


class parameters(object):
    """
    This function sets the values of parameters used in the model.

    Args:
        d (data class): Class of data for use in CGE model
        ind (list): List of industry names

    Returns:
        parameters (parameters class): Class of parameters for use in CGE model.
    """

    def __init__(self, d, ind):

        self.sigma = [3, 1.2, 3, 3]  # elasticity of substitution
        self.sigma = Series(self.sigma, index=list(ind))
        self.eta = (
            self.sigma - 1
        ) / self.sigma  # substitution elasticity parameter

        self.psi = [3, 1.2, 3, 3]  # elasticity of transformation
        self.psi = Series(self.psi, index=list(ind))
        self.phi = (
            self.psi + 1
        ) / self.psi  # transformation elasticity parameter

        self.alpha = d.Xp0 / d.XXp0  # share parameter in utility function
        self.alpha = self.alpha["HOH"]
        self.beta = d.F0 / d.Y0  # share parameter in production function
        temp = d.F0**self.beta
        self.b = d.Y0 / temp.prod(
            axis=0
        )  # scale parameter in production function

        self.ax = d.X0 / d.Z0  # intermediate input requirement coefficient
        self.ay = d.Y0 / d.Z0  # composite factor input requirement coefficient
        self.mu = d.Xg0 / d.XXg0  # government consumption share
        self.mu = self.mu["GOV"]
        self.lam = d.Xv0 / d.XXv0  # investment demand share
        self.lam = self.lam["INV"]

        self.tauz = d.Tz0 / d.Z0  # production tax rate
        self.tauz = self.tauz.loc["ACT"]
        self.taum = d.Tm0 / d.M0  # import tariff rate
        self.taum = self.taum.loc["IDT"]

        # import propensity
        self.deltam = (1 + self.taum) * d.M0 / ((1 + self.taum) * d.M0 + d.D0)
        self.deltad = d.D0 / ((1 + self.taum) * d.M0 + d.D0)

        # scale parameter in Armington function
        self.gamma = d.Q0 / (
            self.deltam * d.M0**self.eta + self.deltad * d.D0**self.eta
        ) ** (1 / self.eta)

        # share parameter in transformation function
        self.xie = d.E0 ** (1 - self.phi) / (
            d.E0 ** (1 - self.phi) + d.D0 ** (1 - self.phi)
        )
        #        self.xie = self.xie.iloc[0]
        self.xid = d.D0 ** (1 - self.phi) / (
            d.E0 ** (1 - self.phi) + d.D0 ** (1 - self.phi)
        )
        #        self.xid = self.xid.iloc[0]

        # scale parameter in transformation function
        self.theta = d.Z0 / (
            self.xie * d.E0**self.phi + self.xid * d.D0**self.phi
        ) ** (1 / self.phi)
        #        self.theta = self.theta.iloc[0]

        self.ssp = d.Sp0.values / (
            d.Ff0.sum() - d.Fsh0.values + d.Trf0.values
        )  # average propensity to save
        self.ssp = np.asscalar(self.ssp)
        self.taud = d.Td0.values / d.Ff0.sum()  # direct tax rate
        self.taud = np.asscalar(self.taud)
        self.tautr = d.Trf0.values / d.Ff0["LAB"]  # transfer rate
        self.tautr = np.asscalar(self.tautr)
        self.ginc = d.Td0 + d.Tz0.sum() + d.Tm0.sum()  # government revenue
        self.hinc = d.Ff0.sum()  # household income
