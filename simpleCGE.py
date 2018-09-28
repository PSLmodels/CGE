#import packages
import scipy.optimize as opt
import numpy as np
import pandas as pd
from pandas import Series, DataFrame

#load social accounting matrix
sam = pd.read_excel('SAM.xlsx')

#declare sets
u = ('AGR', 'OIL', 'IND', 'SER', 'LAB', 'CAP', 'LAND', 'NTR',
     'DTX', 'IDT', 'ACT', 'HOH', 'GOV', 'INV', 'EXT' )
ind = ('AGR', 'OIL', 'IND', 'SER')
h = ('LAB', 'CAP', 'LAND', 'NTR')
w = ('LAB', 'LAND', 'NTR')

#initialise variables

Sf0 = DataFrame(sam, index=['INV'], columns=['EXT']) #foreign saving
Sp0 = DataFrame(sam, index=['INV'], columns=['HOH']) #private saving'
Sg0 = DataFrame(sam, index=['INV'], columns=['GOV']) #government saving/budget balance

Fsh0 = DataFrame(sam, index=['EXT'], columns=['HOH']) #repatriation of profits
Kk0 = 10510 #capital stock
Kf0 = 6414.35 #foreign-owned capital stock
Kd0 = Kk0 - Kf0 #domestically-owned capital stock

Td0 = DataFrame(sam, index=['DTX'], columns=['HOH']) #direct tax
Trf0 = DataFrame(sam, index=['HOH'], columns=['GOV']) #transfers
Tz0 = DataFrame(sam, index=['ACT'], columns=list(ind)) #production tax
Tm0 = DataFrame(sam, index=['IDT'], columns=list(ind)) #import tariff

F0 = DataFrame(sam, index=list(h), columns=list(ind)) #the h-th factor input by the i-th firm
Ff0 = F0.sum(axis=1) #factor endowment of the h-th factor
Y0 = F0.sum(axis=0) #composite factor (value added)
X0 = DataFrame(sam, index=list(ind), columns=list(ind)) #intermediate input
Xx0 = X0.sum(axis=0)#total intermediate input by the j-th sector
Z0 = Y0 + Xx0 #output of the j-th good

Xp0 = DataFrame(sam, index=list(ind), columns=['HOH']) #household consumption of the i-th good
Xg0 = DataFrame(sam, index=list(ind), columns=['GOV']) #government consumption
Xv0 = DataFrame(sam, index=list(ind), columns=['INV']) #investment demand
E0 = DataFrame(sam, index=list(ind), columns=['EXT']) #exports
E0 = E0['EXT']
M0 = DataFrame(sam, index=['EXT'], columns=list(ind)) #imports
M0 = M0.loc['EXT']

tauz = Tz0/Z0 #production tax rate
tauz = tauz.loc['ACT']
taum = Tm0/M0 #import tariff rate
taum = taum.loc['IDT']

Q0 = Xp0['HOH'] + Xg0['GOV'] + Xv0['INV'] + X0.sum(axis=1) #domestic supply/Armington composite good
D0 = (1 + tauz) * Z0 - E0 #domestic
#D0 = D0.loc['ACT']

Yy0 = Y0.sum()
XXp0 = Xp0.sum()
XXv0 = Xv0.sum()
XXg0 = Xg0.sum()
Mm0 = M0.sum()
Ee0 = E0.sum()
Gdp0 = XXp0 + XXv0 + XXg0 + Ee0 - Mm0

g = XXv0/Kk0
R0 = Ff0['CAP']/Kk0

pWe = np.ones(len(ind)) #export price index
pWe = Series(pWe, index=list(ind))
pWm = np.ones(len(ind)) #import price index
pWm = Series(pWm, index=list(ind))


# Calibration

sigma = ([3, 1.2, 3, 3]) #elasticity of substitution
sigma = Series(sigma, index=list(ind))
eta = (sigma - 1) / sigma #substitution elasticity parameter

psi = ([3, 1.2, 3, 3]) #elasticity of transformation
psi = Series(psi, index=list(ind))
phi = (psi + 1) / psi #transformation elasticity parameter


alpha = Xp0 / XXp0 #share parameter in utility function
alpha = alpha ['HOH']
beta = F0 / Y0 #share parameter in production function
temp = F0 ** beta
b = Y0 / temp.prod(axis=0)#scale parameter in production function

ax = X0 / Z0 #intermediate input requirement coefficient
ay = Y0 / Z0 #composite factor input requirement coefficient
mu = Xg0 / XXg0 #government consumption share
mu = mu['GOV']
lam = Xv0 / XXv0 #investment demand share
lam = lam['INV']

#share parameter in Armington function
deltam = ( (1 + taum) * M0 ** (1 - eta) /
         ( (1 + taum) * M0 ** (1 - eta) + D0 ** (1 - eta) ) )

deltad = ( D0 ** (1 - eta) /
         ( (1 + taum) * M0 ** (1 - eta) + D0 ** (1 - eta) ) )

#scale parameter in Armington function
gamma = Q0 / ( deltam * M0 ** eta + deltad *D0 ** eta ) ** (1 / eta)

#share parameter in transformation function
xie = E0 ** (1 - phi) / (E0 ** (1 - phi) + D0 ** (1 - phi) )
xie = xie.iloc[0]
xid = D0 ** (1 - phi) / (E0 ** (1 - phi) + D0 ** (1 - phi) )
xid = xid.iloc[0]

#scale parameter in transformation function
theta = Z0 / (xie * E0 ** phi + xid * D0 ** phi) ** (1 / phi)
theta = theta.iloc[0]

ssp = Sp0.values / (Ff0.sum() - Fsh0.values + Trf0.values) #average propensity to save
ssp = np.asscalar(ssp)
taud = Td0.values / Ff0.sum() #direct tax rate
taud = np.asscalar(taud)
tautr = Trf0.values / Ff0['LAB'] #transfer rate
tautr = np.asscalar(tautr)
ginc = Td0 + Tz0.sum() + Tm0.sum()   #government revenue
hinc = Ff0.sum() #household income





def cge_system(pvec, *args):
    (eta, phi, alpha, beta, b, ax, ay, mu, lam, deltam, deltad, gamma, xid, xie,
     theta, ssp, taud, tautr, tauz, taum, g, pWe, pWm, Kk0, Ff0, XXg, R, ind, h, er, Z, Q, Kd, pd, Ff) = args

    py = pvec[0:len(ind)]
    pq = pvec[len(ind):len(ind)*2]
    pf = pvec[len(ind)*2:12]

    py = Series(py, index=list(ind))
    pq = Series(pq, index=list(ind))
    pf = Series(pf, index=list(h))

    pe = eqpe(er, pWe)
    pm = eqpm(er, pWm)

    pz = eqpz(ay, ax, py, pq)

    Kk = eqKk(pf, Ff, R, lam, pq)
    XXv = eqXXv(g, Kk)
    Xv = eqXv(lam, XXv)

    Xg = eqXg(mu, XXg)

    Kf = eqKf(Kk, Kd)
    Fsh = eqFsh(R, Kf, er)
    Sf = eqSf(g, lam, pq, Kf)

    Td = eqTd(taud, pf, Ff)
    Trf = eqTrf(tautr, pf, Ff)
    Tz = eqTz(tauz, pz, Z)

    X = eqX(ax, Z)
    Y = eqY(ay, Z)
    F = eqF(beta, py, Y, pf)

    Sp = eqSp(ssp, pf, Ff, Fsh, Trf)
    Xp = eqXp(alpha, pf, Ff, Sp, Td, Fsh, Trf, pq)

    E = eqE(theta, xie , tauz, phi, pz, pe, Z)
    D = eqD(theta, xid , tauz, phi, pz, pd, Z)

    M = eqM(gamma, deltam, deltad, eta, Q, pq, pm, taum)
    Tm = eqTm(taum, pm, M)
    Sg = eqSg(mu, Td, Tz, Tm, XXg, Trf, pq)

    pq_error = eqpq(Q, Xp, Xg, Xv, X)
    pf_error = eqpf(F, Ff0)
    pk_error = eqpk(F, Kk, Kk0, Ff0)
    py_error = eqpy(b, F, beta, Y)

    pf_error = pf_error.append(pk_error)
    pf_error = DataFrame(pf_error)
    pf_error = pf_error.T
    pf_error = DataFrame(pf_error, columns = list(h))
    pf_error = pf_error.iloc[0]

    py_error = py_error.values
    pf_error = pf_error.values
    pq_error = pq_error.values

    p_error = np.append(py_error, pq_error)
    p_error = np.append(p_error, pf_error)

    return p_error



#solve cge_system

dist = 10
tpi_iter = 0
tpi_max_iter = 30
tpi_tol = 1e-10
xi = 0.1

#pvec = pvec_init
pvec = np.ones(12)

XXg = XXg0
R = R0
er = 1

Zbar = Z0
Ffbar = Ff0
Kdbar = Kd0
Qbar = Q0
pdbar = pvec[0:len(ind)]

'''
#checking system
py = pvec[0:len(ind)]
pq = pvec[len(ind):len(ind)*2]
pf = pvec[len(ind)*2:12]

py = Series(py, index=list(ind))
pq = Series(pq, index=list(ind))
pf = Series(pf, index=list(h))

pe = eqpe(er, pWe)
pm = eqpm(er, pWm)

pz = eqpz(ay, ax, py, pq)

Kk = eqKk(pf, Ffbar, R, lam, pq)
XXv = eqXXv(g, Kk)
Xv = eqXv(lam, XXv)

Xg = eqXg(mu, XXg)

Kf = eqKf(Kk, Kdbar)
Fsh = eqFsh(R, Kf, er)
Sf = eqSf(g, lam, pq, Kf)

Td = eqTd(taud, pf, Ffbar)
Trf = eqTrf(tautr, pf, Ffbar)
Tz = eqTz(tauz, pz, Zbar)

X = eqX(ax, Zbar)
Y = eqY(ay, Zbar)
F = eqF(beta, py, Y, pf)

Sp = eqSp(ssp, pf, Ffbar, Fsh, Trf)
Xp = eqXp(alpha, pf, Ffbar, Sp, Td, Fsh, Trf, pq)

E = eqE(theta, xie , tauz, phi, pz, pe, Zbar)
D = eqD(theta, xid , tauz, phi, pz, pdbar, Zbar)

M = eqM(gamma, deltam, deltad, eta, Qbar, pq, pm, taum)
Tm = eqTm(taum, pm, M)
Sg = eqSg(mu, Td, Tz, Tm, XXg, Trf, pq)

pq_error = eqpq(Qbar, Xp, Xg, Xv, X)
pf_error = eqpf(F, Ff0)
pk_error = eqpk(F, Kk, Kk0, Ff0)
py_error = eqpy(b, F, beta, Y)


pf_error = pf_error.append(pk_error)
pf_error = DataFrame(pf_error)
pf_error = pf_error.T
pf_error = DataFrame(pf_error, columns = list(h))
pf_error = pf_error.iloc[0]
#pf_error.columns = list(ind)

py_error = py_error.values
pf_error = pf_error.values
pq_error = pq_error.values

p_error = np.append(py_error, pq_error)
p_error = np.append(p_error, pf_error)
#p_error = py_error.append([pq_error, pf_error])
#p_error = p_error.values
#---------------------------------------------
'''

while (dist > tpi_tol) & (tpi_iter < tpi_max_iter):
    tpi_iter += 1

    cge_args = (eta, phi, alpha, beta, b, ax, ay, mu, lam, deltam, deltad, gamma, xid, xie,
         theta, ssp, taud, tautr, tauz, taum, g, pWe, pWm, Kk0, Ff0, XXg, R, ind, h, er,
         Zbar, Qbar, Kdbar, pdbar, Ffbar)

    results = opt.root(cge_system, pvec, args=(cge_args),
                       method='lm', tol=1e-5)
    pprime = results.x


    pyprime = pprime[0:len(ind)]
    pqprime = pprime[len(ind):len(ind)*2]
    pfprime= pprime[len(ind)*2:12]

    pvec = pprime

    temp = cge_system(pvec,cge_args)

    Qprime = eqQ(gamma, deltam, deltad, eta, M, D)
    pdprime = eqpd(gamma, deltam, deltad, eta, Qprime, pqprime, D)
    Zprime = eqZ(theta, xie, xid, phi, E, D)
    Zprime = Zprime.iloc[0]
    Kdprime = eqKd(g, Sp, lam, pqprime)
    Ffprime = Ff0
    Ffprime['CAP'] = R * Kk * (lam * pq).sum() / pf[1]
'''
    dist = (((Zbar - Zprime) ** 2 ) ** (1 / 2)).sum()
    print('Distance at iteration ', tpi_iter, ' is ', dist)
    pdbar = xi * pdprime + (1 - xi) * pdbar
    Zbar = xi * Zprime + (1 - xi) * Zbar
    Kdbar = xi * Kdprime + (1 - xi) * Kdbar
    Qbar = xi * Qprime + (1 - xi) * Qbar
    Ffbar = xi * Ffprime + (1 - xi) * Ffbar
'''

'''
    bop_error = eqbop(pWe, pWm, E, M, Sf, Fsh, er)

    pd = eqpd(gamma, deltam, deltad, eta, Q, pq, D)
    Z = eqZ(theta, xie, xid, phi, E, D)
    Kd = eqKd(g, Sp, lam, pq)
    Q = eqQ(gamma, deltam, deltad, eta, M, D)
'''
