'''
------------------------------------------------------------------------
This program simulates effective tax data for 1,600 different tax
functions. S x T sets of data are for labor income by S ages and T
years, and S x T sets of data are for capital income by S ages and T
years.

This Python script calls the following functions:
    sumsq: calculate weighted sum of squared percent deviations of
           estimated tax rates from data

This Python script outputs the following pickle files:
    DataObs_n.pkl: data_n and obs_n objects
    DataObs_b.pkl: data_b and obs_b objects
------------------------------------------------------------------------
'''

# Import packages
import numpy as np
import numpy.random as rnd
import scipy.optimize as opt
try:
    import cPickle as pickle
except:
    import pickle
import matplotlib
import matplotlib.pyplot as plt

'''
------------------------------------------------------------------------
Declare parameters and define functions
------------------------------------------------------------------------
S       = integer in [3,80], number of ages for which we have data
T       = integer >= 1, number of time periods forward to compute
graph_n = boolean, =True if graph the estimate labor income tax rate
          versus the data
graph_b = boolean, =True if graph the estimate capital income tax rate
          versus the data
------------------------------------------------------------------------
'''
S = int(80)
T = int(10)
graph_n = False
graph_b = False


def sumsq(params, *objs):
    '''
    --------------------------------------------------------------------
    This function generates the sum of squared percent deviations of
    predicted values of effective tax rates as a function of income and
    functional form parameters.

    tau(y) = (maxt - mint)*(A*y^2 + B*y)/(A*y^2 + B*y + C) + mint
    --------------------------------------------------------------------
    params     = [5,] vector, guesses for maxt, mint, A, B, C
    maxt       = scalar > 0, guess for maximum value of tax rate
                 function
    mint       = scalar, guess for minimum value of tax rate function
    A          = scalar > 0, tax function parameter A
    B          = scalar > 0, tax function parameter B
    C          = scalar > 0, tax function parameter B
    objs       = (3,) tuple, array objects passed in to function
    avinc      = [n,] vector, average AGI of income bins
    avgtax_dta = [n,] vector, average effective tax rate for each
                 income bin
    incwgts    = [n,] vector, weights on each (income, tax) point for
                 estimation
    avgtax_est = [n,] vector, average estimated effective tax rate for
                 each income bin
    pctdev     = [n,] vector, weighted percent deviation (times 100) of
                 estimated tax rates from data tax rates
    wssqdev    = scalar > 0, weighted sum of squared percent deviations

    returns: wssqdev
    --------------------------------------------------------------------
    '''
    maxt, mint, A, B, C = params
    avginc, avgtax_dta, incwgts = objs

    avgtax_est = ((maxt - mint) * ((A * avginc ** 2 + B * avginc) /
        (A * avginc ** 2 + B * avginc + C)) + mint)
    pctdev = ((avgtax_est - avgtax_dta) / avgtax_dta) * 100 * incwgts
    wssqdev = (pctdev ** 2).sum()

    return wssqdev


'''
------------------------------------------------------------------------
Estimate labor income tax rate function of the form:
tau(y) = (maxt - mint)*(A*y^2 + B*y)/(A*y^2 + B*y + C) + mint
------------------------------------------------------------------------
avginc_n      = [18,] vector, average AGI for each income category
avgtax_n      = [18,] vector, average effective labor income tax rate in
                income bins
incwgts_n     = [18,] vector, weights on each (income, tax) point for
                estimation
tau_n_objs    = (3,) tuple, objects to be passed in to minimizer
maxt_n_init   = scalar > 0, initial guess for maximum value of estimated
                tax rate function
mint_n_init   = scalar, initial guess for minimum value of estimated tax
                rate function
A_n_init      = scalar >= 0, initial guess for tax function parameter A
B_n_init      = scalar >= 0, initial guess for tax function parameter B
C_n_init      = scalar >= 0, initial guess for tax function parameter C
params_n_init = [5,] vector, parameters chosen to minimize function
bnds_n        = (5,) tuple, each element is a (min, max) tuple for upper
                and lower bounds of each of the 5 parameters
params_n      = [5,] vector, estimated parameters for labor income tax
                function
maxt_n        = scalar > 0, estimated maximum value of labor income tax
                rate function
mint_n        = scalar, estimated minimum value of labor income tax rate
                function
A_n           = scalar >= 0, estimated A parameter of labor income tax
                rate function
B_n           = scalar >= 0, estimated B parameter of labor income tax
                rate function
C_n           = scalar >= 0, estimated C parameter of labor income tax
                rate function
avgtax_n_est  = [18,] vector, estimated labor income tax rates for each
                income bin
------------------------------------------------------------------------
'''
avginc_n = np.array([2616.14, 7603.34, 12505.28, 17434.36, 22416.35,
           27436.52, 34782.76, 44765.24, 61553.10, 86452.04, 134214.26,
           285681.09, 677280.38, 1208952.52, 1720703.43, 2978820.89,
           6839675.60, 30911333.39])
avgtax_n = np.array([0.0231, 0.0412, 0.0562, 0.0684, 0.0788, 0.0866,
           0.0977, 0.1130, 0.1272, 0.1500, 0.1773, 0.2444, 0.3154,
           0.3306, 0.3322, 0.3467, 0.3632, 0.3730])
incwgts_n = np.log(avginc_n)
tau_n_objs = (avginc_n, avgtax_n, incwgts_n)
maxt_n_init = avgtax_n.max()
mint_n_init = avgtax_n.min()
A_n_init = 0.00001
B_n_init = 3.0
C_n_init = 400000
params_n_init = np.array([maxt_n_init, mint_n_init, A_n_init, B_n_init,
                C_n_init])
bnds_n = ((0, None), (None, None), (0, None), (0, None), (0, None))
params_n = opt.minimize(sumsq, params_n_init, args=(tau_n_objs),
           method="L-BFGS-B", bounds=bnds_n, tol=1e-15)
maxt_n, mint_n, A_n, B_n, C_n = params_n.x
avgtax_n_est = ((maxt_n - mint_n) *
    ((A_n * avginc_n ** 2 + B_n * avginc_n) /
    (A_n * avginc_n ** 2 + B_n * avginc_n + C_n)) + mint_n)

if graph_n == True:
    # Plot average labor tax from data against estimated average labor tax
    fig, ax = plt.subplots()
    plt.plot(avginc_n, avgtax_n, 'b', label='Actual tax rate')
    plt.plot(avginc_n, avgtax_n_est, 'r--', label='Estimated tax rate')
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.legend(loc='center right')
    plt.title('Actual vs. Estimated Effective Labor Income Tax Rate')
    plt.xlabel(r'Average income (dollars)')
    plt.ylabel(r'Effective tax rate')
    # plt.savefig('labinctax')
    plt.show()

    # Plot average labor tax from data against estimated average labor tax
    # with log scale on x-axis (income)
    fig, ax = plt.subplots()
    plt.plot(np.log(avginc_n), avgtax_n, 'b', label='Actual tax rate')
    plt.plot(np.log(avginc_n), avgtax_n_est, 'r--', label='Estimated tax rate')
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.legend(loc='center right')
    plt.title('Actual vs. Estimated Effective Labor Income Tax Rate')
    plt.xlabel(r'Average log income (dollars)')
    plt.ylabel(r'Effective tax rate')
    # plt.savefig('labloginctax')
    plt.show()


'''
------------------------------------------------------------------------
Estimate capital income tax rate function of the form:
tau(y) = (maxt - mint)*(A*y^2 + B*y)/(A*y^2 + B*y + C) + mint
------------------------------------------------------------------------
avginc_b      = [17,] vector, average AGI for each income category
avgtax_b      = [17,] vector, average effective labor income tax rate in
                income bins
incwgts_b     = [17,] vector, weights on each (income, tax) point for
                estimation
tau_b_objs    = (3,) tuple, objects to be passed in to minimizer
maxt_b_init   = scalar > 0, initial guess for maximum value of estimated
                tax rate function
mint_b_init   = scalar, initial guess for minimum value of estimated tax
                rate function
A_b_init      = scalar >= 0, initial guess for tax function parameter A
B_b_init      = scalar >= 0, initial guess for tax function parameter B
C_b_init      = scalar >= 0, initial guess for tax function parameter C
params_b_init = [5,] vector, parameters chosen to minimize function
bnds_b        = (5,) tuple, each element is a (min, max) tuple for upper
                and lower bounds of each of the 5 parameters
params_b      = [5,] vector, estimated parameters for capital income tax
                function
maxt_b        = scalar > 0, estimated maximum value of capital income tax
                rate function
mint_b        = scalar, estimated minimum value of capital income tax rate
                function
A_b           = scalar >= 0, estimated A parameter of capital income tax
                rate function
B_b           = scalar >= 0, estimated B parameter of capital income tax
                rate function
C_b           = scalar >= 0, estimated C parameter of capital income tax
                rate function
avgtax_b_est  = [17,] vector, estimated labor income tax rates for each
                income bin
------------------------------------------------------------------------
'''
avginc_b = np.array([7603.34, 12505.28, 17434.36, 22416.35, 27436.52,
           34782.76, 44765.24, 61553.10, 86452.04, 134214.26, 285681.09,
           677280.38, 1208952.52, 1720703.43, 2978820.89, 6839675.60,
           30911333.39])
avgtax_b = np.array([0.0273, 0.0379, 0.0442, 0.0494, 0.0535, 0.0611,
           0.0727, 0.0857, 0.1054, 0.1266, 0.2006, 0.2705, 0.2795,
           0.2767, 0.2852, 0.2919, 0.3012])
incwgts_b = np.log(avginc_b)
tau_b_objs = (avginc_b, avgtax_b, incwgts_b)
maxt_b_init = avgtax_b.max()
mint_b_init = avgtax_b.min()
A_b_init = 0.00001
B_b_init = 2.0
C_b_init = 500000
params_b_init = np.array([maxt_b_init, mint_b_init, A_b_init, B_b_init,
                C_b_init])
bnds_b = ((0, None), (None, None), (0, None), (0, None), (0, None))
params_b = opt.minimize(sumsq, params_b_init, args=(tau_b_objs),
           method="L-BFGS-B", bounds=bnds_b, tol=1e-15)
maxt_b, mint_b, A_b, B_b, C_b = params_b.x
avgtax_b_est = ((maxt_b - mint_b) *
    ((A_b * avginc_b ** 2 + B_b * avginc_b) /
    (A_b * avginc_b ** 2 + B_b * avginc_b + C_b)) + mint_b)

if graph_b == True:
    # Plot average labor tax from data against estimated average labor tax
    fig, ax = plt.subplots()
    plt.plot(avginc_b, avgtax_b, 'b', label='Actual tax rate')
    plt.plot(avginc_b, avgtax_b_est, 'r--', label='Estimated tax rate')
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.legend(loc='center right')
    plt.title('Actual vs. Estimated Effective Capital Income Tax Rate')
    plt.xlabel(r'Average income (dollars)')
    plt.ylabel(r'Effective tax rate')
    # plt.savefig('Capinctax')
    plt.show()

    # Plot average labor tax from data against estimated average labor tax
    # with log scale on x-axis (income)
    fig, ax = plt.subplots()
    plt.plot(np.log(avginc_b), avgtax_b, 'b', label='Actual tax rate')
    plt.plot(np.log(avginc_b), avgtax_b_est, 'r--', label='Estimated tax rate')
    plt.grid(b=True, which='major', color='0.65',linestyle='-')
    plt.legend(loc='center right')
    plt.title('Actual vs. Estimated Effective Capital Income Tax Rate')
    plt.xlabel(r'Average log income (dollars)')
    plt.ylabel(r'Effective tax rate')
    # plt.savefig('Caploginctax')
    plt.show()


'''
------------------------------------------------------------------------
Generate block of 800 data series of average AGI levels and effective
labor income tax rates for each age (s) and time period (t).
------------------------------------------------------------------------
avgobs_n   = integer > 0, average number of observations for each data
             series of labor income and corresponding tax rates
stdvobs_n  = scalar > 0, standard deviation of observations for each
             data series of labor income and corresponding tax rates
obs_n      = [S, T] matrix of integers, number of observations for each
             data series of average AGI and corresponding effective
             labor income tax rates
maxtvals_n = [S, T] matrix > 0, maximum values of labor income tax rate
             functions for each age (s) and time period (t)
mintvals_n = [S, T] matrix, minimum values of labor income tax rate
             functions for each age (s) and time period (t)
A_nbd      = scalar > 0, value of A with lower bound strictly greater
             than 0
B_nbd      = scalar > 0, value of B with lower bound strictly greater
             than 0
C_nbd      = scalar > 0, value of C with lower bound strictly greater
             than 0
Avals_n    = [S, T] matrix > 0, values for labor income tax function
             parameter A for each age (s) and time period (t)
Bvals_n    = [S, T] matrix > 0, values for labor income tax function
             parameter B for each age (s) and time period (t)
Cvals_n    = [S, T] matrix > 0, values for labor income tax function
             parameter C for each age (s) and time period (t)
Dvals_n    = [S, T] matrix > 0, values for labor income tax function
             parameter D (maxt - mint) for each age (s) and time period
             (t)
tfparams_n = [S, T, 4] array, four tax function parameter values
             (A,B,C,D) for each age (s) and time period (t)
data_n     = [S, T, N, 2] array, two data series of maximum length N of
             average AGI and effective tax rates for each age (s) and
             time period (t). Missing values = 999. Object obs_n tells
             how many data points are nonmissing for each (s, t). Income
             series are in data_n[s, t, :, 0] and tax rate series are in
             data_n[s, t, :, 1]
avinc      = (n,) vector, ascending average income levels for which n
             equals the number of nonmissing observations for particular
             (s,t) from obs_n
avtax_noer = (n,) vector, n predicted tax rates for each (s, t) as a
             function of avinc and corresponding estimated parameters
dict_n     = dictionary, stores data_n and obs_n in dictionary
pkl_path   = string, path where pickle file is saved
------------------------------------------------------------------------
'''
avgobs_n = int(200)
stdvobs_n = 5.
obs_n = np.rint(rnd.normal(avgobs_n, stdvobs_n, (S, T))).astype(int)
maxtvals_n = np.maximum(rnd.normal(maxt_n, maxt_n/100, (S, T)), 1e-14)
mintvals_n = rnd.normal(mint_n, np.absolute(mint_n)/100, (S, T))
A_nbd = np.maximum(A_n, 1e-14)
B_nbd = np.maximum(B_n, 1e-14)
C_nbd = np.maximum(C_n, 1e-14)
Avals_n = np.maximum(rnd.normal(A_nbd, A_nbd/100, (S, T)), 1e-14)
Bvals_n = np.maximum(rnd.normal(B_nbd, B_nbd/100, (S, T)), 1e-14)
Cvals_n = np.maximum(rnd.normal(C_nbd, C_nbd/100, (S, T)), 1e-14)
Dvals_n = np.maximum(maxtvals_n - mintvals_n, 1e-14)
tfparams_n = np.dstack((Avals_n, Bvals_n, Cvals_n, Dvals_n))

data_n = 999 * np.ones((S, T, obs_n.max(), 2))
for s in xrange(S):
    for t in xrange(T):
        avinc = np.exp(np.linspace(np.log(min(avginc_n)),
                np.log(max(avginc_n)), obs_n[s, t]))
        data_n[s, t, :obs_n[s, t], 0] = avinc
        avtax_noer = ((maxtvals_n[s, t] -
            mintvals_n[s, t]) * (Avals_n[s, t] * avinc ** 2 +
            Bvals_n[s, t] * avinc) / (Avals_n[s, t] * avinc ** 2 +
            Bvals_n[s, t] * avinc + Cvals_n[s, t]) + mintvals_n[s, t])
        data_n[s, t, :obs_n[s, t], 1] =  (avtax_noer +
            rnd.normal(0, avtax_noer / 20, len(avtax_noer)))

dict_n = dict([('data_n', data_n), ('obs_n', obs_n)])
pkl_path = "DataObs_n.pkl"
pickle.dump(dict_n, open(pkl_path, "wb"))

'''
------------------------------------------------------------------------
Generate block of 800 data series of average AGI levels and effective
capital income tax rates for each age (s) and time period (t).
------------------------------------------------------------------------
avgobs_b   = integer > 0, average number of observations for each data
             series of capital income and corresponding tax rates
stdvobs_b  = scalar > 0, standard deviation of observations for each
             data series of capital income and corresponding tax rates
obs_b      = [S, T] matrix of integers, number of observations for each
             data series of average AGI and corresponding effective
             capital income tax rates
maxtvals_b = [S, T] matrix > 0, maximum values of capital income tax
             rate functions for each age (s) and time period (t)
mintvals_b = [S, T] matrix, minimum values of capital income tax rate
             functions for each age (s) and time period (t)
A_bbd      = scalar > 0, value of A with lower bound strictly greater
             than 0
B_bbd      = scalar > 0, value of B with lower bound strictly greater
             than 0
C_bbd      = scalar > 0, value of C with lower bound strictly greater
             than 0
Avals_b    = [S, T] matrix > 0, values for capital income tax function
             parameter A for each age (s) and time period (t)
Bvals_b    = [S, T] matrix > 0, values for capital income tax function
             parameter B for each age (s) and time period (t)
Cvals_b    = [S, T] matrix > 0, values for capital income tax function
             parameter C for each age (s) and time period (t)
Dvals_b    = [S, T] matrix > 0, values for capital income tax function
             parameter D (maxt - mint) for each age (s) and time period
             (t)
tfparams_b = [S, T, 4] array, four tax function parameter values
             (A,B,C,D) for each age (s) and time period (t)
data_b     = [S, T, N, 2] array, two data series of maximum length N of
             average AGI and effective tax rates for each age (s) and
             time period (t). Missing values = 999. Object obs_b tells
             how many data points are nonmissing for each (s, t). Income
             series are in data_b[s, t, :, 0] and tax rate series are in
             data_b[s, t, :, 1]
avinc      = (n,) vector, ascending average income levels for which n
             equals the number of nonmissing observations for particular
             (s,t) from obs_b
avtax_noer = (n,) vector, n predicted tax rates for each (s, t) as a
             function of avinc and corresponding estimated parameters
dict_b     = dictionary, stores data_b and obs_b in dictionary
pkl_path   = string, path where pickle file is saved
------------------------------------------------------------------------
'''
avgobs_b = int(200)
stdvobs_b = 5.
obs_b = np.rint(rnd.normal(avgobs_b, stdvobs_b, (S, T))).astype(int)
maxtvals_b = np.maximum(rnd.normal(maxt_b, maxt_b/100, (S, T)), 1e-14)
mintvals_b = rnd.normal(mint_b, np.absolute(mint_b)/100, (S, T))
A_bbd = np.maximum(A_b, 1e-14)
B_bbd = np.maximum(B_b, 1e-14)
C_bbd = np.maximum(C_b, 1e-14)
Avals_b = np.maximum(rnd.normal(A_bbd, A_bbd/100, (S, T)), 1e-14)
Bvals_b = np.maximum(rnd.normal(B_bbd, B_bbd/100, (S, T)), 1e-14)
Cvals_b = np.maximum(rnd.normal(C_bbd, C_bbd/100, (S, T)), 1e-14)
Dvals_b = np.maximum(maxtvals_b - mintvals_b, 1e-14)
tfparams_b = np.dstack((Avals_b, Bvals_b, Cvals_b, Dvals_b))

data_b = 999 * np.ones((S, T, obs_b.max(), 2))
for s in xrange(S):
    for t in xrange(T):
        avinc = np.exp(np.linspace(np.log(min(avginc_b)),
                np.log(max(avginc_b)), obs_b[s, t]))
        data_b[s, t, :obs_b[s, t], 0] = avinc
        avtax_noer = ((maxtvals_b[s, t] -
            mintvals_b[s, t]) * (Avals_b[s, t] * avinc ** 2 +
            Bvals_b[s, t] * avinc) / (Avals_b[s, t] * avinc ** 2 +
            Bvals_b[s, t] * avinc + Cvals_b[s, t]) + mintvals_b[s, t])
        data_b[s, t, :obs_b[s, t], 1] =  (avtax_noer +
            rnd.normal(0, avtax_noer / 20, len(avtax_noer)))

dict_b = dict([('data_b', data_b), ('obs_b', obs_b)])
pkl_path = "DataObs_b.pkl"
pickle.dump(dict_b, open(pkl_path, "wb"))
