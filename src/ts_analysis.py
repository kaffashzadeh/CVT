"""Various statistical analyses on time series data"""

# data should be provided as pandas DataFrame with a 'time' column

import numpy as np
from statsmodels.tsa import stattools


def acf(df, unbiased=False, nlags=10, qstat=False, fft=True, alpha=0.05, missing='none', print_results=True):
    """Calculates auto correlation function"""
    res, conf = stattools.acf(df, unbiased=unbiased, nlags=nlags, qstat=qstat, fft=fft, alpha=alpha, missing=missing)
    if print_results:
        print('auto correlation function\nres = ', res, ', conf = ', conf)
    return res, conf


def pacf(df, nlags=10, method='yw_adjusted', alpha=0.05, print_results=True):
    """Calculates auto correlation function"""
    res, conf = stattools.pacf(df, nlags=nlags, method=method, alpha=alpha)
    if print_results:
        print('partial auto correlation function\nres = ', res, ', conf = ', conf[1:3])
        if np.max(np.abs(res[2:]/res[1])) > 0.1:
            print('Warning: time series may not be appropriately analysed as AR(1) process!')
    return res, conf


""" perhaps implement some of the following?
stattools.periodogram(x)
Compute the periodogram for the natural frequency of x.

stattools.adfuller(x[, maxlag, regression, …])
Augmented Dickey-Fuller unit root test.

stattools.kpss(x[, regression, nlags, store])
Kwiatkowski-Phillips-Schmidt-Shin test for stationarity.

stattools.zivot_andrews
Zivot-Andrews structural-break unit-root test.

stattools.coint(y0, y1[, trend, method, …])
Test for no-cointegration of a univariate equation.

stattools.bds(x[, max_dim, epsilon, distance])
BDS Test Statistic for Independence of a Time Series

stattools.q_stat(x, nobs[, type])
Compute Ljung-Box Q Statistic.
"""

if __name__ == "__main__":
    import ar1_process_1
    # data = ar1_process.generate_ar1(distribution="gamma", shape=0.7, scale=1.)
    data = ar1_process_1.generate_ar1(distribution="normal", shape=0.7, scale=1.)
    acf(data)
    pacf(data)
