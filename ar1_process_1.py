import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from functools import partial
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF
from ts_analysis import pacf

__author__ = "M. G. Schultz"
__email__ = "m.schultz@fz-juelich.de"

class AR1Data:
    """
    Class for generating data series and time series with auto-correlation. The data can be drawn
    from different underlying frequency distributions ('normal', 'gamma', ...).

    Usage: initialize object with the desired parameters of the distribution, then call generate() method.
        This will return a pandas dataframe. Use describe() to obtain some statistics about the data.

    Examples:
        1) norm_no_autocorr = AR1Data(mu=30., sigma=5.)
           data = norm_no_autocorr.generate(N=100000)
        2) gamma_data = AR1Data('gamma', mu=30., sigma=5., phi=0.8).generate(100000)
    """
    _ALLOWED_DISTR = ['normal', 'gamma']

    def __init__(self, distribution='normal', mu=0., sigma=1., phi=0.):
        """
        initializes the AR1Data object with parameters defining the underlying frequency distribution
        and the auto-correlation phi.

        :param distribution: name of the distribution from which samples shall be drawn to generate
            the AR1 series. One of 'normal' (default), 'gamma', [lognormal, ... to be implemented]
        :param mu: the mean value of the AR1 data series. Note that this doesn't necessarily
            correspond to the 'loc' parameter in the numpy random distributions as in some cases
            a correction neds to be applied because of the auto-correlation.
        :param sigma: the standard deviation of the AR1 series. As with mu this value may need to
            be transformed before the distribution function is called.
        :param phi: the auto-correlation coefficient. Range 0..0.9999; default is 0, i.e. a
            series without auto-correlation will be genrated by default.
        """
        self.distr = distribution.lower()
        if distribution not in self._ALLOWED_DISTR:
            raise NotImplementedError(f'Distribution {distribution} has not been implemented!')
        self.mu = mu
        self.sigma = sigma
        self.phi = phi
        self.data = None

    def prepare_dist(self):
        """
        mangles the distribution parameters and returns a ufunc which only takes the requested
        number of samples as argument. This is needed by the arma_generate_sample method.
        """
        np.random.seed(123456)     # ensure reproducible results; can be commented out in production
        if self.distr == "normal":
            loc = (1. - self.phi) * self.mu
            var = self.sigma**2 * (1. - self.phi**2)
            params = {'loc': loc, 'scale': np.sqrt(var)}
            return partial(np.random.normal, **params)
        elif self.distr == "gamma":
            var = self.sigma ** 2
            scale = (1. + self.phi ** 2) * var / self.mu
            shape = (1. - self.phi) * self.mu / scale
            params = {'scale': scale, 'shape': shape}
            return partial(np.random.gamma, **params)
        elif self.distr == "lognormal":
            params = {}
            return partial(np.random.lognormal, **params)
        elif self.distr == "exponential":
            params = {}
            return partial(np.random.exponential, **params)

    def generate(self, N=10000, tstart=None, freq='H', tz=None, normalize_time=False, colname='values',
                 describe=False):
        """
        generate the AR1 data series, optionally as time series

        :param N: numbe of samples to generate (default 10000)
        :param tstart: if given, a time series with freq freq will be generated beginning at the
            given date. See pandas.date_range for details
        :param freq: defines the time interval between samples if tstart is given
        :param tz: use timezone for time index (see pandas.date_range)
        :param normalize_time: normalize the time index (see pandas.date_range)
        :param colname: name of the data column in the result dataframe (default: 'values')
        :param describe: print a description of the generated data (default: False)

        Note: when generating very large time series, make sure that tstart + N*freq will not lead to
            out-of-range error
        """
        rnd = self.prepare_dist()
        ar = [1., -self.phi]      # zero-lag phi, lag-1 phi
        ma = [1.]                   # zero-lag ma
        X = arma_generate_sample(ar, ma, N, distrvs=rnd, scale=1.)
        if tstart is None:
            # generate plain index values
            index = list(range(N))
        else:
            # generate time index for time series
            index = pd.date_range(start=tstart, periods=N, freq=freq, tz=tz, normalize=normalize_time,
                                  name='time')
        self.data = pd.DataFrame(X, index=index, columns=[colname])
        # if self.distribution == 'normal':
        #    X = X * self.__sigma + self.__mu
        if describe:
            self.describe()
        return self.data


    def describe(self):
        """
        prints some statistics of the generated AR1 series
        """
        if self.data is None:
            raise ValueError("No data has been generated yet!")
        df = self.data
        colname = df.columns[0]
        print(f'Column name: {df.columns[0]}, N = {df[colname].count()}')
        print(f'    mean = {df[colname].mean():.4f}  (requested: {self.mu:.4f})')
        print(f'    sigma = {df[colname].std():.4f}  (requested: {self.sigma:.4f})')
        phi, _ = pacf(df[colname], print_results=False)
        print(f'    phi = {phi[1]:.4f}  (requested: {self.phi:.4f})')


if __name__ == "__main__":
    # Example 1
    norm_no_autocorr = AR1Data(mu=30., sigma=5.)
    data = norm_no_autocorr.generate(N=100000)
    norm_no_autocorr.describe()
    # Example 2
    gamma_data = AR1Data('gamma', mu=30., sigma=5., phi=0.8).generate(50000, tstart='2000-01-01 00:00',                                                               colname='gamma', describe=True)
    # Example 3
    norm_data = AR1Data('normal', mu=120., sigma=15., phi=0.45).generate(50000, colname='normal', describe=True)
