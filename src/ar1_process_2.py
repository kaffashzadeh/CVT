"""
This program is written to make a synthetic time series based on AR(1).
"""

# imports Python packages for data science
import numpy as np
import pandas as pd
from statsmodels.tsa.arima_process import arma_generate_sample
from functools import partial
from scipy.stats import norm
from statsmodels.distributions.empirical_distribution import ECDF


class SynthTS:

    def __init__(self, **kwargs):
        """
        It initializes the attributes for the AR1 class.
        """
        self.distribution = None     # distribution name
        # self.__dist = dist
        # self.__shape = shape
        self.__res = None
        self.__size = None
        self.__mu = 0 #0
        self.__sigma = 2#1
        self.__phi = 0
        self.__df = None
        # parse kwargs
        if 'loc' in kwargs:
            self.__mu = kwargs.pop('loc')
        if 'mean' in kwargs:
            self.__mu = kwargs.pop('mean')
        if 'sigma' in kwargs:
            self.__sigma = kwargs.pop('sigma')
        if 'phi' in kwargs:
            self.__phi = kwargs.pop('phi')

    def get_distrvs(self, **kwargs):
        """
        It gets a callable object to generate random samples of specific distributions

        :param distribution: name of the distribution. One of normal, lognormal, exponential, gamma
        :param kwargs: parameters that need to be specified to define the distribution, e.g. shape and scale
                       for a gamma distribution.
        :return: callable object

        Example:
              gamma = get_distrvs("gamma", shape=0.7, scale=0.5)
        """
        dist = self.distribution.lower()
        # return {'normal': partial(np.random.normal, **kwargs),
        #         'exponential': partial(np.random.exponential, **kwargs),
        #         'lognormal': partial(np.random.lognormal, **kwargs),
        #         'gamma': partial(np.random.gamma, **kwargs)}.get(dist)
        # print(kwargs)
        np.random.seed(100)
        if dist == "normal":
            return partial(np.random.normal, **kwargs)
        elif dist == "lognormal":
            return partial(np.random.lognormal, **kwargs)
        elif dist == "exponential":
            return partial(np.random.exponential, **kwargs)
        elif dist == "gamma":
            return partial(np.random.gamma, **kwargs)
        else:
            raise ValueError("Invalid distribution name")

    def generate_ar1_values(self, N=10000, phi=0.8, **kwargs):
        """
        It generates the values (series) based on AR(1) model.

        :param N: the sample size
        :param mean: the requested mean value
        :param stddev: the requested standard deviation
        :param phi: the lag-1 auto-correlation parameter
        :param kwargs: any keywords to further specify the distribution function
        :return: a numpy array with auto-correlated values based on the specified sample distribution

        Notes:
            - Calling the statsmodels.tsa.arima_process sample generation
            See documentation at https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_process.arma_generate_sample.html#statsmodels.tsa.arima_process.arma_generate_sample
            - In this sample generator, a random series with the given underlying distribution is generated first
            and then this series is passed through a low-pass filter from scipy.signal.
            See https://en.wikibooks.org/wiki/Signal_Processing/Digital_Filters: Autoregressive Filters (AR)
            for an explanation.
            - The arima sample generator allows for arbitrary length AR and MA arguments. Here, we only
            set AR(1) parameters. Note the sign change of the lag-1 auto correlation parameter phi
            (explained in the original documentation).
            - Further reading:
            https://stats.stackexchange.com/questions/286298/how-to-create-a-markov-chain-with-gamma-marginal-distribution-and-ar1-coeffici
            for getting parameters of a AR(1) gamma
        """
        ar = [1., -self.__phi]      # zero-lag phi, lag-1 phi
        ma = [1.]            # zero-lag ma
        if self.distribution == 'gamma':
            # obtain distribution params from mean and var requests
            # theoretical mean of AR(1) process with gamma dist is shape*scale/(1-rho)
            # theoretical variance is shape*scale**2/(1-rho**2)
            mean = kwargs.pop('mean', self.__mu)
            var = kwargs.pop('var', self.__sigma**2)
            scale = (1.+phi**2)*var/mean
            shape = (1.-phi)*mean/scale
            print(f'requested: mean={mean}, var={var}, params: shape={shape}, scale={scale}')
            kwargs['shape'] = shape
            kwargs['scale'] = scale
        if self.distribution == 'normal':
            print('## in generate_ar1_values: kwargs = ', kwargs)
            loc = kwargs.pop('loc', 0.)
            scale = kwargs.pop('scale', 1.)  # apply correction to variance
            print('## for dist = normal: loc, scale = ', loc, scale)
        if self.distribution == 'lognormal':
            loc = kwargs.pop('loc', 0.)
            # scale = kwargs.pop('scale', 1.) * np.sqrt(1. - self.__phi ** 2)
        # distrvs is a function of random number generator
        rnd = self.get_distrvs(**kwargs)
        del rnd.keywords['distribution']
        X = arma_generate_sample(ar, ma, N, distrvs=rnd, scale=1.)
        if self.distribution == 'normal':
            print('>>>>>>>>>>>>>>>>>>.')
            print(self.__sigma, self.__mu)
            X = X * self.__sigma + self.__mu
        return X

    def estimate_prob_ecdf(self):
        """
        It uses empirical distribution to calculate the probability.
        """
        print(self.__df.mean(), self.__df.std(), self.__df.corrwith(self.__df.shift(1)))
        sample = list(self.__df['value'].values)
        ecdf = ECDF(sample)
        for n in [-2., -1., 0., 1., 2.]:
            x = self.__mu + n * self.__sigma
            p1 = (ecdf(x + self.__res) - ecdf(x-self.__res))
            distribution = norm(loc=self.__mu + self.__phi * (x + self.__mu),
                                scale=np.sqrt((1. - self.__phi ** 2) * self.__sigma))
            p2 = distribution.cdf(x + self.__res) - distribution.cdf(x-self.__res)
            print(f'for x={x}: p1={p1}, p2={p2}')

    def generate_ar1(self, N=10000, freq='M', start='2020-01-01 00:00', end=None, tz=None,
                     normalize_time=False, name='time', **kwargs):
        """
        It generates an AR1 time series as pandas DataFrame.

        :param start: start date (from pd.date_range)
        :param end: end date (from pd.date_range)
        :param N: number of samples (periods in pd.date_range, also used for generate_ar1_values)
        :param freq: frequency (from pd.date_range), default are hourly values
        :param tz: time zone specification (from pd.date_range)
        :param normalize_time: adjust start and end date to midnight (from pd.date_range)
        :param name: name of the datetime index column (from pd.date_range)
        :param kwargs: all other arguments to generate_ar1_values e.g. shape, scale, etc.
        :return: a pandas data-frame with the AR(1) time series

        Note:
            It calls generate_ar1_values and adds a datetime index.
        """
        self.distribution = kwargs['distribution']
        dti = pd.date_range(start=start, end=end, periods=N, freq=freq, tz=tz, normalize=normalize_time,
                            name=name)
        print("## in generate_ar1: kwargs = ", kwargs)
        values = self.generate_ar1_values(N, **kwargs)
        self.__df = pd.DataFrame(values, index=dti, columns=['value'])
        # self.__res=1
        # sns.distplot(self.__df)
        # self.__df.plot(kind='hist')
        # plt.show()
        # res = 0.1
        # self.add_cve()
        # self.__mu = self.__df.mean()
        # self.__sigma = self.__df.var()
        # self.estimate_prob_ecdf()
        # plt.show()
        return self.__df