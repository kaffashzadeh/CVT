# !user/bin/env python

"""
This program is written to run the CVT test on an input data time series.
Theoretical concept of this test was developed and provided by Kai-Lan Chang and Martin G. Schultz!
"""

# imports Python packages for data science
import pandas as pd
import numpy as np
from scipy.stats import norm, lognorm, gamma

__authors__ = "Najmeh Kaffashzadeh"
__email__ = "n.kaffashzadeh@fz-juelich.de"


class CVT:
    """
    The CVT class estimates the plausibility of each individual values in a data time series.
    """

    name = 'constant value test'

    def __init__(self, ts=None, mu=None, sigma=None, corr=None, res=None):
        """
        It creates the variables associated with the CVT class.

        :param ts: input data time series
        :param mu: mean
        :param sigma: standard deviation
        :param corr: auto-correlation
        :param res: data digital (numerical) resolution
        """
        self.__ds = ts
        self.__mu = mu
        self.__r = corr
        self.__res = res
        self.__sigma = sigma
        # windows size for rolling function
        self.__window = None
        # the number of CVs (i.e. t in the cvt eq)
        self.__ds_count = None
        # estimated probability
        self.__prob = None
        # values variance
        self.__var = self.__sigma ** 2

    def check_distribution(self):
        """
        It finds and checks the data distribution to be in a given category (i.e. normal, lognormal,
        and gamma).

        :return True: if data fits to one of the normal, lognormal, or gamma distribution
        :return False: if data does not fit to any of the normal, lognormal, or gamma distribution

        Note:
            fit: maximum likelihood estimation of distribution parameters, including location
                and scale
            nnlf: negative log likelihood function
            fit_loc_scale: estimation of location and scale when shape parameters are given
            expect: calculate the expectation of a function against the pdf or pmf
            distribution fits can be extended (see https://docs.scipy.org/doc/scipy/reference/stats.html)
            Source: https://stackoverflow.com/questions/21623717/fit-data-to-all-possible-distributions-and-return-the-best-fit
        """
        distributions = [norm, lognorm, gamma]
        mles = []
        for distribution in distributions:
            pars = distribution.fit(self.__ds.dropna())
            mle = distribution.nnlf(pars, self.__ds.dropna())
            mles.append(mle)
        best_fit = sorted(zip(distributions, mles), key=lambda d: d[1])[0]
        if best_fit[0].name in 'norm':
            print(f'Best fit was reached using {best_fit[0].name}')
            return True
        elif best_fit[0].name in ('lognorm', 'gamma'):
            print(f'Best fit was reached using {best_fit[0].name}. The CVT is not recommended.')
            return False
        else:
            print('None of the distributions_outputs were fitted!')
            return False

    def take_serial_difference(self):
        """
        It calculates the serial difference, i.e. X[t]-X[t-1].

        :return: absolute difference
        """
        return self.__ds.diff(periods=1).abs()

    def detect_CVs(self):
        """
        It checks if there is successive identical values in the data.

        :return True: if there is CVs
        :return False: if there is no CVs

        Note:
              It not only detects the CVs, but also saves them in self.__ds_count
        """
        ds_delta = self.take_serial_difference()
        self.count_CVs(ds=ds_delta.iloc[:])
        count = np.count_nonzero(self.__ds_count.values)
        if count > 0:
            print('CVs were detected.')
        else:
            print('No CVs were detected.')
        return count

    def count_CVs(self, ds=None):
        """
        It counts the number of successive identical values.

        :param ds: input data time series (here the serial differences)

        Note:
            The method is based on the number of cumulative zeros.
        """
        ds_count = ds.groupby((ds != ds.shift(1)).cumsum()).transform('count')
        mask = ds_count.gt(0) & ds.eq(0)
        self.__ds_count = ds_count.loc[mask].reindex(ds.index, fill_value=0)
        self.__ds_count.replace(to_replace=0, method='bfill', limit=1, inplace=True)
        self.__ds_count = self.__ds_count.apply(lambda x: x + 1 if x != 0 else x)
        return self.__ds_count

    def describe(self):
        """
        It describes data values statistics (whole time series).
        """
        if self.__mu is None:
            self.__mu = self.__ds[self.__ds_count == 0].dropna().mean()
        # if self.__sigma is None:
        self.__sigma = np.sqrt(self.__var)    # this fails if __var hasn't been defined before
        self.__var = self.__ds[self.__ds_count == 0].dropna().var()
        # else:
        #     self.__var = self.__sigma ** 2
        if self.__r is None:
            self.__r = self.__ds[self.__ds_count == 0].dropna().autocorr(lag=1)
        if self.__res is None:
            self.find_digit_resolution()
        print('mu=', self.__mu, 'res=', self.__res, 'corr=', self.__r, 'sigma=', self.__sigma)

    def find_digit_resolution(self):
        """
        It checks the digit resolution of the data (e.g. 1, 0.1, 0.01, 0.001).
        """
        # self.__res = (np.min(np.abs(np.diff(np.unique(self.__ds[self.__ds_count == 0].dropna()), 1))), 10)
        self.__res = np.min(np.abs(np.diff(np.unique(self.__ds[self.__ds_count == 0].dropna()), 1)))

    def calc_cond_prob(self, x, t):
        """
        It calculates the probability.

        :param x: data value points
        :param t: the number of CVs
        """
        # print(self.__mu, self.__sigma, self.__r)
        distribution = norm(loc=self.__mu + self.__r * (x - self.__mu),
                            scale=np.sqrt((1. - self.__r ** 2) * self.__sigma**2))
        # print(distribution.cdf(self.__sigma)-distribution.cdf(-self.__sigma))
        print(t, x, self.__mu, self.__res, self.__r, self.__var,
              (distribution.cdf(x) - distribution.cdf(float(self.__res))) ** (t-1))
        return (distribution.cdf(x + float(self.__res)) - distribution.cdf(x - float(self.__res))) ** (t - 1)
        # if x == 2:    # here 2 refers to the zero value for this station
        #     return (distribution.cdf(x+0.5) - distribution.cdf(x)) ** (t - 1)
        #     # return (distribution.cdf(0) - distribution.cdf(x - float(self.__res))) ** (t-1)
        # else:
        #     return (distribution.cdf(x + float(self.__res)) - distribution.cdf(x - float(self.__res))) ** (t-1)

    def check_missing(self):
        """
        It checks missing data.

        Note:
             It re-samples the data time series and inserts NaN for non-recorded missing values.
        """
        new_index = pd.date_range(start=self.__ds.index[0], end=self.__ds.index[-1], freq='H')
        self.__ds = self.__ds.reindex(new_index, fill_value=np.nan)

    def run(self):
        """
        It runs the CVT.
        """
        self.check_missing()
        if self.detect_CVs():
            self.describe()
            df = pd.concat([self.__ds, self.__ds_count], join='outer', axis=1)
            df.columns = ['data values', 'number of CVs']
            self.__prob = df.apply(lambda x: self.calc_cond_prob(x['data values'], x['number of CVs'])
                                   if (x['data values'] != np.NaN and x['number of CVs'] != 0) else 1, axis=1)
            if (self.__prob.max() > 1.) | (self.__prob.min() < 0.):
                raise ValueError
            df = pd.concat([df, self.__prob], join='outer', axis=1)
            df.columns = ['data values', 'number of CVs', 'probability']
            mask_nan = np.isnan(df['data values'])
            df['number of CVs'].mask(mask_nan, other=np.NaN, inplace=True)
            df['probability'].mask(mask_nan, other=np.NaN, inplace=True)
            print(f"prob_min = {df['probability'].min()}", f"prob_max = {df['probability'].max()}")
            return df
        else:
            pass


if __name__ == "__main__":
    pass
