#!user/bin/env python

"""
This program is written to run (explicit) CVT test on a artificial (synthetic) input data time series.
For demonstration purposes and sensitivity test!
"""

# imports Python packages for data science
import numpy as np
import pandas as pd
from cvt import CVT
import matplotlib.pyplot as plt

# import local modules
from ar1_process_2 import SynthTS
from ts_analysis import pacf

__authors__ = "Najmeh Kaffashzadeh"
__email__ = "n.kaffashzadeh@fz-juelich.de"


class SyntheticData:
    """
    This class generates sample data with given variance, auto-correlation, and random noise.
    """

    name = "synthetic data"

    def __init__(self, mean=20, sigma=5, phi=0.8, res=0.1, t=2,
                 dist='normal', length=100000, out_file_name='syn_ts_ar1.csv'):
        """
        It sets and stores parameters of the series.

        :param mean: initial mean of the data time series; default 20
        :param sigma: standard deviation of the data sample; default 5
        :param phi: auto-correlation of the sample series; default 0.8
        :param res: numerical resolution of the sample values; default 0.01
        :param t: the number of CVEs
        :param length: length of series; default 240
        :param out_file_name: output file name; default syn_ts_ar1.csv
        """
        self.__mean = mean
        self.__sigma = sigma
        self.__phi = phi
        self.__res = res
        self.__t = t
        self.__dist = dist
        self.__len = length
        self.__out_fn = out_file_name
        self.__c = self.init_offset(mu=mean, phi=phi)
        self.__se = self.init_sigma_e(sigma=sigma, phi=phi)
        self.__mu = self.init_mean(mu=mean, phi=phi, c=self.__c)
        self.__s = self.init_sigma(sigma=sigma, phi=phi, sigma_e=self.__se)
        # generated time series
        self.__ts = None
        # print(self.__c, self.__se, self.__mu, self.__s)

    def init_offset(self, mu=None, phi=None):
        """
        It initializes the offset.

        :param mu: mean
        :param phi: auto-correlation
        :return: offset
        """
        return mu * (1. - phi) if all(i is not None for i in [mu, phi]) else 3.

    def init_sigma_e(self, sigma=None, phi=None):
        """
        It initializes the scale parameter (sigma) of the noise.

        :param sigma: standard deviation
        :param phi: auto-correlation
        :return: sigma_e
        """
        return np.sqrt(sigma * (1. - phi ** 2)) if all(i is not None for i in [sigma, phi]) else 1.

    def init_sigma(self, sigma=None, phi=None, sigma_e=None):
        """
        It initializes the sigma.

        :param sigma: standard deviation
        :param phi: auto-correlation
        :param sigma_e: standard deviation of the noise
        :return: sigma
        """
        if sigma is None:
            return sigma_e ** 2 / (1. + phi) if all(i is not None for i in [sigma_e, phi]) else 1.
        else:
            return sigma

    def init_mean(self, mu=None, phi=None, c=None):
        """
        It initializes the mean.

        :param mu: mean
        :param phi: auto-correlation
        :param c: offset
        :return: mean
        """
        if mu is None:
            return c / (1. - phi) if all(i is not None for i in [c, phi]) else 0.
        else:
            return mu

    def generate_ts(self, seed=None):
        """
        It generates a new series of data with the initialized parameters.

        :param seed: random seed value, use only if one needs to ensure reproducibility; default: None
        """
        # initialize random seed
        if seed is not None:
            np.random.seed(seed)
        # generate series
        index = pd.date_range(start='2020-01-01 00:00', periods=self.__len, freq='1h')
        data = self.generate_ar1()
        # convert the data to data-frame
        self.__ts = pd.DataFrame(data, index=index, columns=['value'])
        # set output resolution
        # self.__ts['value'] = np.rint(self.__ts['value'] / self.__res)
        # properties of the generated time series
        print(f"properties of the generated ts: mu={self.__ts['value'].mean()}, s={self.__ts['value'].std()}, "
              f"phi={self.__ts['value'].autocorr()}")
        return self.__ts

    def generate_ar1(self):
        """
        It generates an AR(1) data series with the specified parameters.

        :return: AR(1) data series

        Sources:
                Wilks D. S. (1995): Statistical Methods in Atmospheric Sciences.
                                    International Geophysics Series, USA, Academic Press.
                Mudelsee M. (2010): Climate Time Series Analysis Classical Statistical and Bootstrap Methods,
                                    42, Springer, doi: 10.1007/978-90-481-9482-7.
                Pandit, Sudhakar M.; Wu, Shien-Ming (1983). Time Series and System Analysis
                                     with Applications. John Wiley & Sons
                https://en.wikipedia.org/wiki/Autoregressive_model
        """
        # initialized properties
        print(f'initialized properties: mu0={self.__mu}, s0={self.__s}, phi0={self.__phi}')
        # make a signal
        signal = [np.random.normal(self.__mu, self.__s)]
        if self.__dist == 'normal':
            i = 1
            while i < self.__len:
                signal.append(self.__c + self.__phi * (signal[-1]) +
                              np.random.normal(0, self.__se))
                i += 1
            return np.array(signal)
        else:
            raise NotImplementedError

    def add_cve(self, df):
        """
        It adds four CVEs (with the length of t) to the data.
        """
        idx_s = [25, 65, 135, 175]
        for i, j in enumerate(idx_s):
            df.iloc[int(idx_s[i]):int(idx_s[i]+self.__t), 0] = self.__mu * i * self.__s
        self.__ts = df
        return df

    def scale_data(self, df, factor=None):

        """
        It scales the data.

        :param factor: scaling factor

        Note:
            I applied several different transformation.
        """
        # for factorized  transformation
        scaled_data = df * factor
        # for normalized and factorized  transformation
        # scaled_data = (self.__ts - self.__ts.min()) * factor / (self.__ts.max() - self.__ts.min())
        # for stand and factorized transformation
        # scaled_data = (self.__ts - self.__ts.mean()) * factor / self.__ts.std()
        # for norm [0,1] transformation
        # scaled_data = (self.__ts - self.__ts.min()) * 1 / (self.__ts.max() - self.__ts.min())
        # for stand transformation
        # scaled_data = (self.__ts - self.__ts.mean()) * 1 / self.__ts.std()
        # first difference
        # scaled_data = self.__ts.diff(periods=1)
        # scaled_data2 = np.rint(scaled_data / self.__res) * self.__res
        return scaled_data    # np.rint(scaled_data / self.__res) * self.__res

    def save(self, df=None):
        """
        It saves the data time series.

        :param df: data series to save
        """
        if df is not None:
            df.to_csv(self.__out_fn)
        else:
            self.__ts.to_csv(self.__out_fn)

    def run(self):
        """
        It runs the model (generate synthetic data).
        """
        self.generate_ts(seed=5)
        if self.__dist == "normal":
            kwargs = {'loc': self.__mean, 'scale': self.__sigma, 'phi': self.__phi}
        elif self.__dist == "gamma":
            kwargs = {'mean': self.__mean, 'var': self.__var, 'phi': self.__phi}
        self.__ts = SynthTS(**kwargs).generate_ar1(distribution=self.__dist, N=self.__len,
                                                   start='2020-01-01 00:00', freq='H', **kwargs)
        # self.__ts = self.generate_ts()
        #self.__ts.describe()
        return self.__ts


if __name__ == "__main__":
    LENGTH = 240
    RES_PATH = '../results'
    # ############## generate synthetic time series ################
    experiments = ['3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7']
    experiments = ['3.1']

    for exp in experiments:
        df = pd.DataFrame()
        if exp == '3.1':
            # 3.1 reference time series (baseline)
            obj = SyntheticData(mean=10, sigma=4, res=.01, phi=0.8, t=3, dist='normal', length=LENGTH,
                                out_file_name=f'{RES_PATH}/ar1_baseline_ts.csv')
            df = obj.run()
            print(df.mean(), df.std())
            # df = obj.add_cve(df)
            df.columns = ['ref']
            print(df.mean(), df.std())
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5., 5), facecolor='w')
            df.plot(ax=ax)
            plt.xlabel('datetime')
            plt.ylabel('value')
            plt.show()
            exit()
        elif exp == '3.2':
            # 3.2 sensitivity to t
            ts = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            for t in ts:
                obj = SyntheticData(mean=10, sigma=2, res=0.01, phi=0.8, t=t, dist='normal', length=LENGTH,
                                    out_file_name=f'{RES_PATH}/ar1_baseline_ts_t_sens.csv')
                df_tmp = obj.run()
                df_tmp = obj.add_cve(df_tmp)
                df = pd.concat([df, df_tmp], axis=1)
            df.columns = ['t=' + str(t) for t in ts]
        elif exp == '3.3':
            # 3.3 sensitivity to sigma
            sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20]
            for s in sigmas:
                obj = SyntheticData(mean=5, sigma=s, res=0.01, phi=0.8, t=3, dist='normal', length=LENGTH,
                                    out_file_name=f'{RES_PATH}/ar1_baseline_ts_sigma_sens.csv')
                df_tmp = obj.run()
                df_tmp = obj.add_cve(df_tmp)
                df = pd.concat([df, df_tmp], axis=1)
            df.columns = ['std=' + str(s) for s in sigmas]
        elif exp == '3.4':
            # 3.4 sensitivity to (auto)corr or phi
            # phi = 0 - 0.99  --> note that phi=1 is meaningless and infinite
            corrs = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.90, 0.91, 0.92, 0.93,
                     0.94, 0.95, 0.99]
            for c in corrs:
                obj = SyntheticData(mean=10, sigma=4., res=0.01, phi=c, t=3, dist='normal', length=LENGTH,
                                    out_file_name=f'{RES_PATH}/ar1_baseline_ts_phi_sens.csv')
                df_tmp = obj.run()
                df_tmp = obj.add_cve(df_tmp)
                df = pd.concat([df, df_tmp], axis=1)
            df.columns = ['corr=' + str(c) for c in corrs]
        elif exp == '3.5':
            # 3.5 sensitivity to res
            reses = [0.01]  # [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
            obj = SyntheticData(mean=10, sigma=4, phi=0.8, t=3, dist='normal', length=LENGTH,
                                out_file_name=f'{RES_PATH}/ar1_baseline_ts_res_sens.csv')
            df_base = obj.run()
            for r in reses:
                print(f"Running exp 3.5 for res = {r:.4f}")
                # df_tmp = obj.add_cve(df_tmp)
                df_tmp = np.rint(df_base / r) * r
                df = pd.concat([df, df_tmp], axis=1)
            df.columns = ['res=' + str(r) for r in reses]
        elif exp == '3.6':
            # 3.6 sensitivity to scaling
            factors = [0.1, 1, 2, 5, 10]
            # factors = [1]
            for f in factors:
                obj = SyntheticData(mean=10, sigma=4, res=0.01, phi=0.8, t=3, dist='normal', length=LENGTH,
                                    out_file_name=f'{RES_PATH}/ar1_baseline_ts_scaling_sens_fc.csv')
                df_tmp = obj.run()
                # df_tmp = obj.add_cve(df_tmp)
                # df_tmp = obj.scale_data(df_tmp, factor=f)
                df = pd.concat([df, df_tmp], axis=1)
            df.columns = ['fact=' + str(f) for f in factors]
        elif exp == '3.7':
            # 3.7 sensitivity to data distribution
            dists = ['normal', 'lognormal', 'gamma']
            # factors = [1]
            for d in dists:
                obj = SyntheticData(mean=10, sigma=4, res=0.01, phi=0.8, t=3, dist=d, length=LENGTH,
                                    out_file_name=f'{RES_PATH}/ar1_baseline_ts_scaling_sens_dist.csv')
                df_tmp = obj.run()
                # df_tmp = obj.add_cve(df_tmp)
                # df_tmp = obj.scale_data(factor=f)
                df = pd.concat([df, df_tmp], axis=1)
            df.columns = ['dist=' + d for d in dists]
        obj.save(df=df)
        # exit()
    # #################### run CVT #####################
    for f in ['ar1_baseline_ts_t_sens.csv', 'ar1_baseline_ts.csv', 'ar1_baseline_ts_t_sens.csv',
              'ar1_baseline_ts_sigma_sens.csv', 'ar1_baseline_ts_phi_sens.csv',
              'ar1_baseline_ts_res_sens.csv', 'ar1_baseline_ts_scaling_sens.csv',
              'ar1_baseline_ts_dist_sens.csv']:
        # f = f'{RES_PATH}/ar1_baseline_ts_res_sens.csv'
        f = f'{RES_PATH}/ar1_baseline_ts_t_sens.csv'
        df = pd.read_csv(f, header=0, index_col=0, parse_dates=True)
        # print("DF after creation: ", df)
        col2 = ['data values', 'number of CVs', 'probability']
        df_out = pd.DataFrame()
        for col in df.columns:
            # two next line is only for the sensitivity to the scaling factors
            # res = float(col.split(sep='=')[1])
            # obtain params for CVT
            mu = df[col].mean()
            sigma = df[col].std()
            corr, _ = pacf(df[col], print_results=False)
            corr = corr[1]    # only use first partial auto-correlation coefficient
            res = col.split('=')[1]
            print(f"Dataset summary: N={df[col].count()}, mu={mu:.3f}, sigma={sigma:.3f}, "
                  f"corr={corr:.3f}, res={res}")
            # count = CVT(df[col], mu=mu, sigma=sigma, corr=corr, res=res).detect_CVs()
            mu, sigma, res, corr = 10, 4, 0.01,0.8
            out_tmp = CVT(df[col], mu=mu, sigma=sigma, corr=corr, res=res).run()
            # print(count, count/df[col].count())
            out_tmp.columns = pd.MultiIndex.from_product([[col], out_tmp.columns])
            df_out = pd.concat([df_out, out_tmp], axis=1)
            print(df_out)
        df_out.to_csv(f.split(sep='.')[0]+'ar1_baseline_ts_t_sens_cvt_results.csv')
        exit()


