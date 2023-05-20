# !user/bin/env python

"""
This program is written to run the CVTs on a input data time series.
here the real-world data (in data directory)
"""

# imports python standard libraries
import os
import sys

# imports Python packages for data science
import pandas as pd
import numpy as np

# import local modules
from cvt import CVT

__authors__ = "Najmeh Kaffashzadeh"
__email__ = "n.kaffashzadeh@fz-juelich.de"


class CVTRun:
    """
    A class to run the CVT on real-time data time series.
    """

    name = 'CVT run'

    def __init__(self):
        """
        It initializes the inputs for the CVTRun class.
        """
        self.__output = None
        self.__ts = None

    def __repr__(self):
        """
        It returns representation of the object.
        """
        return {self.__class__.__name__, self.name}

    def read_from_file(self, filename=None):
        """
        To read the input data (i.e. time series) as a pd.series.

        :param filename: the input file (path and name)
        """
        try:
            self.__ts = pd.read_csv(filename, header=0, index_col=0, parse_dates=True)
            if (self.__ts.columns[0] != 'value') & (len(self.__ts.columns) == 1):
                self.__ts.columns = ['value']
            new_in = pd.date_range(start=self.__ts.index[0], end=self.__ts.index[-1], freq='M')
            self.__ds = self.__ts.reindex(new_in, fill_value=np.nan)
            self.__ts=self.__ts[0:240]
        except FileNotFoundError:
            raise FileNotFoundError

    def run(self):
        """
        It runs the CVTs.
        """
        print('mu = ', self.__ts['value'].mean())
        print('std = ', self.__ts['value'].std())
        print('phi = ', self.__ts['value'].autocorr(lag=1))
        self.__output = CVT(ts=self.__ts['value'], sigma=self.__ts['value'].std()).run()

    def save_output(self, filename=None):
        """
        It saves the output.

        :param filename: output file name
        """
        pathname = os.path.dirname(os.path.realpath(__file__))
        try:
            self.__output.droplevel(1, axis=1).to_csv(pathname + '/cvt_' + filename)
        except IndexError:
            self.__output.to_csv(pathname + '/cvt_' + filename)
        except FileExistsError:
            raise FileExistsError
        except AttributeError:
            pass


if __name__ == "__main__":
    # df = df[(df['State Name']=='California') & (df['County Name']=='Fresno')]
    # df['date'] = pd.to_datetime(df['Date GMT'] + ' ' + df['Time GMT'])
    # df_new = df.set_index(df['date'])
    # to read a given file in a given path
    pathname = sys.path[0] + '/../data'
    filenames = ['CO_Fresno_198001.csv']
                # 'temp_CapeGrim_198301.csv', 'ozone_Azusa_199011.csv',
                # 'ozone_Azusa_201111.csv', 'CO_Fresno_202201.csv']
    cve_count = {}
    # make object
    obj = CVTRun()
    for fn in filenames:
        # to separate the file name from the path
        fn = fn.rsplit('/')[-1]
        if pathname is not None:
            obj.read_from_file(filename=pathname+'/'+fn)
        else:
            obj.read_from_file(filename=fn)
        obj.run()
        obj.save_output(filename=fn)
