# !user/bin/env python

"""
This program is written to have a harmonized plots of the CVT output for data time series.
"""

# imports Python packages for data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.dates as mdates
mpl.use('tkagg')

font = {'family': 'serif', 'weight': 'medium', 'size': 12}
mpl.rc('font', **font)

__authors__ = "Najmeh Kaffashzadeh"
__email__ = "n.kaffashzadeh@fz-juelich.de"


class MakePlot:
    """
    A class to plot the data for demonstration purpose and documentation.
    """

    name = 'make plot'

    def __init__(self):
        """
        It initializes the input variable(s) for the MakePlot class.
        """
        pass

    def __repr__(self):
        """
        It returns representation of the object.
        """
        return {self.__class__.__name__, self.name}

    def truncate_cmap(self, cmap, minval=0.0, maxval=1.0, n=256):
        """
        It customizes the color map (cmap).

        Note:
            It is obtained from:
                           http://stackoverflow.com/questions/18926031/how-to-extract-a-subset-of-a-col
                           ormap-as-a-new-colormap-in-matplotlib
        """
        return mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval,
                                                         b=maxval), cmap(np.linspace(minval, maxval, n)))

    def scatter_plot(self, ax_obj=None, x=None, y=None, c=None):
        """
        It plots (scatter) the data.

        Args:
            ax_obj(obj): axes object
            x(int): the values of the x axis
            y(float): the values of the y axis
            c(float): the value for marker filling
        """
        if c is None:
            return ax_obj.scatter(x=x, y=y,
                                  s=10, marker='o', color='g')
        else:
            return ax_obj.scatter(x=x, y=y,
                                  c=c, s=20, marker='o',
                                  edgecolor='k', linewidths=0.1,
                                  zorder=2,
                                  cmap=self.truncate_cmap(cm.rainbow_r, 0.2,
                                                          0.8, 256),
                                  vmin=0, vmax=1)

    def line_plot(self, ax_obj=None, df=None):
        """
        It plots the data.

        Args:
            ax_obj(obj): axes object
            df(pd.dataframe): the values
        """
        return df.plot(ax=ax_obj, kind='line', color='k', marker='o',
                       markersize=4, markerfacecolor='k', alpha=0.5)

    @staticmethod
    def add_colorbar(image=None, fig=None):
        """
        It adds a color-bar (for probability) to the figure.

        Args:
             image(obj): the image
             fig(obj): the figure
        """
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.82, 0.12, 0.02, 0.73],
                               ('a', 'b'))  # [left, bottom, width, height]
        return fig.colorbar(image, cax=cbar_ax, label="quality score")

    @staticmethod
    def make_xticks_label(ax=None, x=None):
        """
        It makes the x ticks labels.

        Args:
            ax(object): axis
            x(datetime): x-axis values
        """
        if len(x.year.unique()) < 2:
            ax.xaxis.set_minor_locator(
                mdates.MonthLocator(interval=1, bymonthday=1))
            ax.xaxis.set_minor_formatter(mdates.DateFormatter('%b'.rjust(1)))
            # Tick every year on 1st day of the 1st month
            ax.xaxis.set_major_locator(
                mdates.YearLocator(2, month=x.month[0], day=x.day[15]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y'))
            ax.xaxis.set_tick_params(rotation=0)
        else:
            # Tick every 2 years on January 1st
            ax.xaxis.set_major_locator(mdates.YearLocator(2, month=1, day=1))

    @staticmethod
    def add_fig_title(fig=None, title=None):
        """
        It adds a title to the figure.

        Args:
            fig(obj): figure
            title(str): the figure title
        """
        return fig.suptitle(title, size=18)

    @staticmethod
    def make_subplots(n_rows=None, n_cols=None):
        """
        It makes subplots.

        Args:
            n_rows(int) : the number subplot rows
            n_cols(int): the number of subplot columns
        """
        if n_cols > n_rows:
            fig = plt.figure(figsize=(16, 4), facecolor='w')
            grid = plt.GridSpec(n_rows, n_cols, wspace=0.25, hspace=0)
        else:
            fig = plt.figure(figsize=(8, 5), facecolor='w')  # (8, 8)
            grid = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.4)
        return fig, grid

    def bar_plot(self, df=None, file_name=None):
        """
        It plots the sum of cve occurrence in the days.

        Args:
            df(pd.DataFrame): the input data for plotting
            file_name(str): output file name
        """
        if idx1 and idx2 is not None:
            df = df.loc[idx1:idx2]
        else:
            pass
        fig, grid = self.make_subplots(n_rows=1, n_cols=1)
        ax1 = plt.subplot(grid[0])
        df1 = df['number of CVs'].where(df['number of CVs'] == 2).dropna()
        df2 = df1.groupby(df1.index.hour).count()
        df2['number of CVs'].plot(ax=ax1, kind='bar', color='blue')
        plt.xlabel('hours of the day (utc)')
        plt.ylabel('$\Sigma$ t')
        plt.savefig(file_name+'.png',  bbox_inches='tight')
        plt.close()

    def run(self, df=None, file_name=None, idx1=None, idx2=None):
        """
        It plots the data for a given station.

        Args:
            df(pd.DataFrame): the input data for plotting
            file_name(str): output file name
            idx1(str): start datetime interval
            idx2(str): end datetime interval
        """
        # select a subset of the data time series
        if idx1 and idx2 is not None:
            df = df.loc[idx1:idx2]
        else:
            pass
        df2 = df[df['probability'] < 0.999999]
        print(df2['probability'].min())
        print(df2['probability'].max())
        fig, grid = self.make_subplots(n_rows=1, n_cols=1)
        ax1 = plt.subplot(grid[0])
        self.line_plot(ax_obj=ax1, df=df['data values'])
        ax1.set_xlabel('datetime', size=12)
        ax1.set_ylabel('values (nmol mol$^{-1}$)', size=12)
        ax1.set_ylabel('values ($\degree$C)', size=12)
        ax1.set_ylabel('values (ppm)', size=12)
        ax2 = ax1.twinx()
        df['probability'].plot(ax=ax2, color='blue', zorder=1, logy=True)
        ax2.set_ylabel('P', size=12)
        # plt.ylim((10**-10, 10**0))
        plt.savefig(file_name.split('.')[0] + '.png', bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    file_names = [ ('cvt_CO_Fresno_202201.csv', '2022-01-01 08:00:00', '2022-01-11 07:00:00'),
                  ('cvt_CO_Fresno_198001.csv', '1980-01-01 08:00:00', '1980-01-11 07:00:00'),
                  ('cvt_temp_CapeGrim_198301.csv', '1983-01-10 12:00:00', '1983-01-20 12:00:00'),
                  ('cvt_ozone_Azusa_201111.csv', '2011-11-10 00:00:00', '2011-11-20 12:00:00'),
                  ('cvt_ozone_Azusa_199011.csv', '1990-11-10 00:00:00', '1990-11-20 12:00:00')]
    for fn, idx1, idx2 in file_names:
        df = pd.read_csv(fn, header=0, index_col=0, parse_dates=True)
        MakePlot().run(df=df[['data values', 'probability']], file_name=fn, idx1=idx1, idx2=idx2)

