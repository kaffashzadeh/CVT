# !user/bin/env python

"""
This program is written to have plot the sensitivity results of the synthetic data.
"""

# imports Python packages for data science
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.dates as mdates
import sys

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
                                                                                             b=maxval),
                                                         cmap(np.linspace(minval, maxval, n)))

    @staticmethod
    def make_colormap1(my_seq=None):

        """
        It returns a LinearSegmentedColormap.

        Args:
            my_seq(list ): floats number and RGB-tuples. The floats should be increasing
                         and in the interval (0,1).

        Reference:
                 https://stackoverflow.com/questions/16834861/
                 create-own-colormap-using-matplotlib-and-plot-color-scale
        """
        seq = [(None,) * 3, 0.0] + list(my_seq) + [1.0, (None,) * 3]
        cdict = {'red': [], 'green': [], 'blue': []}
        for i, item in enumerate(seq):
            if isinstance(item, float):
                r1, g1, b1 = seq[i - 1]
                r2, g2, b2 = seq[i + 1]
                cdict['red'].append([item, r1, r2])
                cdict['green'].append([item, g1, g2])
                cdict['blue'].append([item, b1, b2])
        return mcolors.LinearSegmentedColormap('CustomMap', cdict)

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

    @staticmethod
    def line_plot(ax_obj=None, df=None, c='k', style='-',
                  alpha=0.5, zorder=1, log=False):
        """
        It plots the data.

        Args:
            ax_obj(obj): axes object
            df(pd.dataframe): the values
            c(str): colour; default black
            style(str): line style; default is line '-'.
        """
        if log:
            return df.plot(ax=ax_obj, kind='line', color=c, marker='o', logy=True,
                           markersize=1, markerfacecolor=c, style=style,
                           alpha=alpha, zorder=zorder, xticks=[])
        else:
            return df.plot(ax=ax_obj, kind='line', color=c, marker='o',
                           markersize=1, markerfacecolor=c, style=style,
                           alpha=alpha, zorder=zorder, xticks=[])

    @staticmethod
    def bar_chart(ax_obj=None, x=None, values=None, width=None, c='k',
                  alpha=0.5, zorder=1, log=False):
        """
        It plots bar chart.

        Args:
            ax_obj(obj): axes object
            x(float): the values
            width(int): width of bar
            c(str): colour; default black
        """
        if log:
            return ax_obj.bar(x, values, width, log=True, color=c, alpha=alpha, zorder=zorder)
        else:
            return ax_obj.bar(x, values, width, color=c, alpha=alpha, zorder=zorder)

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
                mdates.YearLocator(1, month=x.month[0], day=x.day[15]))
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
            fig = plt.figure(figsize=(15, 8), facecolor='w')
            grid = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.8)
        else:
            fig = plt.figure(figsize=(5, 5), facecolor='w')
            grid = plt.GridSpec(n_rows, n_cols, wspace=0.4, hspace=0.4)
        return fig, grid

    def run(self, df1=None, df2=None, cols=None, which_pertrub=None, which_trans=None):
        """
        It plots the data for a given station.

        Args:
            df1(pd.DataFrame): the input for data values plot
            df2(pd.DataFrame): the input for probability plot
            cols(list): the column names
            which_perturb(str): data perturbation
            which_trans(str): data transformation
        """
        if which_pertrub == 'mu':
            cols = [('mu=5', 'green'), ('ref', 'k'), ('mu=20', 'blue')]
            leg = ['$\mu$=5', 'ref ($\mu$=1)', '$\mu$=20']
            cols = [('ref', 'k')]
            leg = ['ref ($\mu$=10)']
            xlabel = '$\mu$'
        elif which_pertrub == 'sigma_sens_':
            x = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 20]
            cols = [('std=2', 'green'), ('std=4', 'k'), ('std=8', 'blue')]
            leg = ['$\sigma$=20', 'ref ($\sigma$=4)', '$\sigma$=8']
            xlabel = '$\sigma$'
        elif which_pertrub == 'phi_sens_':
            x = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93,
                 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
            cols = [('corr=0.5', 'green'), ('ref', 'k'), ('corr=0.9', 'blue')]
            leg = ['$\phi$=0.5', 'ref ($\phi$=0.8)', '$\phi$=0.9']
            xlabel = "$\phi$"
        # elif which_pertrub == 'phi_sens_':
        #     x = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
        #     cols = [('corr=0.5', 'green'), ('ref', 'k'), ('corr=0.9', 'blue')]
        #     leg = ['$\phi$=0.5', 'ref ($\phi$=0.8)', '$\phi$=0.9']
        #     xlabel = "$\phi$"
        elif which_pertrub == 'res_sens_':
            x = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5]
            cols = [('ref', 'k'), ('res=5', 'blue')]
            leg = ['ref (res=.01)', 'res=5']
            xlabel = "$res$"
        elif which_pertrub == 't_sens_':
            x = [2, 3, 4, 5, 6, 7, 8, 9, 10]
            # t_values in the sensitivity study
            leg = ['ref (res=.01)', 'res=5']
            lst = [i for i in df1.columns.get_level_values(0) if 't=' in i]
            cols = list(set(lst))
            xlabel = "t"
        elif (which_pertrub == 'scaling_sens_') | (which_pertrub == 'scaling_sens_norm_fc_') | \
             (which_pertrub == 'scaling_sens_stand_fc_') | (which_pertrub == 'scaling_sens_fc_') | \
             (which_pertrub == 'scaling_sens_stand_new_'):
            x = [0.1, 0.2, 5, 10]
            lst = [i for i in df1.columns.get_level_values(0) if 't=' in i]
            cols = list(set(lst))
            xlabel = "fc"
        else:
            cols = [('ref', 'k')]
            leg = ['ref']
            xlabel=" "
        labels = ['CVE1', 'CVE2', 'CVE3', 'CVE4']
        CVEs = [('2020-01-02 01:00:00', '2020-01-02 03:00:00', 'r'),
                ('2020-01-03 17:00:00', '2020-01-03 19:00:00', 'b'),
                ('2020-01-06 15:00:00', '2020-01-06 17:00:00', 'g'),
                ('2020-01-08 07:00:00', '2020-01-08 09:00:00', 'yellow')]

        # idx = pd.IndexSlice
        # to select the desired columns
        # df1_sel = df1.loc[:, idx[sel_col]]
        for which_val in ['probability', 'data values']:
            fig, grid = self.make_subplots(n_rows=1, n_cols=1)
            ax1 = plt.subplot(grid[0])
            if which_val == 'probability':
                # for i, col in enumerate(cols):  # col[0] is column_name and col[1] is colour
                for cve in CVEs:
                    # to show probability results as time series
                    # self.line_plot(ax_obj=ax1, df=df1[('fact=1', 'data values')],
                    #                c='k', style='-', alpha=1, zorder=2, log=False)
                    # ax2 = ax1.twinx()
                    # print(df1[('ref', 'probability')].min(), df1[('ref', 'probability')].max())
                    # self.line_plot(ax_obj=ax2, df=df1[('fact=1', 'probability')],
                    #               c='r', style='--', alpha=1, zorder=2, log=True)
                    # to show probability results as bar chart
                    # df_prob = df2[(col, which_val)]
                    idx = pd.IndexSlice  # to sparse index; output was multi-index data-frame
                    # # # vals = df1.loc[cve[0], idx[:, 'probability']].values
                    # # # vals = df1.loc[ix, idx[:, 'probability']].values
                    # # # CVEs[:][0] for the start date for selecting the CVEs
                    # # # CVEs[:][2] for color of the CVEs
                    for i in np.arange(0, 4):
                        print(f"min={np.min(df1.loc[CVEs[i][0], idx[:, 'probability']].values)}")
                        print(f"max={np.max(df1.loc[CVEs[i][0], idx[:, 'probability']].values)}")
                    print(df1.loc[CVEs[1][1], idx[:, 'probability']].values)
                    print(df1.loc[CVEs[2][1], idx[:, 'probability']].values)
                    print(df1.loc[CVEs[3][1], idx[:, 'probability']].values)
                    ax1.plot(x, df1.loc[CVEs[0][1], idx[:, 'probability']].values, CVEs[0][2],
                             x, df1.loc[CVEs[1][1], idx[:, 'probability']].values, CVEs[1][2],
                             x, df1.loc[CVEs[2][1], idx[:, 'probability']].values, CVEs[2][2],
                             x, df1.loc[CVEs[3][1], idx[:, 'probability']].values, CVEs[3][2])
                    ax1.set_yscale('log')
                    # ax1.set_xscale('log')
                    plt.show()
                    # print(df1.loc['2020-01-02 01:00:00', idx[:, 'probability']].values)
                    # mean_cve = [df_prob.xs('episode1').mean(), df_prob.xs('episode2').mean(),
                    #            df_prob.xs('episode3').mean(), df_prob.xs('episode4').mean()]
                    # self.bar_chart(ax_obj=ax1, x=bar_xs[0], values=mean_cve, width=[5, 5, 5, 5],
                    #               log=True, c=col[1], alpha=0.9, zorder=3)
                    # ax1.legend(leg, bbox_to_anchor=(0.79, 0.997),
                    #           loc='upper left', borderaxespad=0., prop={'size': 10})
                    # ax1.set_title(tit)
                # plt.xticks(bar_xs[0], labels)
                # ax1.set_ylabel(which_val)
                # ax1.set_ylabel('values')
                ax1.set_ylabel('P')
                ax1.set_xlabel(xlabel)
                # ax2.axes.get_xaxis().set_visible(False)
                # ax1.axes.get_xaxis().set_visible(False)
                # for t1, t2, c in CVEs:
                #    ax1.axvspan(t1, t2, facecolor=c, edgecolor='none', alpha=.2)
                # plt.grid()
                plt.savefig('ar1_baseline_ts_' + which_pertrub +
                            'cvt_results.png',
                            bbox_inches='tight')
                plt.close()
            elif which_val == 'data values':
                pass


if __name__ == "__main__":
    for which in ['', 'res_sens_', 't_sens_']:
        # for data_trans in ['', 'normalized_', 'scaled_', 'standardized_', 'stationary_']:
        which = 'res_sens_'
        which = 't_sens_'
        lst = []
        # file names
        fn1 = '/../results/ar1_baseline_ts_'+which+'cvt_results.csv'
        # fn2 = '../synthetic_data/sensitivity_test/synthetic_'+data_trans+'data_cvt_results_added_idx.csv'
        # to read the data
        df1 = pd.read_csv(sys.path[0]+fn1, header=[0,1], index_col=0, parse_dates=True)
        print(df1)
        # df2 = pd.read_csv(fn2, header=[0, 1], index_col=[0, 1], parse_dates=True)
        # to select those column with a given name (e.g. mu, res, etc.)
        # lst = [i for i in df1.columns.get_level_values(0) if 't=' in i]
        # lst.append('ref')
        # to delete duplicate
        # sel_col = list(set(lst))
        idx = pd.IndexSlice
        # to select the desired columns
        # df1_sel = df1.loc[:, idx[sel_col]]
        # df2_sel = df2.loc[:, idx[sel_col]]
        # to plot the data and prob results
        MakePlot().run(df1=df1, df2=df1, cols=None, which_trans=None, which_pertrub=which)
        exit()
        # print('plots for ' + data_trans + ' data were done!')
    print('so plots for the ' + which+' sensitivity were done!')
