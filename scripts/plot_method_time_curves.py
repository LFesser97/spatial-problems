# Copyright 2023 Andrew Holliday
# 
# This file is part of the Transit Learning project.
#
# Transit Learning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
# 
# Transit Learning is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# Transit Learning. If not, see <https://www.gnu.org/licenses/>.

import argparse
from pathlib import Path

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from make_results_table import method_as_cols

# set up seaborn
sns.set_theme(style="whitegrid", palette='colorblind')


def lineplot(df, statistic):
    # divide non-baselines by the baseline, multiply by 100, subtract 100
    unique_num_nodes = df.index.get_level_values('# nodes').unique()
    # stack multiple barplots by $n$
    grp = df.loc[:, (slice(None), statistic)]
    grp = grp.stack('Method')
    grp.index.rename('# nodes $n$', level='# nodes', inplace=True)
    # grp['# nodes $n$'] = grp['# nodes']
    ax = sns.lineplot(x='# routes', y=statistic, hue='Method', 
                      style='# nodes $n$', data=grp, errorbar=None, 
                      markers=True,
                      hue_order=["Bee Colony Optimization", "Ours, greedy",
                                 "Ours, sampling 20", "Hyperheuristic"])
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1,1))

    # set x label to "$S$"
    plt.yscale('log')
    plt.xlabel("# of routes $S$")
    # set y label to "percentage"
    plt.ylabel(f"Wall-clock compute time (s)")

    plt.tight_layout()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str,
                        help='path to csv file with data to plot')
    parser.add_argument('-s', '--save', type=str,
                        help='path at which to save the plot')
    # parser.add_argument("statistic", help="The statistic to show in the table")
    
    args = parser.parse_args()

    # read data
    df = pd.read_csv(args.data, header=[0,1], index_col=[0,1,2])
    # grp = method_as_cols(df, args.statistic)

    lineplot(df, 'time (s)')

    if args.save:
        plt.savefig(args.save)
    else:        
        plt.show()

if __name__ == "__main__":
    main()