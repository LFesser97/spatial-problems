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


# set up seaborn
sns.set_theme(style="whitegrid", palette='colorblind')


def barplot(df, baseline_method, statistic):
    # divide non-baselines by the baseline, multiply by 100, subtract 100
    unique_num_nodes = df.index.get_level_values('# nodes').unique()
    fig, axes = plt.subplots(len(unique_num_nodes), 1)
    is_first = True
    for num_nodes, ax in zip(unique_num_nodes, axes):
        grp = df.loc[(slice(None), num_nodes), (slice(None), statistic)]
        np.seterr(invalid='raise')
        baseline = grp[(baseline_method, statistic)]
        grp = grp.drop(columns=(baseline_method, statistic))

        for cc in grp:
            ratio = grp[cc] / baseline
            percent_change = ratio * 100 - 100
            grp[cc] = percent_change


        grp = grp.stack('Method').reset_index(['# routes', 'Method'])

        sns.barplot(x='# routes', y=statistic, hue='Method', data=grp,
                    errorbar=None, ax=ax)
        ax.set_xlabel('')
        ax.set_ylabel(f'$n$ = {num_nodes}')
        if not is_first:
            ax.legend([], [], frameon=False)

        is_first = False
    # set x label to "$S$"
    plt.xlabel("# of routes $S$")
    # set y label to "percentage"
    fig.supylabel(f"% cost difference from {baseline_method}")

    plt.tight_layout()

    # stack multiple barplots by $n$?


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str,
                        help='path to csv file with data to plot')
    parser.add_argument('-s', '--save', type=str,
                        help='path at which to save the plot')
    parser.add_argument("baseline", help="The method to use as the baseline")
    parser.add_argument("statistic", help="The statistic to show in the table")
    
    args = parser.parse_args()

    # read data
    df = pd.read_csv(args.data, header=[0,1], index_col=[0,1,2])
    # grp = method_as_cols(df, args.statistic)

    barplot(df, args.baseline, args.statistic)

    if args.save:
        plt.savefig(args.save)
    else:        
        plt.show()

if __name__ == "__main__":
    main()