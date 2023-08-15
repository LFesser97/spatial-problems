import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from make_main_table import parse_csvs

# set up seaborn
sns.set_theme(style="whitegrid", palette='colorblind')
sns.set_context("paper", font_scale=1.5, rc={"lines.markersize": 10})
plt.rcParams['figure.constrained_layout.use'] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to plot')
    # parser.add_argument('-e', '--env',  
    #                     help='If provided, plot only this environment')
    parser.add_argument('-o', help='If provided, save to file')
    args = parser.parse_args()

    # load the csv files into dataframes
    unified_df = parse_csvs(args.data)

    # use seaborn to plot city size vs costs with error bars over seeds
    sns.pointplot(x='Environment', y='Cost $C$', hue='Method', 
                  data=unified_df, 
                #   capsize=.1, 
                scale=1.5,
                  markers=
                  ['o', 'x', '*', 's', 'd', 'v', '^', '>', '<', 'p', 'h'],
                  linestyles='dashed')
    
    # make xtick labels diagonal
    plt.xticks(rotation=45, ha='right')
    
    # plt.yscale('log')
    if args.o:
        plt.savefig(args.o)
    else:
        plt.show()


if __name__ == "__main__":
    main()