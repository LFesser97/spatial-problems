import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from make_main_table import assign_method, add_env, HEADER_NAMES

# set up seaborn
sns.set_theme(style="whitegrid", palette='colorblind')
sns.set_context("paper", font_scale=2, rc={"lines.markersize": 15})
plt.rcParams['figure.constrained_layout.use'] = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to plot')
    parser.add_argument('-e', '--env',  
                        help='If provided, plot only this environment')
    parser.add_argument('-b', '--errorbars', action='store_true',
                        help='If provided, plot error bars')
    parser.add_argument('-o', help='If provided, save to file')
    args = parser.parse_args()

    if args.errorbars and not args.env:
        raise ValueError('error bars only supported for single environment')

    dfs = []
    for path in args.data:
        # load the csv files, making sure they have the correct column names
        df = pd.read_csv(path)
        if 'cost' not in df.columns:
            df = pd.read_csv(path, header=0, names=HEADER_NAMES)
        if df.columns[0] != '$\\alpha$':
            # set the first column's name to be alpha
            df.rename(columns={df.columns[0]: '$\\alpha$'}, inplace=True)

        filename = Path(path).stem
        last_part = filename.rpartition('_')[-1]
        try:
            df['$\\alpha$'] = float(last_part)
        except ValueError:
            pass
            
        if Path(path).stem.partition('_')[0] == 'neural_bco':
            print(Path(path).stem)

        assign_method(df, filename)
        add_env(df)
        dfs.append(df)

    # concatenate the dataframes
    unified_df = pd.concat(dfs)
    if args.env:
        unified_df = unified_df.loc[unified_df['Environment'] == args.env]
        style = 'Method'
        # print only the ATT and RTT columns of the dataframe
        print(unified_df[['$\\alpha$', 'ATT', 'RTT']])
    else:
        style='Environment'

    unified_df.rename(columns={'ATT': '$C_p$ (minutes)', 
                               'RTT': '$C_o$ (minutes)'}, inplace=True)

    grp = unified_df.groupby(['Method', 'Environment', '$\\alpha$'])
    mean = grp.mean()
    std = grp.std()

    # uncomment the below to get the alternative style
    # styles = ['solid', 'dashed', 'dotted', 'dashdot']
    if args.errorbars:
        for (_, md), (_, sd) in zip(mean.groupby('Method'), 
                                    std.groupby('Method')):
            plt.errorbar(md['$C_p$ (minutes)'], md['$C_o$ (minutes)'],
                         sd['$C_o$ (minutes)'], sd['$C_p$ (minutes)'],
                         fmt='o')

    # also comment out this line if you want the alternative style
    g1 = sns.lineplot(x='$C_p$ (minutes)', y='$C_o$ (minutes)', hue='Method', 
                      style=style, markers=True, data=mean)

    g1.set(xlabel=None, ylabel=None)

    if args.env:
        plt.legend(frameon=False)
    else:
        plt.yscale('log')
        plt.legend(frameon=False, 
                   fontsize=13
                   )

    if args.o:
        plt.savefig(args.o)
    else:
        plt.show()


if __name__ == "__main__":
    main()