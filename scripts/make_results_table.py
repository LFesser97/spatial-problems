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

import pandas as pd


def method_in_index(df):
    grp = df.groupby('# routes').mean()
    stacked = grp.stack([0,1]).reset_index('Method')
    # stacked['Method'] = stacked['Method'] + ', ' + stacked['Mode']
    stacked['Method'] = \
        stacked['Method'].str.replace('BCO, _', 'Bee Colony Optimization')
    # stacked = stacked.drop(columns='Mode')

    return stacked


def method_as_cols(df, value_key):
    grp = df.groupby(['# nodes', '# routes']).mean()
    grp = grp.loc[:, (slice(None), value_key)]
    grp = grp.droplevel('Statistic', 'columns')
    return grp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("statistic", help="The statistic to show in the table")
    # parser.add_argument("-o", "--output", 
    #                     help="file in which to save the latex table")
    args = parser.parse_args()

    # sns.set_style("whitegrid")
    # sns.color_palette("colorblind")
    # sns.set_context("paper")

    df = pd.read_csv(args.csv, header=[0,1], index_col=[0,1,2])

    table = method_in_index(df)
    table = method_as_cols(df, args.statistic)
    latex = table.to_latex(float_format="%.3g", escape=False, 
                           multicolumn=False)
    # if args.output:
    #     with open(args.output, 'w') as f:
    #         f.write(latex)
    # else:
    print(latex)


if __name__ == "__main__":
    main()