import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


HEADER_NAMES = ['$\\alpha$', '# routes', 'cost', 'ATT', 'RTT', '$d_0$', 
                '$d_1$', '$d_2$', '$d_{un}$', 'n_uncovered', 'n_stop_oobs', 
                'Time', '# iterations']


def assign_method(df, filename):
    # add a column to each dataframe with the method name
    if filename.startswith('bco'):
        df['Method'] = 'BCO'
    elif 'at1b' in filename:
        df['Method'] = '$\Delta$-NBCO'
    elif filename.startswith('neural_bco_short'):
        df['Method'] = 'Neural BCO (short)'
    elif filename.startswith('neural_bco') or filename.startswith('nbco'):
        df['Method'] = 'Neural BCO'
    elif '4k' in filename:
        df['Method'] = 'LP-4k'
    elif '40k' in filename:
        df['Method'] = 'LP-40k'
    elif filename.startswith('neural'):
        df['Method'] = 'LP-100'


def add_env(df):
    df['Environment'] = 'Mandl'
    df.loc[df['# routes'] == 6, 'Environment'] = 'Mandl'
    df.loc[df['# routes'] == 12, 'Environment'] = 'Mumford0'
    df.loc[df['# routes'] == 15, 'Environment'] = 'Mumford1'
    df.loc[df['# routes'] == 56, 'Environment'] = 'Mumford2'
    df.loc[df['# routes'] == 60, 'Environment'] = 'Mumford3'


def parse_csvs(csv_paths):
    # load the csv files into dataframes
    dfs = []
    for path in csv_paths:
        df = pd.read_csv(path)
        if 'cost' not in df.columns:
            df = pd.read_csv(path, header=0, names=HEADER_NAMES)

        filename = Path(path).stem
        assign_method(df, filename)
        add_env(df)

        if 'pp' in filename:
            df['Setting'] = 'PP'
            df['$\\alpha$'] = 1.0
        elif 'op' in filename:
            df['Setting'] = 'OP'
            df['$\\alpha$'] = 0.0
        elif 'balanced' in filename:
            df['Setting'] = 'Balanced'
            df['$\\alpha$'] = 0.5

        dfs.append(df)

    # concatenate the dataframes
    unified_df = pd.concat(dfs)
    # discard the Mandl instances with # routes != 6
    unified_df = unified_df.loc[(unified_df['Environment'] != 'Mandl') |
                                (unified_df['# routes'] == 6)]

    # unified_df.rename(columns={'ATT': '$C_p$', 'RTT': '$C_o$', 
    #                            'cost': 'Cost $C$'}, 
                    #   inplace=True)

    return unified_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', nargs='+', 
                        help='path to csv file with data to format')
    parser.add_argument('-v', '--value', default='cost',
                        help='plot this value')
    args = parser.parse_args()

    # load the csv files into dataframes
    unified_df = parse_csvs(args.data)

    grp_columns = ['Environment', 'Method']
    pivot_index = ['Method']
    if unified_df['Setting'].nunique() > 1:
        grp_columns.insert(0, '$\\alpha$')
        pivot_index.insert(0, '$\\alpha$')
    grp = unified_df.groupby(grp_columns)
    # pivot the table so each Environment is a column
    mean = grp.mean().pivot_table(index=pivot_index, columns=['Environment'], 
                                  values=args.value)
    std = grp.std().pivot_table(index=pivot_index, columns=['Environment'], 
                                values=args.value)
    std = std * 100 / mean
    mean = pd.DataFrame([row.apply(lambda xx: '{:.3f}'.format(xx))
                         for _, row in mean.iterrows()])
    std = pd.DataFrame([row.apply(lambda xx: '{:.0f}\%'.format(xx))
                        for _, row in std.iterrows()])
    both = mean + ' $\pm$ ' + std
    print(both.to_latex(float_format='%.3f', escape=False))

    cvs = (unified_df['n_uncovered'] > 0) | (unified_df['n_stops_oob'] > 0)
    unified_df['constraints violated'] = cvs.astype(bool)
    
    grp = unified_df.groupby(['Method', 'Environment', '$\\alpha$']).sum()
    pvt = grp.pivot_table(index=pivot_index, columns=['Environment'], 
                          values='constraints violated')
    print(pvt)


if __name__ == "__main__":
    main()