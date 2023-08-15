import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator

# set up seaborn
sns.set_theme(style="whitegrid", palette='colorblind')
sns.set_context("paper", font_scale=1.5, rc={"lines.markersize": 10})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iters', action='store_true',
                        help='if provided, plot iterations instead of time')
    parser.add_argument('data', nargs='+',
                        help='path to csv file with data to plot')
    parser.add_argument('-o', help='If provided, save to file')    
    args = parser.parse_args()

    # load the data from the tensorboard event files
    sequences = {}
    dfs = []
    for subfolder in args.data:
        subfolder = Path(subfolder)
        # find the newest file in the sub-directory
        scalar_files = list(subfolder.iterdir())
        scalar_files.sort(key=lambda x: x.stat().st_mtime_ns)
        if scalar_files:
            ea = EventAccumulator(scalar_files[-1].as_posix())
            ea.Reload()
            # load data from the event accumulator
            w_times, step_nums, vals = zip(*ea.Scalars('mean cost'))
            # convert the wall times to seconds since start
            w_times = np.array(w_times)
            w_times -= w_times[0]
            # set the data type of the time to int so seaborn can group it
            w_times = w_times.astype(int)
            # round all times to the nearest 10 seconds
            w_times = np.round(w_times, -1)
            df = pd.DataFrame(
                {'Running time (s)': w_times, 'step_num': step_nums, 
                 'Cost': vals})
            method = subfolder.stem
            if method.startswith('bco_neuralinit'):
                df['Method'] = 'BCO (LP init)'
            elif 'bcolong' in method:
                df['Method'] = 'BCO (long)'
            elif method.startswith('bco'):
                df['Method'] = 'BCO'
            elif method.startswith('neural_bco'):
                df['Method'] = 'Neural BCO'
            elif method.startswith('neural'):
                df['Method'] = 'Learned Planner'
            sequences[subfolder] = df
            dfs.append(df)
    
    # concatenate the dataframes
    unified_df = pd.concat(dfs)

    x_col = 'step_num' if args.iters else 'Running time (s)'
    sns.lineplot(data=unified_df, x=x_col, y='Cost', hue='Method')
        
    # plt.yscale('log')
    plt.tight_layout()
    if args.o:
        plt.savefig(args.o)
    else:
        plt.show()


if __name__ == "__main__":
    main()