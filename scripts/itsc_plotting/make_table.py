import argparse
from pathlib import Path
import pandas as pd


def reformat():
    """Format our CSV data properly, converting seconds to minutes and
       satisfied demands to percentages."""
    # open all csv files in current directory
    out_dir = Path('out/')
    if not out_dir.exists():
        out_dir.mkdir()
    for csv_path in Path('.').glob('*.csv'):
        df = pd.read_csv(csv_path)
        # divide times in seconds by 60 to get minutes
        df['ATT'] /= 60
        df['RTT'] /= 60
        # divide demand by total demand to get percentage
        total_demand = df['d_0'] + df['d_1'] + df['d_2'] + df['d_{un}']
        df['d_0'] /= total_demand
        df['d_1'] /= total_demand
        df['d_2'] /= total_demand
        df['d_{un}'] /= total_demand
        df.to_csv(out_dir / csv_path.name, index=False)


def combine(directory, exclude=[]):
    """Combine all the dataframes and print them as a latex table in the format 
        we want for the paper."""
    pwd = Path(directory)
    # open all csv files in current directory
    glob = list(pwd.glob('*.csv'))
    if exclude:
        glob = [pp for pp in glob if not any([pp.match(ex) for ex in exclude])]
    glob = sorted(glob, key=lambda x: x.stem)
    dfs = []
    for csv_path in glob:
        df = pd.read_csv(csv_path)
        df.drop(labels=['cost', 'n_uncovered', 'time', 'iters'], axis=1, 
                inplace=True, errors='ignore')
        # values in the table are for both directions, so divide by 2
        df['RTT'] /= 2
        dfs.append(df)

    # make environment, n_routes, and metrics into rows
    dfs = [df.set_index(['Environment', '# routes']).stack() for df in dfs]
    # concatenate all dataframes into one, with each column being one frame
    df = pd.concat(dfs, axis=1, keys=[ff.stem for ff in glob])
    # print the latex table
    print(df.to_latex(float_format='%.2f', escape=False, na_rep='-'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('-x', '--exclude', action='append')
    args = parser.parse_args()
    combine(args.dir, args.exclude)
