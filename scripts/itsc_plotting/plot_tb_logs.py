import argparse
import pickle
from pathlib import Path

from collections import defaultdict
from tqdm import tqdm
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
import matplotlib.pyplot as plt
from matplotlib import font_manager

city_names = ['Mumford0', 'Mumford1', 'Mumford2', 'Mumford3']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['figure.constrained_layout.use'] = True
plt.style.use('tableau-colorblind10')
# plt.rcParams['font.serif'] = ['Times New Roman']

def main(folder, glob, normalize, y_label, ylog=False, legend_loc='best'):
    # font = font_manager.FontProperties(family='Comic Sans MS', size=8)
    datas = defaultdict(dict)
    for filename in Path(folder).glob(glob):
        if filename.suffix != '.pkl':
            continue

        with filename.open('rb') as ff:
            data = pickle.load(ff)
        for city_name in city_names:
            if city_name.lower() in filename.stem.lower():
                datas[city_name][filename.stem] = data

    colour_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for city_name in city_names:
        city_data = datas[city_name]
        colour = colour_cycle.pop(0)
        if normalize:
            max_stepnum = max([dd['step_nums'][-1] for dd in city_data.values()])
            step_norm = 100 / max_stepnum
            max_val = max([dd['vals'].max() for dd in city_data.values()])
            val_norm = 1 / 60 # * max_val
        else:
            step_norm = 1
            # convert from seconds to minutes
            val_norm = 1 / 60

        if 'RTT' in y_label:
            # convert from time in both directions to time in one direction
            val_norm /= 2

        # xs = city_data['step_nums'] / city_data['mumford0']['step_nums'].max()
        for name, data in sorted(city_data.items()):
            xs = data['step_nums'] * step_norm
            ys = data['vals'] * val_norm

            label = city_name
            if 'myinit' in name:
                style = 'solid'
                label += ', NIHH'
                nihh_ys = ys
            else:
                style = 'dashed'
                label += ', PHH'
                phh_min = ys.min()
                plt.hlines(phh_min, xs.min(), xs.max(), 
                    color=colour, linestyle='dotted')

            plt.plot(xs, ys, label=label, linestyle=style, color=colour)
        
        marker_xloc = np.abs(nihh_ys - phh_min).argmin()
        plt.scatter(xs[marker_xloc], phh_min, color=colour, marker='x', s=100)

    y_label = y_label + ' (minutes)'
    if ylog:
        y_label += ', log scale'
        plt.yscale('log')

    plt.legend(loc=legend_loc, ncol=2, frameon=False)
    # if normalize:
        # plt.xlabel('% of total steps')
    # else:
    #     plt.xlabel('Number of steps')
    plt.ylabel(y_label)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=str, help='Folder with pickles')
    parser.add_argument('-g', '--glob', type=str, default='*')
    parser.add_argument('-s', '--size', type=int, default=12, help='Font size')
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-l', '--ylog', action='store_true', 
                        help='Use log scale on y axis')
    parser.add_argument('--ll', '--legendloc', default='best', 
                        help='Location of legend')
    parser.add_argument('-y', '--ylabel')
    args = parser.parse_args()
    plt.rcParams['font.size'] = args.size
    main(args.folder, args.glob, args.normalize, args.ylabel, args.ylog, 
         args.ll)
