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

import pickle
import h5py
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import sys
from sklearn.neighbors import KernelDensity


def density_plot(values, title, bandwidth_ratio=0.05):
    if type(values) is list:
        values = np.array(values)
    if values.ndim == 1:
        values = values[:, None]
    bandwidth = (values.max() - values.min()) * bandwidth_ratio
    estimator = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    est_fn = estimator.fit(values[:, None])
    plot_xs = np.linspace(values.min(), values.max(), 1000)
    plot_ys = np.exp(est_fn.score_samples(plot_xs[:, None]))
    # plt.hist(global_dset)
    plt.plot(plot_xs, plot_ys)
    plt.title(title)
    plt.show()


def plot_saved_value_stats():
    with open("saved_value_stats.pkl", "rb") as ff:
        val_dict = pickle.load(ff)
    
    global_vals = val_dict["global values"]
    route_vals = val_dict["route values"]
    
    for name, values in [("global saved time", global_vals),
                         ("satisfied demand", route_vals[:, 0]),
                         ("ridership", route_vals[:, 1]),
                         ("power", route_vals[:, 2])]:
        
        plt.hist(values)
        plt.title(name)
        plt.show()


def load_local_stats(db_path):
    with h5py.File(db_path, 'r') as db:
        stats = collect_stats_from_group(db)

    return stats


def collect_stats_from_group(h5py_group):
    # read in all global and per-route stats from database
    all_stats = defaultdict(list)
    def collect(name, obj):
        if 'scenario' in name:
            all_stats['num routes'].append(np.array(len(obj[...])))
        elif 'global ' in name or 'per-stop ' in name:
            # remove the part of the name corresponding to the "path" 
            name = name.rpartition('/')[2]
            vals = obj[...]
            all_stats[name].append(vals)

        for key, val in obj.attrs.items():
            all_stats['attr ' + key].append(val)
        return None
    h5py_group.visititems(collect)
    return all_stats


def plot_global_stats(db_path):
    # load database
    with h5py.File(db_path, 'r') as db:
        for key in db:
            if 'global ' in key:
                global_dset = db[key][...]
                dset_min = global_dset.min()
                dset_max = global_dset.max()
                bandwidth = (dset_max - dset_min) / 10
                estimator = KernelDensity(kernel="gaussian", 
                                          bandwidth=bandwidth)
                est_fn = estimator.fit(global_dset[:, None])
                plot_xs = np.linspace(dset_min, dset_max, 1000)
                plot_ys = np.exp(est_fn.score_samples(plot_xs[:, None]))
                # plt.hist(global_dset)
                plt.plot(plot_xs, plot_ys)
                plt.title(key)
                plt.show()


def plot_database_stats(stats_dict):
    # plot histograms for each stat
    for key, value_list in stats_dict.items():
        if len(value_list) == 1:
            value_array = value_list[0]
        if value_list[0].ndim == 0:
            value_array = np.stack(value_list)
        else:
            value_array = np.concatenate(value_list)
        # density_plot(value_array, key)
        value_array = value_array[value_array > 0]
        plt.hist(value_array, bins=20)
        plt.title(key)
        # plt.semilogy()
        # plt.loglog()
        plt.show()

    # read in all per-route stats from database, plot histograms of each
    # determine parameters of power law (or other law) governing global stats
    # if we balance the sample on global stats, does that result in routes with
     # balanced stats?


def draw_balanced_sample(values, sample_size, bandwidth_ratio=0.05):
    if values.ndim == 1:
        # add the 2nd dimension expected by KernelDensity
        values = values[:, None]
    # get a kernel density estimator of the data
    bandwidth = (values.max() - values.min()) * bandwidth_ratio
    estimator = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    est_fn = estimator.fit(values)
    scores = est_fn.score_samples(values)
    exp_scores = np.exp(scores)
    inverse_scores = 1 / exp_scores
    balanced_probs = inverse_scores / inverse_scores.sum()

    # sample values from the input collection according to their probabilities
    value_idxs = range(len(values))
    sample = np.random.choice(value_idxs, size=sample_size, p=balanced_probs)
    # return the sample
    return sample


def plot_stats_of_balanced_globals(db_path, key, sample_size=1000):
    with h5py.File(db_path, 'r') as db:
        global_data = db[key][...]
        sample_idxs = draw_balanced_sample(global_data, sample_size)
        # get route-specific stats for sample indexes
        # plot route-specific stats
        all_stats = defaultdict(list)
        # collect the local data
        for sample_idx in sample_idxs:
            grp = db[str(sample_idx)]
            stats = collect_stats_from_group(grp)
            for key, values in stats.items():
                all_stats[key] += values

        # collect the global data
        for iter_key in db:
            if 'global ' in iter_key:
                print("adding stats for global key", iter_key)
                sampled_values = db[iter_key][...][sample_idxs]
                all_stats[iter_key] = [sampled_values]

    # plot the results
    plot_database_stats(all_stats)


def test_loading_routes(db_path):
    import time
    with h5py.File(db_path, 'r') as db:
        st = time.perf_counter()
        cdts_grp = db['candidates']
        routes = []
        costs = []
        for dset in cdts_grp.values():
            routes.append(dset[...])
            costs.append(dset.attrs["cost"])
        total_time = time.perf_counter() - st
        print("total loading time:", total_time)


if __name__ == "__main__":
    # plot_global_stats(sys.argv[1])

    stats_dict = load_local_stats(sys.argv[1])
    plot_database_stats(stats_dict)
    # test_loading_routes(sys.argv[1])
    # plot_stats_of_balanced_globals(sys.argv[1], "global saved time")