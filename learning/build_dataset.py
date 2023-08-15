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
import numpy as np
from tqdm import tqdm
import pickle
import h5py
from collections import defaultdict
from sklearn.neighbors import KernelDensity

from bagloee import get_newton_routes
from simulation.timeless_sim import TimelessSimulator


def sample_batch(batch_size, points, use_counts, costs, max_budget, eps=1e-6):
    # determine point "temperatures" for each round: numbers between 0 and 1
    if points.max() > 0:
        # normalize the points
        point_scores = points / points.max()
    else:
        point_scores = points
    # this is to avoid NaNs when raising very small numbers to exponents.
    point_scores = point_scores.clip(min=eps)
    point_exps = np.random.random(batch_size)
    point_scores = point_scores[None] * point_exps[:, None]
    # more usage in the past = less chance of being used now
    # avoid division by 0
    use_scores = 1 - use_counts / max(use_counts.max(), 1)
    use_scores = use_scores[None] * (1 - point_exps[:, None])
    scores = point_scores + use_scores
    # set random budgets
    
    min_scale = 0.05
    budget_scales = np.random.random(batch_size)
    budget_scales = min_scale + budget_scales / (1 - min_scale)
    budgets = budget_scales * max_budget
    remaining_budget = budgets.copy()
    scenarios = []
    n_candidates = len(points)
    def get_under_budget():
        return remaining_budget[:, None] >= costs[None]
    is_valid_choice = get_under_budget()
    while is_valid_choice.any():
        next_choice = np.ones(batch_size, dtype=int)
        next_choice *= n_candidates
        
        # sample routes according to scores
        # # TODO haha, nope! sample them uniformly
        # scores = np.ones(scores.shape)
        scores[~is_valid_choice] = 0
        any_valid_choice = is_valid_choice.any(axis=1)
        choices = []
        for ei in range(batch_size):
            probs = scores[ei] / scores[ei].sum()
            choice = np.random.choice(n_candidates, p=probs)
            choices.append(choice)
        next_choice[any_valid_choice] = np.stack(choices)
        
        # add to the scenarios
        scenarios.append(next_choice)
        # update the budgets and validity mask
        remaining_budget[any_valid_choice] -= costs[choices]
        # invalidate choices that are now over-budget
        is_valid_choice &= get_under_budget()
        # invalidate choices that are already chosen
        is_valid_choice[any_valid_choice, choices] = False
        
    # concatenate the scenarios
    scenarios = np.stack(scenarios).T
    # compute the padding mask
    padding_mask = scenarios == n_candidates
    assert (~padding_mask).any(axis=1).all()
    # return the values
    return scenarios, padding_mask, budgets


def build_dataset(db_path, sim, candidate_routes, costs, max_budget, 
                  n_scenarios):
    n_cdts = len(candidate_routes)
    # set up tracking collections
    points = np.zeros(n_cdts)
    use_counts = np.zeros(n_cdts)
    all_route_vals = defaultdict(list)
    all_stop_vals = defaultdict(list)

    with h5py.File(db_path, 'w') as db:
        # write the candidate routes and costs to the database
        cdts_grp = db.create_group('candidates')
        for ci, (route, cost) in enumerate(zip(candidate_routes, costs)):
            route_dset = cdts_grp.create_dataset('route ' + str(ci),
                                                 data=np.array(route))
            route_dset.attrs["cost"] = cost

        global_datasets = {}
        for ii in tqdm(range(n_scenarios)):
            grp = db.create_group(str(ii))
            # sample batch_size systems
            scenarios, _, budget = \
                sample_batch(1, points, use_counts, costs, max_budget)
            scenario = scenarios[0]
            grp.create_dataset('scenario', data=np.array(scenario))
            grp.attrs['budget'] = budget

            scen_costs = costs[scenario]
            # simulate them to get the ground truth
            # here we are assuming costs are in capacity
            scen_freqs = sim.capacity_to_frequency(scen_costs)

            new_points = np.zeros(n_cdts)
            new_use_counts = np.zeros(n_cdts)

            assert len(np.unique(scenario)) == len(scenario), \
                "duplicate routes!"

            routes = [candidate_routes[ri] for ri in scenario]
            stop_info, global_info = sim.run(routes, scen_freqs)

            for key, value in global_info.items():
                global_key = 'global ' + key
                if global_key not in global_datasets:
                    # create the global array for this key
                    dataset = db.create_dataset(global_key, (n_scenarios))
                    global_datasets[global_key] = dataset

                # add the data to the global dataset
                global_datasets[global_key][ii] = value
                # grp.create_dataset('global ' + key, data=value)

            for key, values in stop_info.items():
                total_scenario_value = 0
                for cdt_idx, route_values in zip(scenario, values):
                    cdt_key = 'route ' + str(cdt_idx)
                    if cdt_key not in grp:
                        route_grp = grp.create_group(cdt_key)
                    else:
                        route_grp = grp[cdt_key]
                    dset = route_grp.create_dataset('per-stop ' + key, 
                                                   data=route_values)
                    total_route_value = route_values.sum()
                    dset.attrs['route total ' + key] = total_route_value
                    total_scenario_value += total_route_value

                    all_stop_vals[key].append(np.array(route_values))
                    all_route_vals[key].append(total_route_value)

                grp.attrs['total ' + key] = total_scenario_value

            # update use counts of routes
            new_use_counts[scenario] += 1
            # compute new points
            riderships = \
                np.array([sr.sum() for sr in stop_info["boarders"]])

            if riderships.max() > 0:
                ridership_fracs = riderships / riderships.sum()
                scenario_points = global_info["saved time"] * ridership_fracs
            else:
                scenario_points = np.zeros(riderships.shape)

            scenario_points /= budget / max_budget
            new_points[scenario] += scenario_points
            # scale points up based on budgets; if there's less budget, we 
                # shouldn't penalize the points for that

            use_counts += new_use_counts
            # average points over number of times the route appears in the batch
            is_in_batch = new_use_counts > 0
            avg_new_points = new_points[is_in_batch] / \
                             new_use_counts[is_in_batch]
            # assert not avg_new_points.isnan().any()
            # replace old point values
            points[is_in_batch] = avg_new_points

        db.attrs['# scenarios'] = n_scenarios
        
        for key, vals_list in all_route_vals.items():
            if vals_list[0].ndim == 0:
                value_array = np.stack(vals_list)
            else:
                value_array = np.concatenate(vals_list)
            db.create_dataset('all routes ' + key, data=value_array)

        for key, vals_list in all_stop_vals.items():
            if vals_list[0].ndim == 0:
                value_array = np.stack(vals_list)
            else:
                value_array = np.concatenate(vals_list)
            db.create_dataset('all stops ' + key, data=value_array)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_config', help="The simulator configuration file.")
    parser.add_argument('costs', 
        help="A pickle file containing the candidate route costs.")
    parser.add_argument('num_scenarios', type=int,
        help="The number of scenarios to generate and simulate.")
    parser.add_argument("--mb", type=float, default=30000, 
        help="maximum budget")
    parser.add_argument('--loop', action='store_true', 
        help="If provided, only looping routes will be generated.")
    parser.add_argument('-o', '--outfile', default="presim_database.h5",
        help="The path to the new database.")
    args = parser.parse_args()

    sim = TimelessSimulator(args.sim_config, filter_nodes=True, 
                            # stops_path="/localdata/ahollid/laval/gtfs/stops.txt"
                            )
    routes = get_newton_routes(sim, min_access_egress=1, 
                               all_routes_are_loops=args.loop)
    with open(args.costs, "rb") as ff:
        costs, _ = pickle.load(ff)

    build_dataset(args.outfile, sim, routes, costs, args.mb, 
                  args.num_scenarios)


if __name__ == "__main__":
    main()