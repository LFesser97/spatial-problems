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
import logging as log
import pickle
from pathlib import Path
from collections import defaultdict
from typing import Tuple
from datetime import datetime

import yaml
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from torch_utils import floyd_warshall
from simulation.timeless_sim import TimelessSimulator, get_inter_stop_demand
from learning.num_scenarios_tracker import TotalInfoTracker


def bagloee_filtering(sim, seive_threshold=0.5, cluster_threshold_m=300):
    """Based on the route-generation method of Bagloee and Ceder 2011"""
    # compute weights of each candidate stop
    demand_mat = sim.get_demand_matrix()
    basin_mat = get_basin_adjacency_mat(sim)
    weights = get_stop_demand_weights(basin_mat, demand_mat)

    # seive nodes based on their weights?
    if seive_threshold is not None:
        seived_nodes = torch.where(weights >= seive_threshold)[0]
    else:
        # keep all nodes
        seived_nodes = torch.arange(len(weights))
        
    # cluster and extract candidate stops?
    if cluster_threshold_m is not None:
        kept_nodes = []
        # sort the nodes by descending order of weights
        candidates = sorted(seived_nodes, key=lambda xx: weights[xx],
                            reverse=True)
        # iterate over sorted nodes:
        for candidate in candidates:
            if len(kept_nodes) == 0 or \
                (sim.drive_dists[candidate, kept_nodes] >= \
                    cluster_threshold_m).all():
                # node is not within 0.3km of any kept node, so keep it
                kept_nodes.append(candidate)
        
        # convert it to a tensor
        kept_nodes = torch.tensor(kept_nodes)
    else:
        kept_nodes = seived_nodes

    kept_weights = weights[kept_nodes]

    log.info(f"{len(kept_nodes)} of "\
        f"{sim.street_graph.number_of_nodes()} nodes were kept")
    return kept_nodes, kept_weights


def get_newton_routes(sim, alpha=0.5, dsp_step=0.05, 
                      min_access_egress=5.92, max_dsp_threshold=50, 
                      dsp_feeder_threshold=42, dsp_local_threshold=10, 
                      min_route_length_m=1800, use_biased_length_index=True, 
                      mass_epsilon=0.03, all_routes_are_loops=False, 
                      device=None):
    """Based on the route-generation method of Bagloee and Ceder 2011"""
    # select only the parts of the basin matrix that we keep
    basin_mat_dense = get_basin_adjacency_mat(sim, device)
    demand_mat_dense = sim.get_demand_matrix().to(device=device)
    weights = get_stop_demand_weights(basin_mat_dense, demand_mat_dense)
    inter_stop_demand = get_inter_stop_demand(basin_mat_dense.T, 
                                              demand_mat_dense)

    # compute an approximate measure of stop-level demand
    demand_mat = demand_mat_dense.to_sparse()

    # scale it so it has the correct total demand
    scale_factor = demand_mat_dense.sum() / inter_stop_demand.sum()
    inter_stop_demand *= scale_factor
    fractional_demand = 1 + (inter_stop_demand / inter_stop_demand.max())

    # compute the attraction indices for each link (pair of nodes) (eqn 13)
    attr_indices = weights[None] + weights[:, None]
    attr_indices.fill_diagonal_(0)
    drivetimes_s = sim.drive_times.to(device)
    # convert to minutes, since that's what the paper uses
    drivetimes_mnt = drivetimes_s / 60

    drivedists_m = sim.drive_dists.to(device)
    attr_indices /= torch.exp(drivedists_m / 300)

    # compute the free transit flow over each link. This is demand flow 
        # over each link assuming all demand follows the shortest path.

    # get journey times and and shortest paths in the street graph
    # reduce the graph
    log.info("reducing graph...")
    reduced_street_edges = sim.get_street_edge_matrix()

    # compute all shortest paths
    path_obj = floyd_warshall(reduced_street_edges)

    log.info("computing free transit flow")
    free_transit_flow = torch.zeros((sim.n_street_nodes, sim.n_street_nodes), 
                                    device=device)
    demand_by_free_path_time = defaultdict(float)

    # for each demand, compute its contribution to free transit flow
    basin_wt_mat = sim.get_basin_walk_times_matrix(device)
    for ii, jj in demand_mat.indices().t():
        if ii == jj:
            continue

        candidate_times = []
        for si, iwalk in enumerate(basin_wt_mat[ii]):
            if iwalk == float('inf'):
                continue
            for sj, jwalk in enumerate(basin_wt_mat[jj]):
                if jwalk == float('inf'):
                    continue
                candidate_time = iwalk + jwalk + drivetimes_s[si, sj]
                candidate_times.append((si, sj, candidate_time))
        if len(candidate_times) == 0:
            continue
        si, sj, _ = min(candidate_times, key=lambda tt: tt[2])

        # filter the path, discarding non-kept nodes
        path = path_obj.get_path(si, sj)
        # allocate the edge's demand to the street edges on the path
        for ll, ss in enumerate(path):
            free_transit_flow[ss, path[ll+1:]] += demand_mat[ii, jj]
        drivetime_mnt = drivetimes_mnt[path[0], path[-1]]
        demand_by_free_path_time[drivetime_mnt.item()] += \
            demand_mat[ii, jj].item()

    log.info("computing manipulated shortest paths")
    # equation 12
    # using the above, compute the manipulated travel time on each link
    local_adjusted_path_times = drivetimes_mnt / \
        (attr_indices + 0.09)**alpha
    adjusted_path_times = local_adjusted_path_times / \
        (free_transit_flow + 1)**(1 - alpha)
    # add stop times to costs, since taking an edge in this matrix means
    # stopping at its endpoint
    adjusted_path_times += sim.cfg["mean_stop_time_s"] / 60

    # compute all shortest paths through the adjusted path graph
    nonlocal_paths = floyd_warshall(adjusted_path_times)
    nonlocal_tr_se = nonlocal_paths.dists + nonlocal_paths.dists.t()
    local_paths = floyd_warshall(local_adjusted_path_times)
    local_tr_se = local_paths.dists + local_paths.dists.t()

    if use_biased_length_index:
        # equations 10 and 11
        log.info("computing biased length index params")

        # calibrate logistic function of relative trip length to get
            # formula for delta
        # assemble the input xs and zs
        times_and_demands = list(demand_by_free_path_time.items())
        # this sorts by the first element, trip time
        times, demands = np.array(list(zip(*sorted(times_and_demands))))
        # xs = normalized trip lengths
        t_max = times.max()
        xs = (times / t_max).reshape(-1, 1)
        # ys = fractions of trips with length leq x
        ys = demands.cumsum() / demands.sum()
        # make it just a bit less than 1, to avoid division by zero in zs
        ys[-1] = 0.99999
        # zs = logit(ys)
        zs = np.log(ys / (1 - ys))
        # run linear regression and get parameters
        delta_fit = LinearRegression().fit(xs, zs)
        log.info(f"r^2 for our trip length model is \
            {delta_fit.score(xs, zs)}")
        # check the quality of the estimated model, print its r^2
        zpreds = delta_fit.predict(xs)
        # y = sigmoid(z)
        ypreds = 1 / (1 + np.exp(-zpreds))
        rmse = np.sqrt(((ypreds - ys)**2).mean())
        log.info(f"RMSE of our logistic predictor: {rmse}")

    # select candidates!

    # calculate masses and frictions and set thresholds for the mass phase
    start_masses = weights + mass_epsilon
    # friction (s,e) in first phase is geometric mean of travel time from
        # s->e and e->s
    drivetime_geomeans = (drivetimes_mnt * drivetimes_mnt.t()).sqrt()

    # total in-and-out free flow volume at each node
    vols = free_transit_flow.sum(dim=0) + free_transit_flow.sum(dim=1)

    proposed_routes = []
    access_egress_counts = torch.zeros(sim.n_street_nodes, device=device)
    times_selected = torch.zeros(sim.n_street_nodes, device=device)
    # mask to ignore already-chosen routes and self-connections.
    log.info("route-gen loop begins!")
    # avoid selecting s,e pairs closer than the min walking distance
    walk_dists = torch.min(drivedists_m, drivedists_m.t())
    if all_routes_are_loops:
        shortest_times = drivetimes_s + drivetimes_s.t()
    else:
        shortest_times = drivetimes_s
    mask = (walk_dists < min_route_length_m) | (inter_stop_demand == 0)
    while (access_egress_counts < min_access_egress).any() and \
            not mask.all():
        # set initial DSP - the paper is unclear on how this is done
        dsp_threshold = max_dsp_threshold
        paths = nonlocal_paths
        too_indirect = nonlocal_tr_se > shortest_times * np.pi / 2
        mass_products = start_masses[:, None] * start_masses[None]

        max_dsp_remaining = inter_stop_demand[~(mask | too_indirect)].max()
        dsp_threshold = min(max_dsp_threshold, max_dsp_remaining)

        prior_num_inneed = (access_egress_counts < min_access_egress).sum()

        while dsp_threshold > 0:
            below_dsp_threshold = (inter_stop_demand < dsp_threshold)
            # recalculate the gravity indices with new random rho samples
             # (equations 7, 8, 9)
            # TODO we're assuming here that the same random value is used
            # for the exponents of each candidate route, but that's not 
            # clear from the paper.  I've sent an e-mail to the authors,
            # I hope they'll clarify.
            rho = torch.rand(1, device=device)
            if dsp_threshold > dsp_feeder_threshold:
                numerator = (max_dsp_threshold - dsp_threshold)
                denominator = (max_dsp_threshold - dsp_feeder_threshold)
                p_i = 0.5 * (1 + numerator / denominator)
                if all_routes_are_loops:
                    frictions = drivetime_geomeans ** p_i
                else:
                    frictions = drivetimes_mnt ** p_i
            else:
                mass_s = weights.detach().clone()
                mass_e = vols + min_access_egress * access_egress_counts
                if dsp_threshold > dsp_local_threshold:
                    # we're not in the last stage
                    mass_s *= rho ** access_egress_counts
                    mass_e *= access_egress_counts ** rho
                # calculate mass products for the feeder phase
                mass_s += mass_epsilon
                mass_e += mass_epsilon
                if dsp_threshold <= dsp_local_threshold:
                    # in the last stage:
                    # only start at inadequately-connected stops
                    mass_s[access_egress_counts >= min_access_egress] = 0
                    # only end at adequately-connected stops
                    mass_e[access_egress_counts < min_access_egress] = 0
                mass_products = mass_s[:, None] * mass_e[None]
            
            gravs = (mass_products**rho).to(torch.float64)
            # avoid division by 0
            frictions.fill_diagonal_(1)
            gravs /= frictions
            # multiply in fractional d_c^{s,e}
            gravs *= fractional_demand
            # multiply in penalty for often-selected terminals
            ts_penalty = rho**times_selected
            gravs *= ts_penalty[:, None] * ts_penalty[None]
            if use_biased_length_index:
                # compute delta
                # TODO should it be a different rho for every trip?  I suspect
                # not but I'm not certain.
                delta_rho = torch.rand(1)
                logit = torch.log((1 - delta_rho) / delta_rho)
                numerator = (logit + delta_fit.intercept_) * t_max
                delta = numerator / torch.tensor(-delta_fit.coef_)
                delta = delta.to(device)
                # multiply in bias-length-index penalty
                biased_length_index = \
                    ((drivetimes_mnt - delta) / delta).abs()
                gravs *= rho ** biased_length_index

            gravs[mask | too_indirect | below_dsp_threshold] = 0

            # pick the route to add with the maximum gravity index
            max_s, max_e = unravel_indices(torch.argmax(gravs), 
                                           gravs.shape)
            if gravs[max_s, max_e] == 0:
                # jump to the next dsp threshold with some validity
                # ignore pairs with mass product 0, since they'll never
                # get chosen
                valid_isd = ~(mask | too_indirect | (mass_products == 0))
                if not valid_isd.any():
                    new_dsp_threshold = 0
                else:
                    max_valid_isd = inter_stop_demand[valid_isd].max()
                    lower = min(max_valid_isd, dsp_threshold)
                    new_dsp_threshold = max(lower - dsp_step, 0)
            else:
                max_s = max_s.item()
                max_e = max_e.item()
                route = paths.get_path(max_s, max_e)
                mask[max_s, max_e] = True
                if all_routes_are_loops:
                    route_back = paths.get_path(max_e, max_s)
                    mask[max_e, max_s] = True
                    route += route_back[1:]

                # add the path to proposed_routes
                proposed_routes.append(route)
                # if dsp_threshold < dsp_local_threshold:
                #     log.info('added a route in the third stage')
                # increment terminals' selected-counts
                times_selected[max_s] += 1
                times_selected[max_e] += 1
                # update access_egress_counts

                for ri in range(len(route) - 1):
                    access_egress_counts[route[ri]] += 1
                    access_egress_counts[route[ri+1]] += 1

                # update the threshold
                new_dsp_threshold = dsp_threshold - dsp_step

            # do any recalculation needed for this stage
            if new_dsp_threshold <= dsp_feeder_threshold and \
                dsp_threshold > dsp_feeder_threshold:
                # calculate frictions for stages 2 and 3
                frictions = drivetime_geomeans.exp()
            if new_dsp_threshold <= dsp_local_threshold and \
                dsp_threshold > dsp_local_threshold:
                # recompute remaining routes to ignore flow
                paths = local_paths
                too_indirect = local_tr_se > shortest_times * np.pi / 2
            
            dsp_threshold = new_dsp_threshold

        need_more = access_egress_counts < min_access_egress
        if need_more.sum() == prior_num_inneed:
            log.warning(
                f"ending with {len(need_more)} stops without enough " \
                    "access!")
            break
        ae_mask = mask[need_more]
        ae_ti = too_indirect[need_more]
        if inter_stop_demand[need_more][~(ae_mask | ae_ti)].sum() == 0:
            # all stops with less than the minimum accesses and egresses
            # have no demand, so halt.
            log.info("ending because all stops that need more accesses have "\
                "no demand")
            break
        
    log.info(f"{len(proposed_routes)} routes proposed")
    return proposed_routes


def bagloee_rsfs(sim, candidate_routes, init_capacity, budget, 
                 termination_limit, quality_metric="saved time", 
                 vehicle=None, fast_demand_assignment=False, sum_writer=None,
                 device=None):
    """
    Route selection and frequency setting as in Bagloee & Ceder 2011.

    arguments
    candidate_routes -- a list of candidate bus routes
    nn -- the number of iterations to run
    init_capacity -- the initial capacity needed for each route.  Should be
        the mean number of boarding passengers in real transit.  400 for
        Winnipeg in the paper.
    budget -- the total available capacity.
    termination_limit -- the number of phase 1 iterations with no improvement 
        after which to terminate.
    quality_metric -- the string key of the sim output to use as the quality
        metric.
    vehicle -- the type of vehicle to use.  Multiple vehicle types are not
        supported at present.  If None, the default from the config will be
        used.
    fast_demand_assignment -- if True, simulate using fast demand 
        assignment.  Otherwise, use the slower but more consistent 
        hyperpath method, more consistent with the original paper.
    sum_writer -- a tensorboard summary writer.  If provided, quality will be
        logged as the algorithm progresses.
    """
    # set up variables for the loop

    # compute duration of period being analyzed, in seconds
    if vehicle is None:
        vehicle = sim.get_default_vehicle_type()
    points = []
    n_cdts = len(candidate_routes)
    req_cpcties = np.ones(n_cdts) * init_capacity
    use_counts = np.zeros(n_cdts, dtype=int)
    past_scenarios = []
    past_freqs = []
    top_qual_and_idxs = []

    scen_tracker = TotalInfoTracker(len(candidate_routes), device)

    def get_weights(scenario):
        # the paper doesn't describe this weighting by selection count, but
         # makes the algorithm run about as fast as they report, and doesn't
         # hurt performance.
        weights = 1 / (use_counts + 1)
        weights[req_cpcties > remaining_budget] = 0
        weights[scenario] = 0
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weights.sum()
        return weights

    phase = 0
    log.info(f"termination limit is {termination_limit}")
    ii = 1
    pbar = tqdm()
    while True:
        freqs_Hz = sim.capacity_to_frequency(req_cpcties)
        headways_s = 1 / freqs_Hz
        scenario = []
        if ii <= 6 or not (use_counts > 0).all():
            # we're in phase 0
            # Build a random scenario, upweighting unused routes
            # set the budget (equation 33)
            remaining_budget = budget * max(0.1, np.random.random() ** 2)
            weights = get_weights(scenario)
            while remaining_budget > 0 and weights.sum() > 0:
                # randomly pick a route from the candidates
                route_idx = np.random.choice(n_cdts, p=weights)
                # add the route to the scenario
                scenario.append(route_idx)
                # update the remaining budget
                remaining_budget -= req_cpcties[route_idx]
                # set weight to zero to avoid double-picking
                weights = get_weights(scenario)
                if weights.sum() == 0 and set(scenario) in past_scenarios and \
                     np.random.random() < 0.1:
                    # this scenario was already used.  Mutate it!
                    _, removed_route_idx = pop_random(scenario)
                    remaining_budget += req_cpcties[removed_route_idx]
                    weights = get_weights(scenario)

            use_counts[scenario] += 1
            n_unused = n_cdts - (use_counts > 0).sum()
            log.debug(f"{n_unused} candidate routes have not been used")
                
        else:
            # we're in phase 1
            if phase == 0:
                log.info(f"phase 0 lasted {ii - 1} iterations")
                steps_since_improvement = 0
                phase = 1
                phase_0_req_cpcties = req_cpcties.copy()
                phase_0_cpcty_checkpoint = (phase_0_req_cpcties, ii - 1)

            # compute the collective points (equation 34)
            top_6_quals, top_6_idxs = zip(*top_6_qual_and_idxs)
            top_6_points = np.array([points[ii] for ii in top_6_idxs])
            proportions_of_best = (np.array(top_6_quals) / top_6_quals[5])**2
            weights = np.random.random(6) * proportions_of_best
            clctv_pts = (weights[:, None] * top_6_points).sum(axis=0)
            random_headway_denom = (2 * np.random.random())**(1/ii)
            max_headway_s = sim.sim_duration_s / random_headway_denom

            # for route in sorted routes:
            sorted_route_idxs = sorted(range(n_cdts), 
                                       key=lambda ri: clctv_pts[ri], 
                                       reverse=True)

            remaining_budget = budget
            for route_idx in sorted_route_idxs:
                if req_cpcties[route_idx] < remaining_budget and \
                    headways_s[route_idx] < max_headway_s:
                    # add the route to the scenario
                    scenario.append(route_idx)
                    remaining_budget -= req_cpcties[route_idx]

            mut_count = 0
            while set(scenario) in past_scenarios and mut_count < 5 * n_cdts:
                # log.info("mutating scenario...")
                # mutate scenario as above
                if len(scenario) > 0:
                    _, removed_route_idx = pop_random(scenario)
                    remaining_budget += req_cpcties[removed_route_idx]
                is_valid = (req_cpcties < remaining_budget) & \
                    (headways_s < max_headway_s)
                is_valid[scenario] = False
                if not is_valid.any():
                    log.warning("phase 1 generating empty scenarios repeatedly!")
                    break
                chosen_idx = \
                    np.random.choice(n_cdts, p=is_valid / is_valid.sum())
                remaining_budget -= req_cpcties[chosen_idx]
                scenario.append(chosen_idx)
                mut_count += 1
            if len(scenario) > 0:
                use_counts[scenario] += 1

        scen_tracker.add_scenario(scenario)

        # simulate the proposed scenario
        scen_routes = [candidate_routes[ri] for ri in scenario]
        scen_freqs = freqs_Hz[scenario]
        # in phase 0, use capacity-free assignment
        stop_info, global_info = \
            sim.run(scen_routes, scen_freqs, [vehicle] * len(scen_routes), 
                    capacity_free=(phase == 0),
                    fast_demand_assignment=fast_demand_assignment)
            
        # get the objective function value
        quality = global_info[quality_metric]
        if sum_writer:
            sum_writer.add_scalar('n scenarios seen', 
                                  scen_tracker.n_scenarios_so_far, ii-1)
            sum_writer.add_scalar('total exploration', 
                                  scen_tracker.total_info(), ii-1)
            sum_writer.add_scalar('quality', quality, ii-1)
            sum_writer.add_scalar('# routes', len(scenario), ii-1)
            # average stops per route
            avg_stops_per_route = np.mean([len(rr) for rr in scen_routes])
            sum_writer.add_scalar('# stops per route avg.', 
                                  avg_stops_per_route, ii-1)

            sum_writer.add_scalar('headway avg', np.mean(headways_s[scenario]),
                                  ii-1)
            sum_writer.add_scalar('headway std dev', 
                                  np.std(headways_s[scenario]), ii-1)
            pcnt_budget_used = 100 * (budget - remaining_budget) / budget
            sum_writer.add_scalar('budget used (%)', pcnt_budget_used, ii-1)
            for kk, vv in global_info.items():
                sum_writer.add_scalar(kk, vv, ii-1)

        # update the top six qualities
        if len(top_qual_and_idxs) < 6:
            # we have less than six, so add each one
            top_qual_and_idxs.append((quality, ii-1))
            if len(top_qual_and_idxs) == 6:
                # we've reached six, so sort them
                top_6_qual_and_idxs = sorted(top_qual_and_idxs)
        elif quality > top_6_qual_and_idxs[0][0]:
            # this scenario should be in the top 6, so remove old 6th-best
            top_6_qual_and_idxs.pop(0)
            # determine this scenario's placement in top 6
            rank_in_top6 = 0
            for t6q, _ in top_6_qual_and_idxs:
                if quality > t6q:
                    rank_in_top6 += 1
                else:
                    break
            top_6_qual_and_idxs.insert(rank_in_top6, (quality, ii-1))

        if phase == 1:
            if quality < top_6_qual_and_idxs[5][0]:
                # we haven't improved...
                steps_since_improvement += 1
                log.debug(f"steps since improvement: {steps_since_improvement}")
            else:
                steps_since_improvement = 0
                log.info(f"best scenario updated with quality {quality}")
                if sum_writer:
                    # log new best
                    sum_writer.add_scalar('best quality', quality, ii-1)

        # update points (equation 28)
        per_route_boarders = \
            np.array([rb.sum() for rb in stop_info["boarders"]])
        n_riders = per_route_boarders.sum()
        if len(points) == 0:
            points_i = np.zeros(n_cdts)
        else:
            points_i = points[-1].copy()
        # make denominator at least 1 to avoid division by 0
        points_i[scenario] = quality * per_route_boarders / max(n_riders, 1)
        assert not np.isnan(points_i).any()
        points.append(points_i)

        new_caps = get_required_capacities(sim, scen_routes, stop_info)

        # update the required capacity (equation 29)
        req_cpcties[scenario] += new_caps / ii
        req_cpcties[scenario] /= (1 + 1 / ii)
        # req_cpcties[scenario] += geomean_route_loads / use_counts[scenario]
        # req_cpcties[scenario] /= (1 + 1 / use_counts[scenario])

        # save this iteration's scenario, even if it's a duplicate
        set_scenario = set(scenario)
        past_scenarios.append(set_scenario)
        past_freqs.append(scen_freqs)
        
        if phase == 1 and steps_since_improvement == termination_limit:
            # no improvement for a while, so terminate.
            break

        ii += 1
        pbar.update(1)

    pbar.close()
    log.info(f"{ii} iterations were run")

    # return the best scenario routes and frequencies
    _, best_idx = top_6_qual_and_idxs[-1]
    best_scenario = list(past_scenarios[best_idx])
    routes = [candidate_routes[ri] for ri in best_scenario]
    freqs = past_freqs[best_idx]
    # save an image of the best routes and frequencies
    image = sim.get_network_image(routes, freqs)
    sum_writer.add_image('City with routes', image, ii)
    
    return routes, freqs, phase_0_cpcty_checkpoint


def get_required_capacities(sim, routes, stop_info):
    # compute the route load (equation 30)
     # they compute the load factors and multiply that by route cpcty,
     # but that seems to just give back the load, so use the load.
    geomean_route_loads = []
    loads_by_route = compute_loads(stop_info["boarders"], 
                                    stop_info["disembarkers"])
    for route, route_loads in zip(routes, loads_by_route):
        times = [sim.drive_times[route[si], route[si + 1]] 
                    for si in range(len(route) - 1)]
        time_weighted_loads = route_loads * times / sum(times)
        mean_load = time_weighted_loads.sum()
        geomean_route_load = np.sqrt(mean_load * max(route_loads))
        assert not np.isnan(geomean_route_load)
        geomean_route_loads.append(geomean_route_load)
    geomean_route_loads = np.array(geomean_route_loads)
    return geomean_route_loads


def unravel_indices(
    indices: torch.LongTensor,
    shape: Tuple[int, ...],
) -> torch.LongTensor:
    r"""Converts flat indices into unraveled coordinates in a target shape.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = torch.div(indices, dim, rounding_mode='floor')

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def pop_random(list_to_pop_from):
    pick = np.random.randint(len(list_to_pop_from))
    return pick, list_to_pop_from.pop(pick)


def compute_loads(boarders, disembarkers):
    # we might sometimes get very small negative values due to float precision
    deltas = [bb.clip(0) - dd.clip(0) 
              for bb, dd in zip(boarders, disembarkers)]
    # the load is the sum of the deltas so far
    # exclude the last, since it's the post-run load, which should be 0
    loads = [dd.cumsum()[:-1] for dd in deltas]
    return loads


def get_basin_adjacency_mat(sim, device=None):
    """Returns an n_stop_nodes x n_demand_nodes matrix with values between
       0 and 1, exponentially decaying """
    basin_mat = sim.get_basin_walk_times_matrix(device).T
    basin_adj_mat = 2 ** (-basin_mat / 60)
    return basin_adj_mat


def get_stop_demand_weights(basin_mat, demand_mat):
    weights = torch.zeros(basin_mat.shape[0])
    demand_mat = demand_mat.to(device=basin_mat.device)
    loc_total_demands = demand_mat.sum(dim=0) + demand_mat.sum(dim=1)

    weights = [(loc_total_demands * row).sum() for row in basin_mat]
    weights = torch.stack(weights)
    return weights


def get_newton_routes_from_config(config_path, device=None, regenerate=False):
    config_path = Path(config_path)
    with open(config_path, "r") as ff:
        config = yaml.load(ff, Loader=yaml.Loader)

    # load the sim config yaml directly for comparison purposes
    sim_cfg_path = config['sim_args']['config_path']
    with open(sim_cfg_path, "r") as ff:
        sim_config = yaml.load(ff, Loader=yaml.Loader)
    config['sim_config'] = sim_config

    sim = TimelessSimulator(**config["sim_args"])
    # filter the nodes
    kept_nodes, _ = bagloee_filtering(sim, **config['filter_args'])
    sim.filter_nodes(kept_nodes)

    pickle_path = config_path.parent / (config_path.stem + '.pkl')
    if not regenerate and pickle_path.exists():
        # load the configuration
        with open(pickle_path, 'rb') as ff:
            saved_config, routes = pickle.load(ff)
        
        if saved_config == config:
            log.info('loading pre-generated newton routes')
            # the configuration matches the input config, so no need to 
            # regenerate.
            return sim, routes

        else:
            log.warning('Provided newton routes config does not match config '\
                'of stored routes.  Regenerating.')

    # if we get here, we need to (re)generate and save the routes
    routes = \
        get_newton_routes(sim, device=device, **config["newton_routes_args"])
    with open(pickle_path, 'wb') as ff:
        pickle.dump((config, routes), ff)
    
    # return the simulator as well, since it's slow to instantiate
    return sim, routes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    # 376 vehicles * 75 seats per vehicle
    parser.add_argument("--budget", "-b", type=int, default=28200, 
                        help="budget in seats for the run")
    parser.add_argument('--logdir', default="training_logs",
        help="Directory in which to log training output.")
    parser.add_argument('--run_name', '--rn',
        help="Name of this run, for logging purposes.")
    parser.add_argument("--qm", default="saved time", 
                        help="quality metric to use during training")
    parser.add_argument('--log', help="Specify the logging level to use.")
    parser.add_argument("--regen", action="store_true",
                        help="Force regeneration of routes.")
    args = parser.parse_args()

    if args.log:
        level = getattr(log, args.log.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"Invalid log level: {level}")
        log.basicConfig(level=level, format="%(asctime)s %(message)s")
    
    device = torch.device("cuda")
    sim, candidate_routes = \
        get_newton_routes_from_config(args.config_path, device, args.regen)
    # number of riders / number of routes in the real-world transit system
    init_capacity = 272
    # build the summary writer
    log_path = Path(args.logdir)
    run_name = "bagloee_"
    if args.run_name is None:
        run_name += datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
    else:
        run_name += args.run_name
    sum_writer = SummaryWriter(log_path / run_name)
    sum_writer.add_text("seating budget", str(args.budget), 0)

    # run route setting and frequency selection
    routes, freqs, phase_0_cpcty_checkpoint = \
        bagloee_rsfs(sim, candidate_routes, init_capacity, args.budget, 
                     termination_limit=12, sum_writer=sum_writer, 
                     quality_metric=args.qm,
                     )

    # save the phase 0 required capacities
    config_path = Path(args.config_path)
    pkl_filename = config_path.stem + 'required_capacities.pkl'
    cpcties_path = config_path.parent / pkl_filename
    with open(cpcties_path, 'wb') as ff:
        pickle.dump(phase_0_cpcty_checkpoint, ff)

    # save data about the run
    metadata = {"best routes": routes,
                "best freqs": freqs,
                "args": args, 
                "sim config": sim.cfg,
                "run type": "Bagloee & Ceder 2011"}
    output_dir = Path("outputs")
    if not output_dir.exists():
        output_dir.mkdir()
    with open(output_dir / (run_name + '_meta.pkl'), "wb") as ff:
        pickle.dump(metadata, ff)

    # render the routes and freqs
    html_path = output_dir / (run_name + '.html')
    sim.render_plan_on_html_map(routes, freqs, outfile=html_path)


if __name__ == "__main__":
    main()