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

import logging as log
import math
import pickle
from pathlib import Path

import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from omegaconf import DictConfig, OmegaConf
import hydra

from torch_utils import reconstruct_all_paths, aggregate_dense_conns, \
    get_update_at_mask, get_batch_tensor_from_routes
from simulation.citygraph_dataset import CityGraphDataset, CityGraphData, \
    get_dataset_from_config
import learning.utils as lrnu


def bee_colony(graph_data, n_routes, min_route_len, max_route_len, cost_obj, 
               n_bees=10, passes_per_it=5, mod_steps_per_pass=2, 
               shorten_prob=0.2, n_iters=400, n_type_1_bees=None, silent=False,
               init_scenario=None, cost_batch_size=None, symmetric_routes=True,
               bee_model=None, sum_writer=None):
    """Implementation of the method of  Nikolic and Teodorovic (2013).
    
    graph_data -- A CityGraphData object containing the graph data.
    cost_fn -- A function that determines the cost (badness) of a scenario.  In
        the paper, this is wieghted total travel time.
    n_routes -- The number of routes in each candidate scenario, called NBL in
        the paper.
    n_bees -- The number of worker bees, called B in the paper.
    passes_per_it -- The number of forward and backward passes to perform each
        iteration, called NP in the paper.
    mod_steps_per_pass -- The number of modifications each bee considers in the
        forward pass, called NC in the paper.
    shorten_prob -- The probability that type-2 bees will shorten a route,
        called P in the paper.  In their experiments, they use 0.2.
    n_iters -- The number of iterations to perform, called IT in the paper.
    n_type_1_bees -- There are 2 types of bees used in the algorithm, which
        modify the solution in different ways.  This parameter determines the
        balance between them.  The paper isn't clear how many of each they use,
        so by default we make it half-and-half.
    silent -- if true, no tqdm output or printing
    bee_model -- if a torch model is provided, use it as the only bee type.
    """
    if n_type_1_bees is None:
        n_type_1_bees = n_bees // 2

    if not isinstance(graph_data, Batch):
        graph_data = Batch.from_data_list([graph_data])
    batch_size = graph_data.num_graphs
    if cost_batch_size is None:
        cost_batch_size = batch_size * n_bees

    dev = graph_data.demand.device
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = graph_data.demand.shape[-1]

    # get all shortest paths
    shortest_paths, _ = reconstruct_all_paths(graph_data.nexts)

    # generate initial scenario
    log.info("generating initial scenario")

    demand = torch.zeros((batch_size, max_n_nodes+1, max_n_nodes+1), 
                         device=dev)
    demand[:, :-1, :-1] = graph_data.demand
    if init_scenario is None:
        best_scenarios = build_init_scenario(shortest_paths, demand, n_routes,
                                             symmetric_routes)
    else:
        best_scenarios = torch.full((batch_size, n_routes, max_n_nodes), -1,
                                     device=dev)
        best_scenarios[:, :, :init_scenario.shape[-1]] = init_scenario

    # compute can-be-directly-satisfied demand matrix
    direct_sat_dmd = get_direct_sat_dmd(demand, shortest_paths, 
                                        symmetric_routes)

    # set up required matrices
    street_node_neighbours = (graph_data.street_adj.isfinite() &
                              (graph_data.street_adj > 0))
    bee_idxs = torch.arange(n_bees, device=dev)                              

    log.info("starting BCO")

    # set up the cost function to work with batches of bees
    data_list = graph_data.to_data_list()
    n_cost_batches = int(math.ceil(n_bees * batch_size / cost_batch_size))
    exp_data = sum(([dd] * n_bees for dd in data_list), [])
    cost_batched_data = [Batch.from_data_list(exp_data[ii:ii+cost_batch_size]) 
                         for ii in range(0, len(exp_data), cost_batch_size)]
    if bee_model is not None:
        exp_batched_data = Batch.from_data_list(exp_data)
        env_states = bee_model.setup_planning(exp_batched_data, n_routes,
                                              min_route_len, max_route_len,
                                              cost_obj.get_weights())
    else:
        env_states = None

    def batched_cost_fn(bee_scenarios):
        # split along the bee dimension
        bee_scenarios = bee_scenarios.flatten(0,1)
        split_scens = torch.tensor_split(bee_scenarios, n_cost_batches, dim=0)

        costs = []
        for scens, batch in zip(split_scens, cost_batched_data):
            chunk_costs = cost_obj(scens, batch)
            costs.append(chunk_costs)

        costs = torch.cat(costs, dim=0)
        return costs.reshape(batch_size, n_bees)

    # initialize bees' scenarios
    # batch size x n_bees x n_routes x max_n_nodes
    bee_scenarios = best_scenarios[:, None].repeat(1, n_bees, 1, 1)
    # evaluate and record the reward of the initial scenario
    bee_costs = batched_cost_fn(bee_scenarios)
    best_costs = bee_costs[:, 0].clone()

    cost_history = torch.zeros((batch_size, n_iters + 1), device=dev)
    cost_history[:, 0] = best_costs

    for iteration in tqdm(range(n_iters), disable=silent):
        for pi in range(passes_per_it):
            # do forward pass
            for mi in range(mod_steps_per_pass):
                # flatten batch and bee dimensions
                expanded_demand = demand[:, None].expand(-1, n_bees, -1, -1)
                flat_exp_demand = expanded_demand.flatten(0, 1)
                flat_bee_scens = bee_scenarios.flatten(0, 1)
                route_dsds = aggregate_dense_conns(flat_bee_scens, 
                                                   flat_exp_demand[..., None])
                route_dsds.squeeze_(-1)
                # choose routes to modify
                route_scores = 1 / route_dsds
                route_scores[route_scores.isinf()] = 10**10
                flat_chosen_route_idxs = route_scores.multinomial(1).squeeze(1)
                chosen_route_idxs = flat_chosen_route_idxs.reshape(batch_size, 
                                                                   n_bees)
                
                new_bee_scenarios = \
                    get_2typebee_variants(bee_scenarios, chosen_route_idxs, 
                                          n_type_1_bees, direct_sat_dmd, 
                                          shorten_prob, street_node_neighbours, 
                                          shortest_paths, 
                                          bee_model, env_states
                                          )

                # compute costs and replace scenarios with better variants
                new_bee_costs = batched_cost_fn(new_bee_scenarios)
                better_idxs = new_bee_costs < bee_costs
                bee_scenarios[better_idxs] = new_bee_scenarios[better_idxs]
                bee_costs[better_idxs] = new_bee_costs[better_idxs]

            # do backward pass
            # decide whether each bee is a recruiter or follower
            max_bee_costs, _ = bee_costs.max(dim=1)
            min_bee_costs, _ = bee_costs.min(dim=1)
            spread = max_bee_costs - min_bee_costs
            # avoid division by 0
            spread[spread == 0] = 1

            norm_costs = (max_bee_costs[:, None] - bee_costs) / spread[:, None]
            min_norm_costs, _ = norm_costs.min(1)
            follow_probs = (-norm_costs + min_norm_costs[:, None]).exp()
            are_recruiters = follow_probs < torch.rand(n_bees, device=dev)

            if not are_recruiters.any():
                # no recruiters, so make everyone keep their own scenario.
                continue

            # decide which recruiter each follower will follow
            denom = (norm_costs * are_recruiters).sum(1)
            # avoid division by 0
            denom[denom == 0] = 1
            recruit_probs = norm_costs / denom[:, None]
            recruit_probs[~are_recruiters] = 0
            # set probs where there is no probability to 1, so multinomial
             # doesn't complain.
            no_valid_recruiters = recruit_probs.sum(-1) == 0
            recruit_probs[no_valid_recruiters] = 1
            recruiters = recruit_probs.multinomial(n_bees, replacement=True)
            recruiters[no_valid_recruiters] = bee_idxs
            recruiters[are_recruiters] = \
                bee_idxs[None].expand(batch_size, -1)[are_recruiters]

            # get the scenarios that the followers will follow
            bee_scenarios = bee_scenarios[batch_idxs[:, None], recruiters]

        # update the best solution found so far
        current_best_cost, current_best_idx = bee_costs.min(1)
        is_improvement = current_best_cost < best_costs
        best_costs[is_improvement] = current_best_cost[is_improvement]
        best_scenarios[is_improvement] = \
            bee_scenarios[is_improvement, current_best_idx[is_improvement]]
        cost_history[:, iteration + 1] = best_costs

        if sum_writer is not None:
            sum_writer.add_scalar('mean cost', best_costs.mean(), iteration)

    # return the best solution
    return best_scenarios, cost_history


def get_direct_sat_dmd(stop_dmd, shortest_paths, symmetric):
    direct_sat_dmd = torch.zeros_like(stop_dmd)
    summed_demands = aggregate_dense_conns(shortest_paths, stop_dmd[..., None])
    if symmetric:
        summed_demands += \
            aggregate_dense_conns(shortest_paths.transpose(1,2),
                                  stop_dmd.transpose(1,2)[..., None])
    direct_sat_dmd[:, :-1, :-1] = summed_demands.squeeze(-1)

    # assumes there's a dummy column and row
    direct_sat_dmd[:, -1] = 0
    direct_sat_dmd[:, :, -1] = 0
    return direct_sat_dmd


def build_init_scenario(shortest_paths, demand, n_routes, symmetric_routes):
    dev = shortest_paths.device
    batch_size = shortest_paths.shape[0]
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = shortest_paths.shape[1]
    dm_uncovered = demand.clone()
    best_scenarios = torch.full((batch_size, n_routes, max_n_nodes), -1, 
                                device=dev)
    log.info('computing initial scenario')
    # stop-to-itself routes are all invalid
    terms_are_invalid = torch.eye(max_n_nodes + 1, device=dev, dtype=bool)
    # routes to and from the dummy stop are invalid
    terms_are_invalid[max_n_nodes, :] = True
    terms_are_invalid[:, max_n_nodes] = True
    terms_are_invalid = terms_are_invalid[None].repeat(batch_size, 1, 1)
    for ri in range(n_routes):
        # compute directly-satisfied-demand matrix
        direct_sat_dmd = get_direct_sat_dmd(dm_uncovered, shortest_paths,
                                            symmetric_routes)
        # set invalid term pairs to -1 so they won't be selected even if there
         # is no uncovered demand.
        direct_sat_dmd[terms_are_invalid] = -1

        # choose maximum DS route and add it to the initial scenario
        flat_dsd = direct_sat_dmd.flatten(1, 2)
        _, best_flat_idxs = flat_dsd.max(dim=1)
        # add 1 to account for the dummy column and row
        best_i = torch.div(best_flat_idxs, (max_n_nodes + 1), 
                           rounding_mode='floor')
        best_j = best_flat_idxs % (max_n_nodes + 1)
        # batch_size x route_len
        routes = shortest_paths[batch_idxs, best_i, best_j]
        best_scenarios[:, ri, :routes.shape[-1]] = routes

        # mark new routes as in use
        terms_are_invalid[batch_idxs, best_i, best_j] = True
        if symmetric_routes:
            terms_are_invalid[batch_idxs, best_j, best_i] = True

        # remove newly-covered demand from uncovered demand matrix
        for ii in range(routes.shape[-1] - 1):
            cur_stops = routes[:, ii][:, None]
            later_stops = routes[:, ii+1:]
            dm_uncovered[batch_idxs[:, None], cur_stops, later_stops] = 0
            if symmetric_routes:
                maxidx = routes.shape[-1] - 1
                cur_stops = routes[:, maxidx - ii]
                later_stops = routes[:, maxidx - ii - 1]
                dm_uncovered[batch_idxs[:, None], later_stops, cur_stops] = 0

    return best_scenarios


def get_2typebee_variants(bee_scenarios, chosen_route_idxs, n_type1, 
                          direct_sat_dmd, shorten_prob, street_node_neighbours,
                          shortest_paths, 
                          bee_model=None, env_states=None
                          ):
    bee_scenarios = bee_scenarios.clone()
    # flatten batch and bee dimensions
    gather_idx = chosen_route_idxs[..., None, None]
    max_n_nodes = bee_scenarios.shape[3]
    gather_idx = gather_idx.expand(-1, -1, -1, max_n_nodes)
    modified_routes = bee_scenarios.gather(2, gather_idx).squeeze(2)
    # modify type 1 routes
    if bee_model is not None:
        # run it on all bees...
        new_type1_scenarios = get_neural_variants(bee_model, env_states, 
                                                  bee_scenarios,
                                                  chosen_route_idxs)
        # ...and keep only the type 1 bee routes
        new_type1_routes = new_type1_scenarios[:, :n_type1, -1]
    else:
        new_type1_routes = get_bee_1_variants(modified_routes[:, :n_type1], 
                                              direct_sat_dmd, shortest_paths)

    # modify type 2 routes
    new_type2_routes = get_bee_2_variants(modified_routes[:, n_type1:], 
                                          shorten_prob, street_node_neighbours)

    assert ((new_type2_routes > -1).sum(dim=-1) > 0).all()

    # insert the modified routes in the new scenario
    new_routes = torch.cat((new_type1_routes, new_type2_routes), 
                            dim=1)
    bee_scenarios.scatter_(2, gather_idx, new_routes[..., None, :])
    return bee_scenarios


def get_neural_variants(model, env_state, bee_scenarios, drop_route_idxs):
    # select routes to drop in the same way as above
    # flatten batch and bee dimensions
    batch_size = bee_scenarios.shape[0]
    n_bees = bee_scenarios.shape[1]
    n_routes = bee_scenarios.shape[2]
    keep_mask = torch.ones(bee_scenarios.shape[:3], dtype=bool, 
                           device=bee_scenarios.device)
    keep_mask.scatter_(2, drop_route_idxs[..., None], False)
    flat_kept_routes = bee_scenarios[keep_mask]
    # this works because we remove the same # of routes from each scenario
    max_n_nodes = bee_scenarios.shape[3]
    flatbee_kept_routes = flat_kept_routes.reshape(
        batch_size * n_bees, n_routes - 1, max_n_nodes)

    # plan a new route with the model
    env_state.clear_routes()
    env_state.add_new_routes(flatbee_kept_routes)
    state, _, _ = model.step(env_state, greedy=False)
    routes_as_lists = state.routes
    routes = get_batch_tensor_from_routes(routes_as_lists, 
                                          bee_scenarios.device)
    routes = routes.reshape(batch_size, n_bees, n_routes, -1)
    pad_size = max_n_nodes - routes.shape[-1]
    routes = torch.nn.functional.pad(routes, (0, pad_size), value=-1)
    return routes


def get_bee_1_variants(batch_bee_routes, direct_sat_dmd_mat, shortest_paths):
    """
    batch_bee_routes: a batch_size x n_bees x n_nodes tensor of routes
    direct_sat_dmd_mat: a batch_size x n_nodes x n_nodes tensor of 
        directly-satisfied demand by the shortest-path route between each pair
        of nodes.
    shortest_paths: a batch_size x n_nodes x n_nodes tensor of the shortest
        paths between each pair of nodes.
    """
    # choose which terminal to keep
    dev = batch_bee_routes.device
    keep_start_term = torch.rand(batch_bee_routes.shape[:2], device=dev) > 0.5

    # choose the new terminal
    # first, compute the demand that would be satisfied by new terminals
    route_lens = (batch_bee_routes > -1).sum(-1)
    batch_idxs = torch.arange(batch_bee_routes.shape[0]).unsqueeze(1)
    route_starts = batch_bee_routes[:, :, 0]
    dsd_from_starts = direct_sat_dmd_mat[batch_idxs, route_starts]
    route_ends = batch_bee_routes.gather(2, route_lens[..., None] - 1)
    route_ends.squeeze_(-1)
    dsd_to_ends = direct_sat_dmd_mat[batch_idxs, :, route_ends]
    # batch_size x n_bees x n_nodes
    new_route_dsds = keep_start_term[..., None] * dsd_from_starts + \
                     ~keep_start_term[..., None] * dsd_to_ends
    
    # second, choose the new terminal proportional to the demand
    # if no demand is satisfiable, set all probs to non-zero for multinomial()
    no_demand_satisfied = new_route_dsds.sum(-1) == 0
    new_route_dsds[no_demand_satisfied] = 1
    # set the terminal that is already in the route to 0, so it can't be chosen
    kept_terms = keep_start_term * route_starts + ~keep_start_term * route_ends
    new_route_dsds.scatter_(2, kept_terms[..., None], 0)
    # sample the terminals
    flat_new_terms = new_route_dsds.flatten(0,1).multinomial(1).squeeze(-1)    
    new_terms = flat_new_terms.reshape(batch_bee_routes.shape[:2])

    # if all demand is zero, dummy might get chosen.  If so, leave route alone.
    dummy_node = direct_sat_dmd_mat.shape[-1] - 1
    chose_dummy = new_terms == dummy_node
    new_terms[chose_dummy] = (~keep_start_term * route_starts + \
                              keep_start_term * route_ends)[chose_dummy]

    new_starts = keep_start_term * route_starts + ~keep_start_term * new_terms
    new_ends = ~keep_start_term * route_ends + keep_start_term * new_terms
    new_routes = shortest_paths[batch_idxs, new_starts, new_ends]

    # pad the end of the new routes to match the existing ones
    n_pad_stops = batch_bee_routes.shape[-1] - new_routes.shape[-1]
    new_routes = torch.nn.functional.pad(new_routes, (0, n_pad_stops), 
                                         value=-1)
    assert ((new_routes > -1).sum(dim=-1) > 0).all()

    return new_routes


def get_bee_2_variants(batch_bee_routes, shorten_prob, are_neighbours):
    """
    batch_bee_routes: a batch_size x n_bees x n_nodes tensor of routes
    shorten_prob: a scalar probability of shortening each route
    are_neighbours: a batch_size x n_nodes x n_nodes boolean tensor of whether
        each node is a neighbour of each other node
        
    """
    # flatten the batch and bee dimensions to ease what follows
    flat_routes = batch_bee_routes.flatten(0,1)
    # expand and reshape are_neighbours to match flat_routes
    n_bees = batch_bee_routes.shape[1]
    are_neighbours = are_neighbours[:, None].expand(-1, n_bees, -1, -1)
    are_neighbours = are_neighbours.flatten(0,1)
    # convert from boolean to float to allow use of torch.multinomial()
    neighbour_probs = are_neighbours.to(dtype=torch.float32)
    # add a padding column for scattering
    neighbour_probs = torch.nn.functional.pad(neighbour_probs, (0, 1, 0, 1))

    dev = batch_bee_routes.device
    keep_start_term = torch.rand(flat_routes.shape[0], device=dev) > 0.5
    keep_start_term.unsqueeze_(-1)

    route_lens = (flat_routes > -1).sum(-1)[..., None]

    # shorten chosen routes at chosen end
    shortened_at_start = flat_routes.roll(shifts=-1, dims=-1)
    shortened_at_start[:, -1] = -1
    shortened_at_end = flat_routes.scatter(1, route_lens - 1, -1)
    shortened = keep_start_term * shortened_at_start + \
        ~keep_start_term * shortened_at_end  

    # extend chosen routes at chosen ends
    n_nodes = batch_bee_routes.shape[-1]
    # choose new extended start nodes
    route_starts = flat_routes[:, 0]
    rs_gatherer = route_starts[:, None, None].expand(-1, n_nodes+1, 1)
    start_nbr_probs = neighbour_probs.gather(2, rs_gatherer).squeeze(-1)
    bbr_scatterer = get_update_at_mask(flat_routes, flat_routes==-1, n_nodes)
    start_nbr_probs.scatter_(1, bbr_scatterer, 0)
    no_start_options = start_nbr_probs.sum(-1) == 0
    start_nbr_probs[no_start_options] = 1
    chosen_start_exts = start_nbr_probs.multinomial(1).squeeze(-1)
    chosen_start_exts[no_start_options] = -1
    extended_starts = flat_routes.roll(shifts=1, dims=-1)
    extended_starts[:, 0] = chosen_start_exts

    # choose new extended end nodes
    route_ends = flat_routes.gather(1, route_lens - 1)
    re_gatherer = route_ends[:, None].repeat(1, 1, n_nodes+1)
    re_gatherer[re_gatherer == -1] = n_nodes
    end_nbr_probs = neighbour_probs.gather(1, re_gatherer).squeeze(-2)
    end_nbr_probs.scatter_(1, bbr_scatterer, 0)
    no_end_options = end_nbr_probs.sum(-1) == 0
    end_nbr_probs[no_end_options] = 1
    chosen_end_exts = end_nbr_probs.multinomial(1)
    chosen_end_exts[no_end_options] = -1
    # pad the routes before scattering, so that full-length routes don't cause
     # an index-out-of-bounds error.
    extended_ends = torch.nn.functional.pad(flat_routes, (0, 1))
    extended_ends = extended_ends.scatter(1, route_lens, chosen_end_exts)
    extended_ends = extended_ends[..., :-1]

    extended_routes = extended_ends * keep_start_term + \
        extended_starts * ~keep_start_term

    # assemble the shortened routes
    shorten = shorten_prob > torch.rand(flat_routes.shape[0], device=dev)
    shorten &= route_lens[..., 0] > 2
    shorten.unsqueeze_(-1)
    shortened_part = shortened * shorten
    # assemble the extended routes
    extend_at_start = ~no_start_options[..., None] & ~keep_start_term
    extend_at_end = ~no_end_options[..., None] & keep_start_term
    extend = (extend_at_start | extend_at_end) & ~shorten
    extended_part = extended_routes * extend
    # assemble the unmodified routes (ones with no valid extension)
    keep_same = ~(extend | shorten)
    same_part = flat_routes * keep_same
    # combine the three
    out_routes = shortened_part + extended_part + same_part
    
    out_lens = (out_routes > -1).sum(-1)
    assert ((out_lens - route_lens[..., 0]).abs() <= 1).all()

    # fold back into batch x bees
    out_routes = out_routes.reshape(batch_bee_routes.shape)
    return out_routes


@hydra.main(version_base=None, config_path="../cfg", config_name="bco_mumford")
def main(cfg: DictConfig):
    global DEVICE
    use_neural_bees = cfg.get('neural_bees', False)
    if use_neural_bees:
        prefix = 'neural_bco_'
    else:
        prefix = 'bco_'

    DEVICE, run_name, sum_writer, cost_fn, model = \
        lrnu.process_standard_experiment_cfg(cfg, prefix, 
                                             weights_required=True)

    # read in the dataset
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    bbs = cfg.get('bee_batch_size', None)
    if model is not None:
        model.eval()

    bee_model = None
    if use_neural_bees:
        bee_model = model

    init_model = None
    if cfg.get('init_from_model', False) and model is not None:
        init_model = model

    draw = cfg.eval.get('draw', False)

    all_route_sets = []
    nt1b = cfg.get('n_type1_bees', cfg.n_bees // 2)
    for n_routes in cfg.eval.n_routes:
        routes = \
            lrnu.test_method(bee_colony, test_dl, n_routes, 
                cfg.eval.min_route_len, cfg.eval.max_route_len, cost_fn, 
                sum_writer=sum_writer, silent=False, cost_batch_size=bbs, 
                n_bees=cfg.n_bees, n_iters=cfg.n_iterations, csv=cfg.eval.csv, 
                n_type_1_bees=nt1b, init_model=init_model, device=DEVICE, 
                symmetric_routes=cfg.experiment.symmetric_routes,
                bee_model=bee_model, draw=draw, return_routes=True)[-1]
        if type(routes) is not torch.Tensor:
            routes = get_batch_tensor_from_routes(routes)        
        all_route_sets.append(routes)
    
    # save the final routes that were produced
    folder = Path('output_routes')
    if not folder.exists():
        folder.mkdir()
    out_path = folder / (run_name + '_routes.pkl')
    with out_path.open('wb') as ff:
        pickle.dump(zip(cfg.eval.n_routes, all_route_sets), ff)


if __name__ == "__main__":
    main()
