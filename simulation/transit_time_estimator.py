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

import torch
import networkx as nx
from dataclasses import dataclass

from torch_utils import floyd_warshall, reconstruct_all_paths, \
    get_batch_tensor_from_routes, get_route_edge_matrix, get_route_leg_times


MEAN_STOP_TIME_S = 60
AVG_TRANSFER_WAIT_TIME_S = 300
UNSAT_PENALTY_EXTRA_S = 3000


def enforce_correct_batch(matrix, batch_size):
    if matrix.ndim == 2:
        matrix = matrix[None]
    if matrix.shape[0] > 1:
        assert batch_size == matrix.shape[0]
    elif batch_size > 1:
        shape = (batch_size,) + (-1,) * (matrix.ndim - 1)
        matrix = matrix.expand(*shape)
    return matrix

@dataclass
class CostHelperOutput:
    total_demand_time: torch.Tensor
    total_route_time: torch.Tensor
    trips_at_transfers: torch.Tensor
    total_demand: torch.Tensor
    unserved_demand: torch.Tensor
    total_transfers: torch.Tensor
    trip_times: torch.Tensor
    n_disconnected_demand_edges: torch.Tensor
    n_stops_oob: torch.Tensor
    n_nodes: torch.Tensor
    batch_routes: torch.Tensor


class CostModule(torch.nn.Module):
    def __init__(self, 
                 mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True
                 ):
        super().__init__()
        self.mean_stop_time_s = mean_stop_time_s
        self.avg_transfer_wait_time_s = avg_transfer_wait_time_s
        self.symmetric_routes = symmetric_routes

    def _cost_helper(self, routes, graph_data):
        """
        symmetric_routes: if True, treat routes as going both ways along their
            stops.
        """
        log.debug("formatting tensors")
        drive_times_matrix = graph_data.drive_times.to(float)
        demand_matrix = graph_data.demand.to(float)
        dev = drive_times_matrix.device

        # check if routes have a batch
        if type(routes) is torch.Tensor:
            routes_have_batch = routes.ndim > 2
        else:
            routes_have_batch = type(routes[0][0]) in [list, torch.Tensor]
        if routes_have_batch:
            batch_size = len(routes)
            drive_times_matrix = \
                enforce_correct_batch(drive_times_matrix, batch_size)
            demand_matrix = \
                enforce_correct_batch(demand_matrix, batch_size)
            
        else:
            batch_size = 1
            if drive_times_matrix.ndim == 2:
                drive_times_matrix = drive_times_matrix[None]
            if demand_matrix.ndim == 2:
                demand_matrix = demand_matrix[None]

        if not routes_have_batch:
            batch_routes = [routes] * batch_size
        else:
            batch_routes = routes

        log.debug("assembling route edge matrices")
        # assemble route graph
        if type(batch_routes) is not torch.Tensor:
            batch_routes = get_batch_tensor_from_routes(batch_routes, dev)

        route_edge_mat = \
            get_route_edge_matrix(batch_routes, drive_times_matrix,
                                  self.mean_stop_time_s,
                                  self.symmetric_routes)

        # run Floyd Warshall through the routing graph with length limit
        log.debug("running fw")
        _, nexts, trip_times = \
            floyd_warshall(route_edge_mat, return_raw_tensors=True)
        
        log.debug("computing path lengths")
        # calculate path lengths and routes and stops used 

        _, path_lens = reconstruct_all_paths(nexts)
        # number of transfers is number of nodes except start and end
        path_n_transfers = path_lens - 2
        path_n_transfers[path_lens == 0] = 0
        
        log.debug("summing cost components")

        # add transfer times
        trip_times += path_n_transfers * self.avg_transfer_wait_time_s    
        # setting infs to 0 will prevent locs without paths from contributing
        nopath = trip_times.isinf()
        # check if route graphs are connected 
        needed_path_missing = nopath & (demand_matrix > 0)
        n_disconnected_demand_edges = needed_path_missing.sum(dim=(1, 2))

        route_lens = (batch_routes > -1).sum(-1)
        n_routes = (route_lens > 0).sum(-1)
        zero = torch.zeros_like(route_lens)
        route_len_delta = (self.min_route_len - route_lens).maximum(zero)
        # don't penalize placeholer "dummy" routes in the tensor
        route_len_delta[route_lens == 0] = 0
        if self.max_route_len is not None:
            route_len_over = (route_lens - self.max_route_len).maximum(zero)
            route_len_delta = route_len_delta + route_len_over
        n_stops_oob = route_len_delta.sum(-1)

        trip_times[nopath] = 0
        demand_time = demand_matrix * trip_times
        total_dmd_time = demand_time.sum(dim=(1, 2))
        demand_transfers = demand_matrix * path_n_transfers
        trips_at_transfers = torch.zeros(batch_size, 4, device=dev)
        for ii in range(3):
            d_i = (demand_matrix * (path_n_transfers == ii)).sum(dim=(1, 2))
            trips_at_transfers[:, ii] = d_i
        
        d_un = (demand_matrix * (path_n_transfers > 2)).sum(dim=(1, 2))
        d_un += (demand_matrix * nopath).sum(dim=(1, 2))
        trips_at_transfers[:, 3] = d_un

        total_transfers = demand_transfers.sum(dim=(1, 2))
        unserved_demand = (demand_matrix * nopath).sum(dim=(1, 2))
        total_demand = demand_matrix.sum(dim=(1,2))

        # compute total route times
        # add zero-padding so dummy values of -1 don't affect the sum
        leg_times = get_route_leg_times(batch_routes, drive_times_matrix,
                                        self.mean_stop_time_s)
        total_route_time = leg_times.sum(dim=(1, 2))

        if self.symmetric_routes:
            transpose_dtm = drive_times_matrix.transpose(1, 2)
            return_leg_times = get_route_leg_times(batch_routes, transpose_dtm,
                                                   self.mean_stop_time_s)
            total_route_time += return_leg_times.sum(dim=(1, 2))

        n_nodes = [dd.num_nodes for dd in graph_data.to_data_list()]
        n_nodes = torch.tensor(n_nodes, device=dev)

        output = CostHelperOutput(
            total_dmd_time, total_route_time, trips_at_transfers, \
            total_demand, unserved_demand, total_transfers, trip_times,
            n_disconnected_demand_edges, n_stops_oob, n_nodes, batch_routes
        )
        
        return output
    
    @staticmethod
    def _compute_mean_demand_time(costs_helper_output):
        served_demand = costs_helper_output.total_demand - \
            costs_helper_output.unserved_demand
        # avoid division by 0
        return costs_helper_output.total_demand_time / (served_demand + 1e-6)
        
    def get_metrics(self, routes, graph_data):
        output = self._cost_helper(routes, graph_data)
        avg_dmd_time = self._compute_mean_demand_time(output)
        avg_dmd_time_minutes = avg_dmd_time / 60
        route_time_minutes = output.total_route_time / 60
        # average trip time, total route time, trips-at-n-transfers, and 
         # whether all demand is connected
        tat = output.trips_at_transfers
        percent_tat = tat * 100 / output.total_demand[:, None]
        return avg_dmd_time_minutes, route_time_minutes, \
            percent_tat, output.n_disconnected_demand_edges, output.n_stops_oob


class MyCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 min_route_len=2, max_route_len=None,
                 symmetric_routes=True, demand_time_weight=0.5, 
                 route_time_weight=0.5, unserved_weight=5,
                 variable_weights=False):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes)
        self.demand_time_weight = demand_time_weight
        self.min_route_len = min_route_len
        self.max_route_len = max_route_len
        self.route_time_weight = route_time_weight
        self.unserved_weight = unserved_weight
        self.variable_weights = variable_weights

    def sample_variable_weights(self, batch_size, device=None):
        if not self.variable_weights:
            dtm = torch.full((batch_size,), self.demand_time_weight, 
                             device=device)
            rtm = torch.full((batch_size,), self.route_time_weight, 
                              device=device)
        else:
            # multiply by 1.3 and then clamp to 0, 1 so that we sometimes get
            # all-the-way 0 or 1.
            dtm = torch.rand(batch_size, device=device) * 1.3
            dtm = dtm.clamp(0, 1) 
            rtm = 1 - dtm
 
        return {
            'demand_time_weight': dtm,
            'route_time_weight': rtm
        }
    
    def get_weights(self, device=None):
        dtm = self.demand_time_weight
        if type(dtm) is not torch.Tensor:
            dtm = torch.tensor([dtm], device=device)
        rtm = self.route_time_weight
        if type(rtm) is not torch.Tensor:
            rtm = torch.tensor([rtm], device=device)
        
        return {
            'demand_time_weight': dtm,
            'route_time_weight': rtm
        }
    
    def set_weights(self, demand_time_weight=None, route_time_weight=None, 
                    unserved_weight=None):
        if demand_time_weight is not None:
            self.demand_time_weight = demand_time_weight
        if route_time_weight is not None:
            self.route_time_weight = route_time_weight
        if unserved_weight is not None:
            self.unserved_weight = unserved_weight

    def forward(self, routes, graph_data, demand_time_weight=None, 
                route_time_weight=None, unserved_weight=None,
                no_norm=False):
        cho = self._cost_helper(routes, graph_data)
        if demand_time_weight is None:
            demand_time_weight = self.demand_time_weight
        if route_time_weight is None:
            route_time_weight = self.route_time_weight
        if unserved_weight is None:
            unserved_weight = self.unserved_weight

        # if we have more weights than routes, truncate the weights
        batch_size = graph_data.num_graphs
        if type(demand_time_weight) is torch.Tensor and \
           demand_time_weight.shape[0] > batch_size:
            demand_time_weight = demand_time_weight[:batch_size]
        if type(route_time_weight) is torch.Tensor and \
           route_time_weight.shape[0] > batch_size:
            route_time_weight = route_time_weight[:batch_size]
        if type(unserved_weight) is torch.Tensor and \
           unserved_weight.shape[0] > batch_size:
            unserved_weight = unserved_weight[:batch_size]

        # normalize all time values by the maximum drive time in the graph
        time_normalizer = graph_data.drive_times.max(-1)[0].max(-1)[0]
        avg_dmd_time = self._compute_mean_demand_time(cho)
        norm_dmd_time = avg_dmd_time / time_normalizer

        # normalize average route time
        routes = cho.batch_routes
        route_lens = (routes > -1).sum(-1)
        n_routes = (route_lens > 0).sum(-1)
        norm_route_time = \
            cho.total_route_time / (3 * time_normalizer * n_routes + 1e-6)

        frac_uncovered = cho.n_disconnected_demand_edges / (cho.n_nodes ** 2)
        if self.max_route_len is None:
            denom = n_routes * self.min_route_len
        else:
            denom = n_routes * self.max_route_len
        rld_frac = cho.n_stops_oob / denom
        constraints_violated = (rld_frac > 0) | (frac_uncovered > 0)
        cv_penalty = 0.1 * constraints_violated

        # average trip time, total route time, and trips-at-n-transfers
        if no_norm:
            cost = avg_dmd_time * demand_time_weight + \
                cho.total_route_time * route_time_weight
        else:
            cost = norm_dmd_time * demand_time_weight + \
                norm_route_time * route_time_weight

        cost += (cv_penalty + frac_uncovered + rld_frac) * unserved_weight
        
        return cost


class NikolicCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True,
                 unsatisfied_penalty_extra_s=UNSAT_PENALTY_EXTRA_S, 
                 ):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes)
        self.unsatisfied_penalty_extra_s = unsatisfied_penalty_extra_s
        self.min_route_len = 2
        self.max_route_len = None

    def forward(self, routes, graph_data):
        """
        symmetric_routes: if True, treat routes as going both ways along their
            stops.
        """
        cho = self._cost_helper(routes, graph_data)
        # Note that unlike Nikolic, we count trips that take >2 transfers as 
         # satisfied.
        tot_sat_demand = cho.total_demand - cho.unserved_demand
        w_2 = cho.total_demand_time / tot_sat_demand
        no_sat_dmd = torch.isclose(tot_sat_demand, 
                                   torch.zeros_like(tot_sat_demand))
        # if no demand is satisfied, set w_2 to the average time of all trips plus
        # the penalty
        w_2[no_sat_dmd] = cho.trip_times[no_sat_dmd].mean(dim=(-2,-1))
        w_2 += self.unsatisfied_penalty_extra_s

        cost = cho.total_demand_time + w_2 * cho.unserved_demand

        assert not ((cost == 0) & (cho.total_demand > 0)).any()

        log.debug("finished nikolic")
        assert cost.isfinite().all(), "invalid cost was computed!"
        
        return cost

