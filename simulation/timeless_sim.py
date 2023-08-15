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

from itertools import product, cycle
from collections import namedtuple, defaultdict
import time
import logging as log
from pathlib import Path
import datetime
from dataclasses import replace
import copy

import numpy as np
import pyproj
import pandas as pd
from scipy.spatial.distance import cdist
import networkx as nx
import yaml
import torch
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import PIL.Image as Image
import folium
import geojson
from torch_geometric.data import Data

from world.transit import PtVehicleType, RouteRep
from config_utils import str_time_to_dt_time, str_time_to_float
from torch_utils import floyd_warshall, get_path_edge_index, square_pdist

# import the rust public transit sim package
import ptsim

EnvRep = namedtuple('EnvRep', ['stop_data', 'demand_data', 'basin_edges', 
                               'basin_weights', 'inter_node_demands'])

    
class TimelessSimulator:
    """This class is a wrapper on our Rust public transit simulator."""
    def __init__(self, config_path, per_stop_reward_fn=None, 
                 stops_path=None, mapframe="EPSG:3348"):
        """
        Arguments
        config_path: path to the simulator's yaml config file
        filter_nodes: if true, apply Bagloee filtering to the nodes, and only
            expose the remaining nodes to the user.
        per_stop_reward_fn: a function taking two arguments, a dict of per-stop
            statistics and a dict of global statistics, which returns a list
            of arrays giving per-stop rewards.
        stops_path: a path to a GTFS stops.txt file.  If provided, only stops
            in this file will be exposed to the user.
        """
        self.per_stop_reward_fn = per_stop_reward_fn
        self.mapframe = mapframe

        with open(config_path, "r") as ff:
            # TODO reading the config separately from the underlying sim is 
            # probably not a great idea...
            self.cfg = yaml.load(ff, Loader=yaml.Loader)

        self.sim = ptsim.StaticPtSim(config_path)
        self.stop_positions = torch.tensor(self.sim.get_stop_node_positions())
        dtm = self.sim.get_drive_times_matrix("road")
        self._drive_times_matrix = torch.tensor(dtm)
        ddm = self.sim.get_drive_dists_matrix("road")
        self._drive_dists_matrix = torch.tensor(ddm)
        # this should be fine because the networkx graph guarantees node order
        # is preserved in python 3.7+, and we're using python 3.8.  But beware
        # when running on different platforms!  Using our conda
        # environment.yaml should enforce this.

        demand_poss, demand_edges = self.sim.get_demand_graph()
        self.demand_graph = nx.DiGraph()
        demand_edges = [(ff, tt, {"weight": dd}) 
            for (ff, tt, dd) in demand_edges]
        self.demand_graph.add_nodes_from(demand_poss)
        demand_edges = [(demand_poss[ii], demand_poss[jj], ww) 
                         for (ii, jj, ww) in demand_edges]
        self.demand_graph.add_edges_from(demand_edges)

        self.street_graph = nx.MultiDiGraph()
        self.street_graph.add_nodes_from(range(self.get_num_stop_nodes()))
        self.street_graph.add_edges_from(self.sim.get_street_edges())

        self.basin_connections = self.sim.get_basin_connections()

        if stops_path:
            self._kept_nodes = \
                _get_stops_from_gtfs(stops_path, self.stop_positions)
        
        else:
            self._kept_nodes = torch.arange(self.street_graph.number_of_nodes())

        self.fixed_transit = self.read_fixed_routes()

    def filter_nodes(self, nodes_to_keep):
        # nodes_to_keep is a collection of indices of already-kept nodes
        self._kept_nodes = self._kept_nodes[nodes_to_keep]

    @property
    def n_street_nodes(self):
        return len(self._kept_nodes)

    @property
    def drive_dists(self):
        valid_rows = self._drive_dists_matrix[self._kept_nodes]
        valid_rows_and_cols = valid_rows[:, self._kept_nodes]
        return valid_rows_and_cols

    @property
    def drive_times(self):
        valid_rows = self._drive_times_matrix[self._kept_nodes]
        valid_rows_and_cols = valid_rows[:, self._kept_nodes]
        return valid_rows_and_cols


    @property
    def walk_dists(self):
        return self.crow_dists * self.cfg["beeline_dist_factor"]

    @property
    def crow_dists(self):
        valid_stop_poss = self.stop_positions[self._kept_nodes]
        return square_pdist(valid_stop_poss)

    @property
    def total_demand(self):
        return np.sum(nx.to_numpy_array(self.demand_graph))

    @property
    def basin_radius_m(self):
        return self.cfg["basin_radius_m"]

    @property
    def sim_duration_s(self):
        return self.sim.get_duration_s()

    def run(self, routes, frequencies_Hz, vehicles=None, 
            include_fixed_transit=True, capacity_free=False, 
            fast_demand_assignment=False):
        mapped_routes = self._map_routes(routes)
        if vehicles == None:
            default_vehicle = self.get_default_vehicle_type()
            vehicles = [default_vehicle for _ in routes]
        frequencies_Hz = list(frequencies_Hz)

        if capacity_free:
            # set seating capacity to total demand, so no bus is ever full
            free_cap = int(np.ceil(self.total_demand))
            vehicles = [replace(vv, seats=free_cap) for vv in vehicles]

        include_fixed_transit &= self.fixed_transit is not None
        if include_fixed_transit:
            fixed_routes, fixed_freqs, fixed_vehs = \
                zip(*self.fixed_transit.values())
            mapped_routes = mapped_routes + list(fixed_routes)
            frequencies_Hz = frequencies_Hz + list(fixed_freqs)
            vehicles = vehicles + list(fixed_vehs)

        per_stop_info, global_info = \
            self.sim.run(mapped_routes, frequencies_Hz, vehicles,
                         fast_demand_assignment)

        if self.per_stop_reward_fn:
            per_stop_rewards = self.per_stop_reward_fn(per_stop_info, 
                                                       global_info)

            per_stop_info["rewards"] = per_stop_rewards

        if include_fixed_transit:
            n_fixed = len(self.fixed_transit)
            for kk in per_stop_info:
                per_stop_info[kk] = per_stop_info[kk][:-n_fixed]

        # global_info["quality"] = global_info["satisfied demand"] - \
        #     weight * global_info["total kW used"]**2

        global_info["ridership and cost"] = global_info["satisfied demand"] - \
            self.cfg["cost_weight"] * global_info["total kW used"]

        return per_stop_info, global_info

    def _map_routes(self, routes):
        # map the routes
        return [[self._kept_nodes[si].item() for si in rr] for rr in routes]

    def get_street_edge_matrix(self):
        street_edges = self._drive_times_matrix.clone()
        is_not_edge = nx.to_numpy_array(self.street_graph) == 0
        street_edges[torch.tensor(is_not_edge)] = float('inf')
        for node_idx in range(self.street_graph.number_of_nodes()):
            if node_idx in self._kept_nodes:
                continue
            # modify the edge matrix to "bridge" this node
            outgoing = street_edges[node_idx]
            incoming = street_edges[:, node_idx]
            bridged_times = incoming[:, None] + outgoing[None]
            has_in = incoming < float('inf')
            has_out = outgoing < float('inf')
            bridged_mask = has_in[:, None] & has_out[None]
            new_edges = bridged_times[bridged_mask].min(
                street_edges[bridged_mask])
            street_edges[bridged_mask] = new_edges                
        
        reduced_street_edges = street_edges[self._kept_nodes]
        reduced_street_edges = reduced_street_edges[:, self._kept_nodes]
        reduced_street_edges.fill_diagonal_(0)
        return reduced_street_edges

    def render_gtfs(self, gtfs_dir, service_id="AOUT13SEM", 
                    map_stops_to_nodes=True):
        route_ids, routes, _ = \
            self.translate_gtfs(gtfs_dir, service_id=service_id, 
                                map_stops_to_nodes=map_stops_to_nodes)
        if map_stops_to_nodes:
            self.render_plan(routes, route_names=route_ids,
                show_demand=False, show_node_labels=False, map_nodes=False)
        else:
            self.render_plan([])
            colours = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
            for route_id, route in zip(route_ids, routes):
                route_graph = nx.DiGraph()
                poss = {}
                for ii, stop_loc in enumerate(route):
                    route_graph.add_node(ii)
                    if ii > 0:
                        route_graph.add_edge(ii - 1, ii)
                    poss[ii] = stop_loc
                
                colour = next(colours)
                edge_col = nx.draw_networkx_edges(route_graph, pos=poss, 
                                                  edge_color=colour, 
                                                  style="dotted", width=3,
                                                  node_size=0)
                edge_col[0].set_label(route_id)
        plt.legend()

    def translate_gtfs(self, gtfs_dir, kept_nodes_only=False, 
                       map_stops_to_nodes=True, service_id="AOUT13SEM"):
        start_time = \
            str_time_to_dt_time(self.cfg["dataset"]["time_range_start"])
        end_time = str_time_to_dt_time(self.cfg["dataset"]["time_range_end"])
        # epsg 4326 corresponds to lat,lon
        transformer = pyproj.Transformer.from_crs("EPSG:4326", self.mapframe)
        if kept_nodes_only:
            stop_poss = self.stop_positions[self._kept_nodes]
        else:
            stop_poss = self.stop_positions

        routes_ids, routes, freqs = gtfs_to_timeless_transit(
            gtfs_dir, stop_poss, transformer.transform, service_id, start_time, 
            end_time, map_stops_to_nodes)

        # avoid duplicating routes that are in the fixed transit
        filtered = []
        for route_ids, route, freq in zip(routes_ids, routes, freqs):
            if not any([id in self.fixed_transit for id in route_ids]):
                filtered.append((route_ids, route, freq))
                
        return (list(ll) for ll in zip(*filtered))

    def read_fixed_routes(self):
        if "dataset" not in self.cfg or \
           "fixed_transit" not in self.cfg["dataset"]:
            return None
        fixed_routes_yaml_path = self.cfg["dataset"]["fixed_transit"]
        with open(fixed_routes_yaml_path) as ff:
            fixed_routes_yaml = yaml.load(ff, Loader=yaml.Loader)

        fixed_routes = {}
        for route_name, route_data in fixed_routes_yaml.items():
            period_s = str_time_to_float(route_data["trip_period"])
            frequency = 1 / period_s
            # map from node ids to node indexes
            route = [self.sim.get_node_index_by_id(str(sd["node_id"]))
                     for sd in route_data["stops"]]
            vehicle = PtVehicleType(**route_data["vehicle"])
            fixed_routes[route_name] = (route, frequency, vehicle)
            
        return fixed_routes

    def get_node_id_by_index(self, node_index):
        return self.sim.get_node_id_by_index(node_index)

    def get_default_vehicle_type(self):
        return PtVehicleType(**self.cfg['vehicle_type'])

    def get_num_stop_nodes(self):
        return len(self.stop_positions)

    def capacity_to_frequency(self, capacity, vehicle_type=None):
        """Return the frequency needed to acheive the capacity on the route."""
        # determine the number of vehicles assigned to the route
        if vehicle_type is None:
            vehicle_type = self.get_default_vehicle_type()
        if type(capacity) is list:
            # make it a numpy array
            capacity = np.array(capacity)
        n_traversals = capacity / vehicle_type.get_capacity()
        # determine how long it takes to run the route
        frequency_Hz = n_traversals / self.sim_duration_s
        return frequency_Hz

    def frequency_to_capacity(self, frequency_Hz, vehicle_type=None):
        if vehicle_type is None:
            vehicle_type = self.get_default_vehicle_type()
        # this assumes that each traversal is made by a separate vehicle
        n_traversals = frequency_Hz * self.sim_duration_s
        capacity = vehicle_type.get_capacity() * n_traversals
        return capacity

    def get_env_rep_for_nn(self, normalize=True, device=None,
                           adjacency_kernel=None, one_hot_node_feats=False):
        stop_features = torch.zeros((self.stop_positions.shape[0], 5),
                                    device=device, dtype=torch.float32)
        stop_features[:, :2] = self.stop_positions
        if one_hot_node_feats:
            oh_features = torch.eye(self.get_num_stop_nodes(), device=device)
            stop_features = torch.cat((stop_features, oh_features), dim=-1)

        if adjacency_kernel is None:
            # adjacency of location to itself = 1
            # adjacency at one-minute driving distance = 0.5
            drive_time_to_halve = 60
            adjacency_kernel = lambda dd: 2 ** (-dd / drive_time_to_halve)

        stop_adj_matrix = self._drive_times_matrix.clone()
        # threshold adjacencies that are neither directly connected in the 
        # street graph nor withing the adjacency threshold
        connected_by_street_mat = nx.to_numpy_array(self.street_graph)
        connected_by_street_mat = torch.tensor(connected_by_street_mat, 
                                               dtype=bool)
        threshold_locs = torch.logical_and(
            self._drive_times_matrix > self.cfg["adjacency_threshold_s"], 
            connected_by_street_mat.logical_not())
        stop_adj_matrix[threshold_locs] = 0
        # remove self-connections
        stop_adj_matrix.fill_diagonal_(0)
        stop_adj_matrix = stop_adj_matrix.to(device)
        # add street in- and out-degree features
        stop_features[:, 2] = (stop_adj_matrix > 0).sum(dim=1)
        stop_features[:, 3] = (stop_adj_matrix > 0).sum(dim=0)

        demand_locs = torch.tensor(list(self.demand_graph.nodes), device=device)
        demand_mat = torch.tensor(nx.to_numpy_array(self.demand_graph), 
                                  device=device, dtype=torch.float32)

        demand_features = torch.zeros((len(self.demand_graph), 5), 
                                    device=device, dtype=torch.float32)
        demand_features[:, :2] = demand_locs
        for ii in range(len(self.demand_graph.nodes)):
            # outgoing demand
            demand_features[ii, 2] = demand_mat[ii, :].sum()
            # incoming demand
            demand_features[ii, 3] = demand_mat[:, ii].sum()
        if one_hot_node_feats:
            oh_features = torch.eye(len(self.demand_graph), device=device)
            demand_features = torch.cat((demand_features, oh_features), dim=-1)

        basin_mat = self._basin_walk_times_matrix(device)
        basin_adj_mat = basin_mat < float('inf')
        # add basin degree feature
        stop_features[:, 4] = basin_adj_mat.sum(dim=0)
        demand_features[:, 4] = basin_adj_mat.sum(dim=1)

        demand_time_mat = demand_mat.clone()
        demand_shortest_times = self.sim.get_demand_shortest_times()
        for src_idx, dst_idx, shortest_time in demand_shortest_times:
            if shortest_time < float('inf'):
                demand_time_mat[src_idx, dst_idx] = shortest_time

        stop_features = stop_features[self._kept_nodes]
        stop_adj_matrix = \
            stop_adj_matrix[self._kept_nodes][:, self._kept_nodes]
        basin_mat = basin_mat[:, self._kept_nodes]
        basin_adj_mat = basin_adj_mat[:, self._kept_nodes]

        stop_edge_idxs = torch.stack(torch.where(stop_adj_matrix))
        demand_edge_idxs = torch.stack(torch.where(demand_mat))
        basin_edge_idxs = torch.stack(torch.where(basin_adj_mat))

        demand_edge_feats = torch.stack((demand_mat, demand_time_mat), dim=-1)
        # demand_edge_feats = demand_mat
        demand_edge_feats = demand_edge_feats[demand_edge_idxs[0], 
                                              demand_edge_idxs[1]]
        demand_edge_feats = demand_edge_feats.to(dtype=torch.float32)                                              
        assert not demand_edge_feats.isnan().any()
        stop_edge_feats = stop_adj_matrix[stop_edge_idxs[0], 
                                          stop_edge_idxs[1]][:, None]
        stop_edge_feats = stop_edge_feats.to(dtype=torch.float32)                                          
        basin_weights = basin_mat[basin_adj_mat][:, None]
        basin_weights = basin_weights.to(dtype=torch.float32)

        if normalize:
            for tensor in [stop_features, demand_features, demand_edge_feats, 
                           stop_edge_feats, basin_weights]:
                normalize_features_inplace(tensor)

        assert not demand_edge_feats.isnan().any()

        # street_time_mat = self.get_street_edge_matrix().to(device)
        stop_data = Data(x=stop_features, edge_index=stop_edge_idxs,
                         edge_attr=stop_edge_feats,
                         pos=self.stop_positions.clone(),
                        #  street_time_matrix=street_time_mat
                         )
        
        demand_data = Data(x=demand_features, edge_index=demand_edge_idxs,
                           edge_attr=demand_edge_feats, pos=demand_locs)

        inter_node_demands = get_inter_stop_demand(basin_adj_mat, demand_mat)
        return EnvRep(stop_data=stop_data, demand_data=demand_data, 
                      basin_edges=basin_edge_idxs, basin_weights=basin_weights,
                      inter_node_demands=inter_node_demands)

    def get_route_reps_for_nn(self, routes, costs, dense_edges=True, 
                              normalize=True, device=None):
        all_route_reps = []
        all_routes_feats = []
        if device is None and type(costs) is torch.Tensor:
            device = costs.device
        if type(costs) is torch.Tensor:
            costs = costs.to(device=device)

        for idx, (route, cost) in enumerate(zip(routes, costs)):
            # assemble the features for each edge
            if type(route) is torch.Tensor:
                route = list(route.cpu().numpy())
            leg_times = self.sim.get_route_leg_times_s(route, "road")
            leg_times = torch.tensor(leg_times, device=device, 
                                     dtype=torch.float32)
            if dense_edges:
                # each stop has an edge to every stop after it on the route
                dense_leg_times = []
                for ii in range(len(leg_times)):
                    cum_times = leg_times[ii:].cumsum(-1)
                    dense_leg_times.append(cum_times)
                edge_times = torch.cat(dense_leg_times, dim=0)
            else:
                edge_times = leg_times
            
            cost_feat = torch.zeros(len(edge_times), device=device)
            edge_feats = torch.stack((edge_times, cost_feat), dim=1)
            all_routes_feats.append(edge_feats)

            # assemble the edge indices
            indices = get_path_edge_index(route, dense_edges).to(device)
            route_rep = RouteRep(route, cost, indices, edge_feats, idx)
            all_route_reps.append(route_rep)
            
        if normalize:
            all_routes_feats_cat = torch.cat(all_routes_feats, dim=0)
            mean_feat = all_routes_feats_cat.mean(dim=0)
            std_feat = all_routes_feats_cat.std(dim=0)
            for route_rep in all_route_reps:
                route_rep.edge_attr -= mean_feat
                route_rep.edge_attr /= std_feat
        
        # set the costs using this function
        all_route_reps = \
            RouteRep.get_updated_costs_collection(all_route_reps, costs, 
                                                  normalize=normalize)

        return all_route_reps

    def _basin_walk_times_matrix(self, device=None):
        basin_mat = torch.ones((len(self.demand_graph), 
                                len(self.street_graph)), dtype=torch.float32,
                                device=device) * float('inf')
        for dmd_idx, connections in self.basin_connections.items():
            for stop_idx, walk_time in connections.items():
                basin_mat[dmd_idx, stop_idx] = walk_time
        return basin_mat

    def get_basin_walk_times_matrix(self, device):
        # basin walk times matrix filtered by kept nodes
        raw_mat = self._basin_walk_times_matrix(device)
        return raw_mat[:, self._kept_nodes]

    def get_demand_matrix(self):
        return torch.tensor(nx.to_numpy_array(self.demand_graph), 
                            dtype=torch.float32)

    def get_all_shortest_paths(self, device=None):
        street_edge_tns = self.get_street_edge_matrix().to(device)
        return floyd_warshall(street_edge_tns)
        
    def get_network_image(self, *args, **kwargs):
        # set up the figure and axis
        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        ax = fig.gca()
        ax.axis('off')
        fig.tight_layout(pad=0)
        ax.margins(0)

        self.render_plan(*args, ax=ax, **kwargs)

        # get the image array
        canvas.draw()
        buf = canvas.buffer_rgba()
        rgba = Image.fromarray(np.asarray(buf))
        # return a 3xHxW array.
        return np.array(rgba.convert('RGB')).transpose(2, 0, 1)

    def render_plan(self, routes, frequencies_Hz=None, route_names=None,
                    show_demand=False, show_node_labels=False, 
                    show_legend=False, arc=False, map_nodes=True, ax=None):
        if map_nodes:
            routes = self._map_routes(routes)

        if type(frequencies_Hz) is torch.Tensor:
            frequencies_Hz = frequencies_Hz.cpu().numpy()

        # draw the city
        stop_pos_dict = {ii: pos.numpy() 
                            for ii, pos in enumerate(self.stop_positions)}
        nx.draw(self.street_graph, with_labels=show_node_labels, ax=ax,
                pos=stop_pos_dict, node_size=0., arrows=False)
        # # render the kept nodes
        # nx.draw_networkx_nodes(self.street_graph, pos=stop_pos_dict, 
        #                        nodelist=[kn.item() for kn in self.kept_nodes],
        #                        ax=ax, node_color="orange", node_size=0.5)

        if show_demand:
            # scale widths by the fraction of max demand they contain
            de_widths = np.array([dd['weight'] for _, _, dd in 
                                    self.demand_graph.edges(data=True)])
            de_widths *= 2 / de_widths.max()
            de_pos_dict = {pos: pos for pos in self.demand_graph}
            nx.draw_networkx_edges(self.demand_graph, edge_color="red", 
                                   style="dotted", pos=de_pos_dict, 
                                   width=de_widths, ax=ax)

        # draw the routes on top of the city
        num_colours = max(3, int(np.ceil((2 * len(routes)) ** (1/3))))
        dim_points = np.linspace(0.1, 0.9, num_colours)
        # filter out grayscale colours
        colours = [cc for cc in product(dim_points, dim_points, dim_points)
                   if len(np.unique(cc)) > 1]
        colours = iter(colours)

        arc_fmt_str = "arc3,rad={}"
        if arc:
            arc_radius_stepsize = min(0.3 / (1 + len(routes)), 0.1)
        else:
            arc_radius_stepsize = 0

        if route_names is None:
            route_names = list(range(len(routes)))
        for rid, route in enumerate(routes):
            colour = next(colours)
            # make an nx graph of just the route's nodes, then draw edges
            # add edges first, to preserve the sequence.
            route_graph = nx.DiGraph()
            for stop_idx in route:
                route_graph.add_node(stop_idx)
            for ii in range(len(route) - 1):
                route_graph.add_edge(route[ii], route[ii + 1])
            
            route_poss = {si: self.stop_positions[si].numpy() for si in route}
            # TODO have variable arc parameters based on how many edges there
            # are already between each node pair
            arc_str = arc_fmt_str.format(arc_radius_stepsize * (1 + rid))
            width = 3
            if frequencies_Hz is not None:
                width = 1.5 * np.sqrt(frequencies_Hz[rid] * 3600)
            else:
                width = 3
            
            edge_col = nx.draw_networkx_edges(route_graph, pos=route_poss, 
                edge_color=colour, width=width, ax=ax,
                connectionstyle=arc_str, node_size=0)

            # make a label with the frequency
            route_name = str(route_names[rid])
            if frequencies_Hz is not None:
                freq = frequencies_Hz[rid] * 3600
                label = "{}: {:.1f} per hour".format(route_name, freq)
            else:
                label = route_name
            edge_col[0].set_label(label)
        
        if show_legend:
            if ax:
                ax.legend(loc='lower center')
            else:
                plt.legend(loc='lower center')

    def render_plan_on_html_map(self, routes, frequencies_Hz=None, 
                                map_nodes=True, outfile="routes.html"):
        if map_nodes:
            routes = self._map_routes(routes)

        # build a geojson file with the coordinates
        tfr = pyproj.Transformer.from_crs(self.mapframe, "EPSG:4326")
        lonlat_routes = []
        for ri, route in enumerate(routes):
            # map route node IDs to lonlats
            stop_pos_route = self.stop_positions[route]
            # reverse the order, since transform gives coords as (lat, lon)
            lonlat_route = geojson.LineString([tfr.transform(*sp)[::-1] 
                                               for sp in stop_pos_route])
            lonlat_routes.append(geojson.Feature(
                geometry=lonlat_route, 
                index=ri
            ))
        feat_collection = geojson.FeatureCollection(lonlat_routes)

        # build a style dict with the appropriate colours
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        def style_func(feature):
            route = routes[feature["index"]]
            colour = colours[sum(route) % len(colours)]
            style_dict = {"color": colour}
            if frequencies_Hz is not None:
                # make the line thickness proportional to the frequency
                freq_perhour = frequencies_Hz[feature["index"]] * 3600
                style_dict["weight"] = 2 * freq_perhour
            return style_dict
        
        # create the folium map
        # start loc is the mean node location
        center_loc = tfr.transform(*self.stop_positions.mean(dim=0))
        plan_map = folium.Map(location=center_loc, 
                              tiles="cartodbpositron",
                              zoom_start=12)
        # add the routes to the map
        folium.GeoJson(feat_collection, name="routes", 
                       style_function=style_func).add_to(plan_map)
        folium.LayerControl().add_to(plan_map)

        # save the map to the outfile
        plan_map.save(str(outfile))


def normalize_features_inplace(features):
    features -= features.mean(dim=0)
    stds = torch.std(features, dim=0)
    stds[stds == 0] = 1
    features /= stds
    

def _get_stops_from_gtfs(stops_path, node_locs, mapframe="EPSG:3348"):
    # epsg 4326 corresponds to lat,lon
    transformer = pyproj.Transformer.from_crs("EPSG:4326", mapframe)
    stops_df = pd.read_csv(stops_path)

    # combine stops that correspond to the same location
    pure_names_to_locs = defaultdict(list)
    for _, row in stops_df.iterrows():
        name_wo_code = row["stop_name"].partition("[")[0]
        pure_name = name_wo_code.partition(" Quai")[0]
        loc = (row["stop_lat"], row["stop_lon"])
        pure_names_to_locs[pure_name].append(loc)

    pure_names_to_locs = {nn: transformer.transform(*np.mean(ll, axis=0))
                          for nn, ll in pure_names_to_locs.items()}
    _, stop_locs = zip(*pure_names_to_locs.items())
    stop_to_node_dists = cdist(stop_locs, node_locs)
    closest_nodes = np.unique(stop_to_node_dists.argmin(axis=1))

    log.info(f"# original stops: {len(stops_df)}")
    log.info(f"# real stop locations: {len(closest_nodes)}")
    return closest_nodes


def gtfs_to_timeless_transit(gtfs_dir, node_locs, latlon_mapper, service_id,
                             start_time=None, end_time=None, 
                             map_stops_to_nodes=True, assoc_threshold=500, 
                             join_route_directions=False):
    """"assoc_threshold limits the distance over which a gtfs stop can be
    associated to a transit node.  The units depend on the coordinate space
    into which latlon_mapper maps; usually this is meters, but not guaranteed.
    """
    
    gtfs_dir = Path(gtfs_dir)

    # we need routes, trips, stops, and stop_times
    trips_df = pd.read_csv(gtfs_dir / "trips.txt")
    trips_df = trips_df[trips_df["service_id"] == service_id]
    stops_df = pd.read_csv(gtfs_dir / "stops.txt")

    # map stops to graph nodes
    # map stop latlons to node coordinate space
    stop_latlons = stops_df[["stop_lat", "stop_lon"]].to_numpy()
    stop_locs = [latlon_mapper(*row) for row in stop_latlons]
    if not map_stops_to_nodes:
        stop_ids_to_locs = {si: sl for si, sl in 
                            zip(stops_df["stop_id"], stop_locs)}
    # compute distances between each stop and each node
    stop_to_node_dists = cdist(stop_locs, node_locs)
    closest_nodes = stop_to_node_dists.argmin(axis=1)

    # build dict from stop IDs to associated node IDs
    stopids_to_nodes = {si: cn 
        for sn, (si, cn) in enumerate(zip(stops_df["stop_id"], closest_nodes))
        if stop_to_node_dists[sn, cn] <= assoc_threshold}

    stop_times_df = pd.read_csv(gtfs_dir / "stop_times.txt")

    # filter out stops with invalid times (< 00:00:00 or >= 24:00:00)
    deptimes = pd.to_datetime(stop_times_df["departure_time"], errors='coerce')
    arrtimes = pd.to_datetime(stop_times_df["arrival_time"], errors='coerce')
    times_are_valid = ~(deptimes.isnull() | arrtimes.isnull())
    stop_times_df = stop_times_df[times_are_valid]

    stops_on_routes, freqs_by_route = \
        get_gtfs_route_stops_and_freqs(trips_df, stop_times_df, start_time, 
                                       end_time)

    # map stop IDs to node IDs
    for route_id, route_stops in stops_on_routes.items():
        if map_stops_to_nodes:
            route_stops = [stopids_to_nodes[rs] for rs in route_stops
                        if rs in stopids_to_nodes]
        else:
            route_stops = [stop_ids_to_locs[rs] for rs in route_stops]
        stops_on_routes[route_id] = route_stops

    # combine different directions of the same route
    routes_df = pd.read_csv(gtfs_dir / "routes.txt")
    routes_ids = []
    routes = []
    freqs = []

    # "route_short_name" indicates which should be combined
    if join_route_directions:
        for route_short_name, route_df in routes_df.groupby("route_short_name"):
            conjoined_route = []
            route_freqs = []
            route_ids = list(route_df["route_id"])
            routes_ids.append(route_ids)
            for route_id in route_ids:
                if route_id not in stops_on_routes:
                    continue
                route = stops_on_routes[route_id]
                if len(conjoined_route) == 0:
                    # copy the first route
                    conjoined_route = list(route)
                if route[0] == conjoined_route[-1]:
                    conjoined_route += route[1:]
                elif route[-1] == conjoined_route[0]:
                    conjoined_route += route[:-1]
                else:
                    conjoined_route += route
                route_freqs.append(freqs_by_route[route_id])
            
            # add the first node at the end to indicate this is a loop
            conjoined_route.append(conjoined_route[0])
                
            if len(conjoined_route) > 0:
                # this means the route runs in the specified time frame
                routes.append(conjoined_route)
                if np.std(route_freqs) > 0:
                    log.warning(
                        f"Different freqs on route {route_short_name}:" \
                            f"{route_freqs}")
                freqs.append(np.mean(route_freqs))
    else:
        # treating separate directions of routes as separate
        for _, row in routes_df.iterrows():
            route_freqs = []
            route_id = row["route_id"]
            if route_id not in stops_on_routes:
                continue
            route = stops_on_routes[route_id]
            routes.append(route)
            freqs.append(freqs_by_route[route_id])
            routes_ids.append(route_id)
        
    return routes_ids, routes, freqs


def get_gtfs_route_stops_and_freqs(trips_df, stop_times_df, start_time, 
                                   end_time, service_id=None):
    # determine the stops on each route
    stops_on_routes = {}
    freqs = {}
    trip_dfs_by_route = trips_df.groupby("route_id")
    stop_times_by_trips = stop_times_df.groupby("trip_id")
    if start_time:
        start_td = datetime.timedelta(hours=start_time.hour,
            minutes=start_time.minute, seconds=start_time.second, 
            microseconds=start_time.microsecond)
    else:
        start_td = datetime.timedelta(hours=0, minutes=0, seconds=0,
                                      microseconds=0)
    if end_time:
        end_td = datetime.timedelta(hours=end_time.hour, 
            minutes=end_time.minute, seconds=end_time.second, 
            microseconds=end_time.microsecond)
    else:
        end_td = datetime.timedelta(hours=23, minutes=59, 
            seconds=59, microseconds=999999)

    timerange_span = end_td - start_td
    # group the trip stops by route ID
    if service_id:
        trips_df = trips_df[trips_df["service_id"] == service_id]
    for route_id, route_trips_df in trip_dfs_by_route:
        route_stops = []
        all_route_variants = []
        start_route_times = []
        for trip_id in route_trips_df["trip_id"]:
            try:
                trip_stop_times_df = stop_times_by_trips.get_group(trip_id)
            except KeyError:
                # this trip was filtered out
                continue
        
            # filter out trips that don't overlap with the specified time range
            deptimes = pd.to_datetime(trip_stop_times_df["departure_time"])
            arrtimes = pd.to_datetime(trip_stop_times_df["arrival_time"])
            if (start_time and deptimes.max().time() < start_time) or \
               (end_time and arrtimes.min().time() > end_time):
                continue
            start_route_times.append(arrtimes.min())

            trip_stops = list(trip_stop_times_df['stop_id'])
            if trip_stops not in all_route_variants:
                all_route_variants.append(trip_stops)
                
            # check for inconsistencies
            if route_stops != trip_stops:
                if len(route_stops) == 0:
                    route_stops = trip_stops
                else:
                    if len(all_route_variants) == 2:
                        log.warning(
                            f"Route {route_id} stops differ between trips")
                    # what if there's ambiguity about stop order?
                    route_stop_set = set(route_stops)
                    trip_stop_set = set(trip_stops)
                    if not (route_stop_set < trip_stop_set or 
                            trip_stop_set < route_stop_set):
                        log.warning("Ambiguity in stop set!")

                    # insert any stops not in route_stops
                    rsi = 0
                    for tsi, trip_stop in enumerate(trip_stops):
                        if rsi >= len(route_stops):
                            route_stops.append(trip_stop)
                        elif route_stops[rsi] != trip_stop:
                            if route_stops[rsi] in trip_stops[tsi:]:
                                # the trip stop is not in the route, insert it
                                route_stops.insert(rsi, trip_stop)
                            elif trip_stop in route_stops[rsi:]:
                                # the trip is missing some stops on the route;
                                 # advance the route until we're past the
                                 # missing part.
                                while rsi < len(route_stops) and \
                                      route_stops[rsi] != trip_stop:
                                    rsi += 1
                            else:
                                raise ValueError("we have a strict deviation!")
                        rsi += 1
            
        if len(all_route_variants) > 1 and \
           route_stops not in all_route_variants:
            raise Exception("There's no all-encompassing route variant. " \
                            "Inspect these route variants!")

        if len(route_stops) > 0:
            # this route has traversals within the time range, so include it
            stops_on_routes[route_id] = route_stops

            deltas = []
            start_route_times = sorted(start_route_times)
            for ii in range(len(start_route_times) - 1):
                delta = start_route_times[ii + 1] - start_route_times[ii]
                deltas.append(abs(delta.total_seconds()))
            if len(deltas) == 0:
                freqs[route_id] = 1 / timerange_span.total_seconds()
            else:
                freqs[route_id] = 1 / np.mean(deltas)

    return stops_on_routes, freqs


def get_inter_stop_demand(basin_adj_mat, demand_mat):
    """
    basin_adj_mat: an d x s matrix
    demand_mat: a d x d matrix
    """
    dtype = basin_adj_mat.dtype
    basin_adj_mat = basin_adj_mat.to(dtype=torch.float32)
    inter_stop_dmds = basin_adj_mat.T.mm(demand_mat).mm(basin_adj_mat)
    inter_stop_dmds.fill_diagonal_(0)
    return inter_stop_dmds.to(dtype=dtype)


# simple scaled ridership
def get_satdemand_reward_fn(scale, *args, **kwargs):
    return lambda psi, _: satdemand_reward(psi, scale)

def satdemand_reward(per_stop_info, scale):
    return [rr * scale for rr in per_stop_info["satisfied demand"]]

# global_reward = global_info[global_reward_metric] * global_reward_scale
# only global
def get_global_reward_fn(scale, quality_metric):
    return lambda psi, gi: global_reward(psi, gi[quality_metric] * scale)

def global_reward(per_stop_info, global_reward):
    rewards = []
    for pp in per_stop_info["satisfied demand"]:
        route_reward = np.ones(len(pp)) * global_reward
        rewards.append(route_reward)
    return rewards

# bagloee-like
def get_bagloee_reward_fn(per_stop_scale, global_scale, 
                          quality_metric="saved time"):
    return lambda psi, gi: bagloee_reward(psi, per_stop_scale,
                                          gi[quality_metric] * global_scale)

def bagloee_reward(per_stop_info, per_stop_scale, global_reward):
    per_stop_rewards = [psd * per_stop_scale * global_reward
                        for psd in per_stop_info["satisfied demand"]]
    return per_stop_rewards

# quadratic global reward
def get_quadratic_reward_fn(per_stop_scale, global_scale, quality_metric):
    return lambda psi, gi: \
        quadratic_global_reward(psi, per_stop_scale, 
                                gi[quality_metric] * global_scale)

def quadratic_global_reward(per_stop_info, per_stop_scale, global_reward):
    per_stop_rewards = [psd * per_stop_scale + global_reward**2
                        for psd in per_stop_info["satisfied demand"]]
    return per_stop_rewards


def main():
    log.basicConfig(level=log.INFO)

    import sys
    cfg_path = sys.argv[1]
    sim = TimelessSimulator(cfg_path, False, 
                            #"/localdata/ahollid/laval/gtfs/stops.txt"
                            )

    laval_gtfs_dir = "/localdata/ahollid/laval/gtfs_nov2013"    
    _, gtfs_routes, gtfs_freqs = sim.translate_gtfs(laval_gtfs_dir)
    sim.render_plan_on_html_map(gtfs_routes, gtfs_freqs, map_nodes=False)

    img = sim.render_plan(gtfs_routes, gtfs_freqs, map_nodes=False)
    plt.show()

    route_lens = [len(rr) for rr in gtfs_routes]
    print(f"There are {len(gtfs_routes)} gtfs routes")
    print(f"Route len mean: {np.mean(route_lens)} min: {min(route_lens)} " \
          f"max: {max(route_lens)}")
    inter_stop_dists = []
    for route in gtfs_routes:
        for ii in range(len(route) - 1):
            dist = sim.drive_dists[route[ii], route[ii+1]]
            inter_stop_dists.append(dist)
    print(f"Inter-stop distance mean: {np.mean(inter_stop_dists)}\n"\
          f"min: {min(inter_stop_dists)}\n"\
          f"25 percentile: {np.percentile(inter_stop_dists, 25)}\n"\
          f"50 percentile: {np.percentile(inter_stop_dists, 50)}\n"\
          f"75 percentile: {np.percentile(inter_stop_dists, 75)}\n"\
          f"99 percentile: {np.percentile(inter_stop_dists, 99)}\n"\
          f"max: {max(inter_stop_dists)}"\
          )
    over_1k = [isd for isd in inter_stop_dists if isd > 1000]
    print(f"of {len(inter_stop_dists)} inter-stop dists, {len(over_1k)} are "\
          "over 1000 m.")

    gtfs_rep_results = []
    start_time = time.perf_counter()
    for ii in range(40):
        log.info("iteration " + str(ii))
        _, info = sim.run(gtfs_routes, gtfs_freqs)
        gtfs_rep_results.append(info)

    time_delta = time.perf_counter() - start_time
    # print("Laval gtfs gave results:", info)
    df = pd.DataFrame(gtfs_rep_results)
    df.to_csv("gtfs_sim_results.csv")
    print(f"in {time_delta} seconds, found:")
    print("gtfs means:", df.mean(), "std devs:", df.std())

    sim.render_gtfs(laval_gtfs_dir)

    # # show kept nodes
    # img = sim.render_plan([], show_demand=True, show_node_labels=False)

    # # calculate and show newton routes
    # routes = sim.get_newton_routes(min_access_egress=1,
    #                                device=torch.device("cuda"))
    # print(len(routes), "routes were generated")

    # print(routes)
    # freqs = [5 / 3600 for _ in routes]
    # print("running...")
    # start_time = time.perf_counter()
    # sim.run(routes, freqs)
    # sim_time = time.perf_counter() - start_time
    # print("ran!  Took", sim_time, "seconds")

    # img = sim.render_plan(routes)
    # plt.show()


if __name__ == "__main__":
    main()
