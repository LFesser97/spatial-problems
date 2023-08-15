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
from itertools import cycle

import yaml
import torch
import networkx as nx
import torch_geometric.utils as pygu
from torch_geometric.data import Data, Batch, DataLoader
import matplotlib.pyplot as plt

from learning.citygraph_dataset import CityGraphData, CityGraphDataset, \
    STOP_KEY, STREET_KEY
from learning.eval_route_generator import sample_from_model
from learning.hyperheuristics import hyperheuristic
from learning.bee_colony import bee_colony
import learning.utils as lrnu
from simulation.transit_time_estimator import nikolic_cost_function


def plot_routes(data, routes, show_demand=False):
    city_pos = data[STOP_KEY].pos

    if show_demand:
        demand = data.demand
        nx_dmd_graph = nx.from_numpy_array(demand.numpy(), 
                                        create_using=nx.DiGraph)
        de_widths = torch.tensor([dd['weight'] for _, _, dd in 
                                    nx_dmd_graph.edges(data=True)])
        de_widths *= 2 / de_widths.max()
        nx.draw_networkx_edges(nx_dmd_graph, edge_color="red", 
                               pos=city_pos.numpy(), style="dashed", 
                               width=de_widths)

    edges = data[STREET_KEY].edge_index
    graph = Data(pos=city_pos, edge_index=edges)
    nx_graph = pygu.to_networkx(graph)
    # draw the street graph
    if routes is None:
        nx.draw(nx_graph, pos=city_pos.numpy(), arrows=True, node_size=100, 
                edge_color='black', width=3, arrowsize=10)

    # draw the routes
    else:
        nx.draw_networkx_nodes(nx_graph, pos=city_pos.numpy(), node_size=100)
                            
        colours = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        min_width = 2
        max_width = 10
        width_step = (max_width - min_width) / (len(routes) - 1)
        width = max_width
        for route_id, route in enumerate(routes):
            route_graph = nx.DiGraph()
            poss = {}
            route_len = (route >= 0).sum()
            route = route[:route_len]
            for si, ii in enumerate(route):
                route_graph.add_node(si)
                if si > 0:
                    route_graph.add_edge(si - 1, si)
                poss[si] = city_pos[ii].numpy()
            
            colour = next(colours)
            edge_col = nx.draw_networkx_edges(route_graph, pos=poss, 
                                              edge_color=colour, width=width, 
                                              arrowsize=width*2,
                                              node_size=0)
            width -= width_step
            edge_col[0].set_label(str(route_id))

    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset")
    parser.add_argument("weights", help=".pt file with model weights")
    parser.add_argument("cfg", 
        help="a config file specifying the configuration of the model.")
    parser.add_argument("n_routes", type=int, help="number of routes to plan")
    parser.add_argument("-s", "--n_samples", type=int, 
        help="if provided, sample this many plans in stochastic mode")
    parser.add_argument('--cpu', action='store_true',
        help="If true, run on the CPU.")
    parser.add_argument("--seed", type=int, default=0, 
        help="random seed")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")

    torch.manual_seed(args.seed)

    with open(args.cfg) as ff:
        cfg = yaml.load(ff, Loader=yaml.Loader)
    model = lrnu.build_model_from_cfg(cfg)

    data = CityGraphDataset(args.dataset, transforms=None)[0]
    if DEVICE.type == "cuda":
        data = data.cuda()

    model.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    plot_routes(data, None, show_demand=True)

    # generate routes
    if args.n_samples is None:
        plan_out = model(data, args.n_routes, greedy=True)
        model_routes = plan_out.routes
    else:
        model_routes = sample_from_model(model, data, args.n_routes, 
                                         nikolic_cost_function, 
                                         n_samples=args.n_samples)

    # plot the routes
    plot_routes(data, model_routes)
    hh_routes, _ = hyperheuristic(data, args.n_routes, nikolic_cost_function,
                                  n_steps=40000, f_0=0)
    plot_routes(data, hh_routes)

    bco_routes = bee_colony(data, args.n_routes, nikolic_cost_function)[0]
    plot_routes(data, bco_routes)


if __name__ == "__main__":
    main()