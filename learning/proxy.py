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

import copy
import argparse
from datetime import datetime
from pathlib import Path
import logging as log

import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.utils import shuffle
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import h5py
from torch_geometric.data import Data, Batch

from models import TransformerStateEncoder, FullGraphStateEncoder, Gcn, \
    LatentAttnRouteEncoder, mean_pool_sequence, get_tensor_from_varlen_lists, \
    get_mlp
from simulation.timeless_sim import TimelessSimulator

torch.autograd.set_detect_anomaly(True)
DEVICE=torch.device("cuda")


GLOBAL_IDXS = {kk: ii for ii, kk in enumerate(["saved time"])}

ROUTE_IDXS = {kk: ii for ii, kk in enumerate(["satisfied demand"])}

STOP_IDXS = {kk: ii for ii, kk in enumerate(
    ["satisfied demand", "boarders", "disembarkers", "power cost"])}


GLOBAL_RANGES = torch.tensor([[1e+4],[1e+6]])
ROUTE_RANGES = torch.tensor([[100], [400]])
STOP_RANGES = torch.tensor([[0, 0, 0, 0], [50, 100, 100, 20]])


class GraphSimProxy(nn.Module):
    def __init__(self, env_rep, embed_dim, n_heads, n_rg_layers, dropout=0.1):
        super().__init__()
        
        n_route_features = 2
        self.backbone = FullGraphStateEncoder(
            env_rep.stop_data.num_features, env_rep.demand_data.num_features,
            n_route_features, env_rep.demand_data.edge_attr.shape[1],
            env_rep.basin_weights.shape[-1], embed_dim, n_heads, 
            n_layers=n_rg_layers, dropout=dropout)

        # initialize predictors: global, per-route, per-stop
        hidden_dim = 4 * embed_dim
        n_stop_outs = len(STOP_IDXS)
        self.stop_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_stop_outs)
        )
        n_route_outs = len(ROUTE_IDXS)
        self.route_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_route_outs)
        )        
        n_global_outs = len(GLOBAL_IDXS)
        self.global_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_global_outs)
        )

    def forward(self, env_rep, batch_route_reps):
        """
        batch_route_idxs: a batch_size list of scenario lists of tensors of 
            the edge indices 
        batch_route_feats: same as above for the edge features
        """
        batch_size = len(batch_route_reps)
        max_n_routes = max([len(rr) for rr in batch_route_reps])
        glbl_out_dim = len(GLOBAL_IDXS)
        route_out_dim = len(ROUTE_IDXS)
        dev = env_rep.stop_data.x.device
        glbl_out = torch.zeros((batch_size, glbl_out_dim), device=dev)
        routes_out = torch.zeros((batch_size, max_n_routes, route_out_dim),
                                 device=dev)
        global_feat, route_out_data, stop_descs = \
            self.backbone(env_rep, batch_route_reps)
        routes_out = self.route_head(route_out_data.route_descs)
        glbl_out = self.global_head(global_feat)

        # # use stop head to predict per-route-stop values...later maybe
        # for bi, routes_feats in enumerate(batch_route_feats):
        #     # run encoder
        #     glbl_feat = route_out_data.route_edge_descs.sum(dim=-2) / \
        #         env_rep.stop_data.num_nodes
        #     # glbl_feat = dmd_out_data.edge_attr.sum(dim=-2) / \
        #     #     env_rep.stop_data.num_nodes
        #     # glbl_feat = dmd_out_data.edge_attr.mean(dim=-2)
        #     glbl_out[bi] = self.global_head(glbl_feat)
        #     # stop_preds = self.stop_head(stop_descs)
        #     # batch_stop_preds.append(stop_preds)

        return glbl_out, routes_out, None


class SimpleSimProxySeparateRoutes(nn.Module):
    def __init__(self, n_stops, embed_dim, n_route_layers, n_head_layers,
                 dropout=0.1):
        super().__init__()
        self.n_stops = n_stops
        self.nonlin = nn.LeakyReLU()
        self.route_embedder = get_mlp(n_route_layers, embed_dim, dropout,
                                      in_dim=n_stops)
        self.bn = nn.BatchNorm1d(embed_dim)

        self.global_head = get_mlp(n_head_layers, embed_dim, dropout, 
                                   out_dim=len(GLOBAL_IDXS))
        self.route_head = get_mlp(n_head_layers, embed_dim, dropout, 
                                  in_dim=embed_dim*2, out_dim=len(ROUTE_IDXS))


    def forward(self, batch_routes, norm_batch_costs):
        max_n_routes = max([len(routes) for routes in batch_routes])
        one_hot_routes = \
            torch.ones((len(batch_routes), max_n_routes, self.n_stops),
                       device=DEVICE) * -1
        pad_mask = torch.zeros(one_hot_routes.shape[:-1], device=DEVICE,
                               dtype=bool)
        # # set the costs as the last feature                               
        # one_hot_routes[:, :, -1] = norm_batch_costs

        for si, scenario_routes in enumerate(batch_routes):
            pad_mask[si, len(scenario_routes):] = True
            for ri, route in enumerate(scenario_routes):
                one_hot_routes[si, ri, route] = 1
        route_descs = self.route_embedder(one_hot_routes)
        route_descs[pad_mask] = 0

        glbl_descs = self.nonlin(self.bn(route_descs.sum(dim=-2)))
        global_preds = self.global_head(glbl_descs)

        tiled_glbl = glbl_descs[:, None, :].repeat(1, max_n_routes, 1)
        route_head_in = torch.cat((tiled_glbl, route_descs), dim=-1)
        route_preds = self.route_head(route_head_in)
        
        return global_preds, route_preds, None


class SimpleSeqSimProxy(nn.Module):
    def __init__(self, n_stops, embed_dim, n_heads, n_route_layers, 
                 n_head_layers, dropout=0.1):
        super().__init__()
        self.n_stops = n_stops
        self.embed_dim = embed_dim
        self.nonlin = nn.LeakyReLU()
        self.embedder = nn.Linear(n_stops, embed_dim)
        self.encoder = LatentAttnRouteEncoder(embed_dim, n_heads, 
                                              n_route_layers, dropout)
        self.global_head = get_mlp(n_head_layers, embed_dim, dropout, 
                                   out_dim=len(GLOBAL_IDXS))
        self.route_head = get_mlp(n_head_layers, embed_dim, dropout, 
                                  in_dim=embed_dim*2, out_dim=len(ROUTE_IDXS))
        
    
    def forward(self, batch_routes):
        node_descs = self.embedder(torch.eye(self.n_stops, device=DEVICE))
        max_n_routes = max([len(routes) for routes in batch_routes])
        all_route_descs = torch.zeros((len(batch_routes), max_n_routes, 
                                       self.embed_dim), device=DEVICE)
        for si, scen_routes in enumerate(batch_routes):
            routes_tensor, pad_mask = \
                get_tensor_from_varlen_lists(scen_routes)
            route_descs = self.encoder(node_descs, routes_tensor,
                                       padding_mask=pad_mask)
            all_route_descs[si, :len(scen_routes)] = route_descs
                                    
        global_feat = all_route_descs.sum(dim=-2) / 100
        global_preds = self.global_head(global_feat)
        tiled_glbl = global_feat[:, None, :].repeat(1, max_n_routes, 1)

        tiled_glbl = self.nonlin(tiled_glbl)
        route_head_in = torch.cat((tiled_glbl, all_route_descs), dim=-1)
        route_preds = self.route_head(route_head_in)

        return global_preds, route_preds, None


class SimpleGraphSimProxy(nn.Module):
    def __init__(self, n_stops, embed_dim, n_graph_layers, n_head_layers,
                 dropout=0.1):
        super().__init__()
        self.n_stops = n_stops
        self.embed_dim = embed_dim
        self.nonlin = nn.LeakyReLU()
        self.embedder = nn.Linear(n_stops, embed_dim)

        feature_dims = [embed_dim] * (n_graph_layers + 1)
        self.gcn = Gcn(feature_dims, nn.LeakyReLU())

        self.global_head = get_mlp(n_head_layers, embed_dim, dropout, 
                                   out_dim=len(GLOBAL_IDXS))
        self.route_head = get_mlp(n_head_layers, embed_dim, dropout, 
                                  in_dim=embed_dim*2, out_dim=len(ROUTE_IDXS))

    def forward(self, env_rep, batch_route_idxs, *args, **kwargs):
        node_descs = self.embedder(torch.eye(self.n_stops, device=DEVICE))
        # assemble graph batch
        datas = []
        for scen_route_idxs in batch_route_idxs:
            edges = torch.cat(scen_route_idxs, dim=1)
            datas.append(Data(x=node_descs, edge_index=edges))

        batch = Batch.from_data_list(datas)
        # run it through the graph net
        out_node_feats = self.gcn(batch)
        batch_size = len(batch_route_idxs)
        out_node_feats = out_node_feats.reshape(batch_size, self.n_stops, -1)

        # predict the outputs
        global_descs = []
        all_route_descs = []
        for si, scen_route_idxs in enumerate(batch_route_idxs):
            used_nodes = torch.cat(scen_route_idxs, dim=-1).unique()
            used_node_feats = out_node_feats[si, used_nodes]
            global_descs.append(used_node_feats.sum(dim=-2) / 1000)
            route_descs = []
            for route_idxs in scen_route_idxs:
                route_desc = \
                    out_node_feats[si, route_idxs.unique()].mean(dim=-2)
                route_descs.append(route_desc)
            route_descs = torch.stack(route_descs, dim=0)
            all_route_descs.append(route_descs)

        global_descs = torch.stack(global_descs)
        global_preds = self.global_head(global_descs)

        max_n_routes = max([len(routes) for routes in batch_route_idxs])
        tiled_glbl = global_descs[:, None, :].repeat(1, max_n_routes, 1)
        tiled_glbl = self.nonlin(tiled_glbl)
        all_route_descs = torch.stack(all_route_descs, dim=0)

        route_head_in = torch.cat((tiled_glbl, all_route_descs), dim=-1)
        route_preds = self.route_head(route_head_in)

        return global_preds, route_preds, None


class SimpleSimProxy(nn.Module):
    def __init__(self, n_stops, embed_dim, n_layers,
                 dropout=0.1):
        super().__init__()
        self.n_stops = n_stops
        self.bn = nn.BatchNorm1d(n_stops)
        self.nonlin = nn.LeakyReLU()
        self.backbone = get_mlp(n_layers, embed_dim, dropout,
                                in_dim=n_stops, 
                                out_dim=len(GLOBAL_IDXS) + embed_dim)
        self.route_head = get_mlp(1, embed_dim, dropout, 
                                  in_dim=embed_dim + n_stops,
                                  out_dim=len(ROUTE_IDXS))

    def forward(self, batch_routes,):
        max_n_routes = max([len(routes) for routes in batch_routes])
        stop_counts = \
            torch.zeros((len(batch_routes), self.n_stops), device=DEVICE)

        for si, scenario_routes in enumerate(batch_routes):
            for route in scenario_routes:
                stop_counts[si, route] += 1
        normed_stops = self.bn(stop_counts)
        outs = self.backbone(normed_stops)

        global_preds = outs[:, :len(GLOBAL_IDXS)]
        global_descs = outs[:, len(GLOBAL_IDXS):]
        tiled_glbl = global_descs[:, None, :].repeat(1, max_n_routes, 1)

        one_hot_routes = \
            torch.zeros((len(batch_routes), max_n_routes, self.n_stops), 
                        device=DEVICE)
        for si, scenario_routes in enumerate(batch_routes):
            for ri, route in enumerate(scenario_routes):
                one_hot_routes[si, ri, route] = 1
        
        tiled_glbl = self.nonlin(tiled_glbl)
        route_head_in = torch.cat((tiled_glbl, one_hot_routes), dim=-1)
        route_preds = self.route_head(route_head_in)
        
        return global_preds, route_preds, None


class SimProxy(nn.Module):
    def __init__(self, env_rep, embed_dim, n_heads, n_route_layers, 
                 n_ctxt_layers, dropout=0.1, min_cost=0, max_cost=273):
        super().__init__()
        self.backbone = TransformerStateEncoder(env_rep, embed_dim, n_heads, n_route_layers, 
                                n_ctxt_layers, dropout, min_cost, max_cost)
        hidden_dim = 4 * embed_dim
        n_route_outs = len(ROUTE_IDXS)
        self.route_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_route_outs)
        )        
        n_global_outs = len(GLOBAL_IDXS)
        self.global_head = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, n_global_outs)
        )

    def forward(self, env_rep, batch_routes, batch_costs, route_pad_mask,
                scenario_pad_mask=None):
        if scenario_pad_mask is None:
            scenario_pad_mask = torch.zeros(batch_costs.shape, dtype=bool,
                                            device=env_rep.stop_data.x.device)
        ctxt_route_descs, global_desc = \
            self.backbone(env_rep, batch_routes, batch_costs,
                          route_pad_mask, scenario_pad_mask)
        route_outs = self.route_head(ctxt_route_descs)
        
        max_n_routes = 120
        n_routes = scenario_pad_mask.shape[-1] - scenario_pad_mask.sum(dim=-1)
        norm_n_routes = n_routes[..., None] * 2 / max_n_routes - 1
        
        global_desc = torch.cat((global_desc, norm_n_routes), dim=-1)
        global_outs = self.global_head(global_desc)
        route_outs[scenario_pad_mask] = 0
        return global_outs, route_outs, None


def normalize(values, ranges):
    # center at 0
    midpoints = (ranges[0] + ranges[1]) / 2
    values = values - midpoints
    # reduce value range to [-1, 1]
    range_size = (ranges[1] - ranges[0]) / 2 
    values = values / range_size
    return values


def denormalize(values, ranges):
    # reduce value range to [-1, 1]
    range_size = (ranges[1] - ranges[0]) / 2 
    values = values * range_size
    # center at 0
    midpoints = (ranges[0] + ranges[1]) / 2
    values = values + midpoints
    return values


def sample_valid(n_samples, valid_mask):
    unmasked_idxs = torch.where(valid_mask)[0]
    return shuffle(unmasked_idxs)[:n_samples]


def get_balanced_sample(values, kde, sample_size):
    if values.ndim == 1:
        # add the 2nd dimension expected by KernelDensity
        values = values[:, None]

    log_scores = kde.score_samples(values)
    inverse_scores = 1 / np.exp(log_scores)
    balanced_probs = inverse_scores / inverse_scores.sum()
    value_idxs = range(len(values))
    sample = np.random.choice(value_idxs, size=sample_size, p=balanced_probs)
    return sample


def get_kde(values, kernel='gaussian', bandwidth_ratio=0.05):
    bandwidth = (values.max() - values.min()) * bandwidth_ratio
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    if values.ndim == 1:
        values = values[:, None]
    kde.fit(values)
    return kde
    

def get_balanced_probs(values, kde):
    if values.ndim == 1:
        # add the 2nd dimension expected by KernelDensity
        values = values[:, None]

    log_scores = kde.score_samples(values)
    inverse_scores = 1 / np.exp(log_scores)
    balanced_probs = inverse_scores / inverse_scores.sum()
    return torch.tensor(balanced_probs).to(device=DEVICE)


class DataLoader:
    def __init__(self, db_path, global_balance_key, route_balance_key, 
                 stop_balance_key, bandwidth_ratio=0.05, max_kde_size=10000):
        self.db = h5py.File(db_path, 'r')
        # build KDEs to balance the classes
        glbl_balance_data = self.db['global ' + global_balance_key][...]
        glbl_kde = get_kde(glbl_balance_data, bandwidth_ratio=bandwidth_ratio)
        self.glbl_probs = get_balanced_probs(glbl_balance_data, glbl_kde)
        self.glbl_balance_key = global_balance_key
        self.route_balance_key = route_balance_key
        self.stop_balance_key = stop_balance_key

    def __del__(self):
        self.db.close()

    def get_routes_and_costs(self):
        routes = []
        costs = []
        idxs = []
        for key,  dset in self.db['candidates'].items():
            routes.append(torch.tensor(dset[...], device=DEVICE))
            costs.append(dset.attrs["cost"])
            idxs.append(int(key.partition(' ')[2]))
        # sort the routes and costs in the same order they had originally,
         # to maintain the significance of their indices
        idxs, routes, costs = zip(*sorted(zip(idxs, routes, costs)))
        costs = torch.tensor(costs, dtype=torch.float32, device=DEVICE)
        return routes, costs

    def sample_batch(self, n_sample_scenarios, n_sample_routes, n_sample_stops):
        sample_idxs = self.glbl_probs.multinomial(n_sample_scenarios, 
                                                  replacement=True)
        sample_idxs = sample_idxs.cpu().numpy()                                                  
        global_vals = torch.zeros((n_sample_scenarios, len(GLOBAL_IDXS)))
        saved_times = self.db["global saved time"][...][sample_idxs]
        global_vals[:, GLOBAL_IDXS["saved time"]] = torch.tensor(saved_times)
        log.debug("sampled global values")
            
        scenarios = []

        # collect all of the route and stop values
        route_values = []
        stop_values = []
        for sample_idx in sample_idxs:
            scen_grp = self.db[str(sample_idx.item())]
            scenario = torch.tensor(scen_grp['scenario'], device=DEVICE)
            scenarios.append(scenario)
            scen_route_vals = []
            scen_stop_vals = []
            scen_route_idxs = []
            for route_key, route_grp in scen_grp.items():
                if 'route ' not in route_key:
                    # skip groups that aren't for routes
                    continue
                # make sure we record the route stats in the same order in
                 # which they appeared in the original sim run
                route_cdt_idx = int(route_key.partition(' ')[2])
                route_scen_idx = torch.where(scenario == route_cdt_idx)[0][0]
                scen_route_idxs.append(route_scen_idx.item())

                stop_boarders = route_grp["per-stop boarders"][...]
                route_stop_vals = torch.zeros((len(stop_boarders),
                                               len(STOP_IDXS)))
                for key, idx in STOP_IDXS.items():
                    vals = torch.tensor(route_grp["per-stop " + key][...])
                    route_stop_vals[:, idx] = vals
                scen_stop_vals.append(route_stop_vals)

                route_row = torch.zeros(len(ROUTE_IDXS))
                for key, idx in ROUTE_IDXS.items():
                    stop_feat_idx = STOP_IDXS[key]
                    route_row[idx] = route_stop_vals[:, stop_feat_idx].sum()
                scen_route_vals.append(route_row)
            
            _, scen_route_vals, scen_stop_vals = zip(*sorted(zip(
                scen_route_idxs, scen_route_vals, scen_stop_vals)))
            route_values.append(scen_route_vals)
            stop_values.append(scen_stop_vals)

        log.debug("assembled routes for chosen scenarios")

        # sample the desired number of route and stop values
        # flatten route values
        flat_route_values = torch.cat([torch.stack(sr) for sr in route_values])
        # compute original un-flattened indices for flattened 
        routes_per_scen = [len(rr) for rr in route_values]
        # columns are scenario index, route index
        flat_to_folded_idxs = torch.zeros((sum(routes_per_scen), 2), dtype=int)
        idx = 0
        for si, sl in enumerate(routes_per_scen):
            flat_to_folded_idxs[idx:idx + sl, 0] = si
            flat_to_folded_idxs[idx:idx + sl, 1] = torch.arange(sl)
            idx += sl
        # sample the routes to use
        # flat_sample_route_idxs = \
        #     torch.randint(len(flat_route_values), (n_sample_routes,))
        balance_vals = flat_route_values[:, ROUTE_IDXS[self.route_balance_key]]
        kde = get_kde(balance_vals)
        route_probs = get_balanced_probs(balance_vals, kde)
        flat_route_idxs = route_probs.multinomial(n_sample_routes, 
                                                  replacement=True)
        sample_route_idxs = flat_to_folded_idxs[flat_route_idxs]
        sample_route_vals = flat_route_values[flat_route_idxs]

        log.debug("sampled route values")

        flat_stop_values = torch.cat([torch.cat(ss) for ss in stop_values])
        stops_per_route = [[len(rsv) for rsv in scen_stop_values]
                           for scen_stop_values in stop_values]
        total_n_stops = sum([sum(nrs) for nrs in stops_per_route])
        # columns are scenario index, route index, stop index
        flat_to_folded_idxs = torch.zeros((total_n_stops, 3), dtype=int)
        scen_idx = 0
        for si, scen_stops_per_route in enumerate(stops_per_route):
            route_idx = 0
            n_scen_stops = sum(scen_stops_per_route)
            flat_to_folded_idxs[scen_idx:scen_idx + n_scen_stops, 0] = si
            
            route_idx = scen_idx
            for ri, n_route_stops in enumerate(scen_stops_per_route):
                flat_to_folded_idxs[route_idx:route_idx + n_route_stops, 1] = ri
                flat_to_folded_idxs[route_idx:route_idx + n_route_stops, 2] = \
                    torch.arange(n_route_stops)
                route_idx += n_route_stops

            scen_idx += n_scen_stops

        # sample stops to use
        balance_vals = flat_stop_values[:, STOP_IDXS[self.stop_balance_key]]
        kde = get_kde(balance_vals)
        stop_probs = get_balanced_probs(balance_vals, kde)
        flat_stop_idxs = stop_probs.multinomial(n_sample_stops, 
                                                replacement=True)

        sample_stop_idxs = flat_to_folded_idxs[flat_stop_idxs]
        sample_stop_vals = flat_stop_values[flat_stop_idxs]

        log.debug("sampled stop values")

        global_vals = global_vals.to(device=DEVICE)
        sample_stop_idxs = sample_stop_idxs.to(device=DEVICE).T
        sample_stop_vals = sample_stop_vals.to(device=DEVICE)
        sample_route_idxs = sample_route_idxs.to(device=DEVICE).T
        sample_route_vals = sample_route_vals.to(device=DEVICE)

        log.debug("min route value:" + str(sample_route_vals.min(dim=0)[0]))
        log.debug("max route value:" + str(sample_route_vals.max(dim=0)[0]))
        log.debug("min stop value:" + str(sample_stop_vals.min(dim=0)[0]))
        log.debug("max stop value:" + str(sample_stop_vals.max(dim=0)[0]))

        # return sampled indices and values
        return (scenarios, global_vals), \
            (sample_route_idxs, sample_route_vals), \
            (sample_stop_idxs, sample_stop_vals)


def train(data_loader, model, env_rep, route_reps, n_steps, batch_size, 
          learning_rate, weight_decay, summary_writer):

    # set up optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
                                
    # set up loss function
    loss_fn = nn.MSELoss()

    # add a dummy cost for padding route indices
    candidate_routes, costs = data_loader.get_routes_and_costs()
    costs = torch.cat((costs.to(device=DEVICE), 
                      torch.tensor([0], device=DEVICE)))
    norm_costs = (costs - costs.mean()) / costs.std()
    # convert candidate routes to a tensor
    cdt_routes_tensor, cdt_routes_mask = \
        get_tensor_from_varlen_lists(candidate_routes, DEVICE, add_dummy=True)

    # TODO figure out randomization of costs; for now used fixed costs
    lowest_loss = float("inf")
    best_model = copy.deepcopy(model)
    accum_loss = None
    alpha = 0.1
    model.train()

    n_sample_routes = batch_size * 10
    n_sample_stops = 5 * n_sample_routes

    saved_global_gts = []
    saved_route_gts = []

    for step in tqdm(range(n_steps + 1)):
        n_scens = step * batch_size

        # sample batch_size systems
        log.debug("sampling batch")
        (scenarios, global_gts), (sample_route_idxs, route_gts), \
            (sample_stop_idxs, stop_gts) = \
            data_loader.sample_batch(batch_size, n_sample_routes, 
                                     n_sample_stops)
        
        for scenario in scenarios:
            assert len(scenario.unique()) == len(scenario)
        saved_global_gts.append(global_gts)
        saved_route_gts.append(route_gts)
        
        scenarios, scen_pad_mask = \
            get_tensor_from_varlen_lists(scenarios, DEVICE)
        scen_costs = costs[scenarios]
        route_tensor = cdt_routes_tensor[scenarios]
        route_pad_mask = cdt_routes_mask[scenarios]
        

        # run them through the model to get predicted outputs
        log.debug("running model forward")
        if type(model) in (GraphSimProxy, SimpleGraphSimProxy):
            batch_route_reps = [[route_reps[ii] for ii in scenario]
                                for scenario in scenarios]
            global_ests, route_ests, stop_ests = \
                model(env_rep, batch_route_reps)
        elif type(model) in (SimpleSimProxy, SimpleSeqSimProxy):
            batch_routes = [[candidate_routes[ri] for ri in scenario]
                            for scenario in scenarios]
            global_ests, route_ests, stop_ests = \
                model(batch_routes)
        elif type(model) is SimpleSimProxySeparateRoutes:
            batch_routes = [[candidate_routes[ri] for ri in scenario]
                            for scenario in scenarios]
            global_ests, route_ests, stop_ests = \
                model(batch_routes, norm_costs[scenarios])

        else:
            global_ests, route_ests, stop_ests = \
                model(env_rep, route_tensor, scen_costs, route_pad_mask, 
                      scen_pad_mask)

        # global_ests, route_ests, stop_ests = model(scenarios)
                        
        # compute loss versus the ground truth
        log.debug("computing loss")
        norm_global_gts = normalize(global_gts, GLOBAL_RANGES)
        norm_route_gts = normalize(route_gts, ROUTE_RANGES)
        global_loss = loss_fn(global_ests, norm_global_gts)
        summary_writer.add_scalar("global loss", global_loss.item(), n_scens)
        assert not global_loss.isnan()
        # probably a minimum of 10 routes per scenario
        flat_route_ests = route_ests[sample_route_idxs[0], 
                                     sample_route_idxs[1]]

        route_loss = loss_fn(flat_route_ests, norm_route_gts)
        summary_writer.add_scalar("per-route loss", route_loss.item(), n_scens)
                                     
        loss = global_loss + route_loss

        if stop_ests is not None:
            norm_stop_gts = normalize(stop_gts, STOP_RANGES)
            flat_stop_ests = stop_ests[sample_stop_idxs[0], 
                                       sample_stop_idxs[1]]

            route_stop_loss = loss_fn(flat_stop_ests, norm_stop_gts)
            summary_writer.add_scalar("route stop loss", 
                                      route_stop_loss.item(), n_scens)
            loss += route_stop_loss

        if accum_loss is None:
            accum_loss = loss.item()
        else:
            accum_loss = accum_loss * (1 - alpha) + loss.item() * alpha
        if accum_loss < lowest_loss:
            best_model = copy.deepcopy(model)
            lowest_loss = accum_loss

        # log errors of different outputs
        log.debug("logging iteration stats")
        summary_writer.add_scalar("loss", loss.item(), n_scens)

        denorm_global_ests = denormalize(global_ests.detach(), GLOBAL_RANGES)
        saved_time_err = (denorm_global_ests[GLOBAL_IDXS["saved time"]] - 
            global_gts[GLOBAL_IDXS["saved time"]]).abs().mean().item()
        summary_writer.add_scalar("saved time error", saved_time_err, n_scens)
        denorm_route_ests = denormalize(flat_route_ests, ROUTE_RANGES)

        for key, idx in ROUTE_IDXS.items():
            err = (denorm_route_ests[:, idx] - route_gts[:, idx]).abs().mean()
            summary_writer.add_scalar(key + ' error', err.item(), n_scens)

        for name, vals in [
            ("saved time", global_gts), 
            ("total satisfied demand", 
             route_gts[:, ROUTE_IDXS["satisfied demand"]].sum()), 
            ]:
            if vals.ndim > 2:
                vals = vals[~scen_pad_mask]
            summary_writer.add_scalar(name + " mean", vals.mean().item(), 
                                      n_scens)
            summary_writer.add_scalar(name + " std", vals.std().item(), n_scens)
            summary_writer.add_scalar(name + " min", vals.min().item(), n_scens)
            summary_writer.add_scalar(name + " max", vals.max().item(), n_scens)

        # the final "step" is just to check performance after the 
         # weight update of the one before 
        if step < n_steps:
            log.debug("doing backprop and weight update")
            # backprop and update weights (and grad norm?)
            loss.backward()
            log.debug("stepping optimizer")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            log.debug("done stepping")

    # def plot_stats(stat_matrix, name_dict):
    #     for name, idx in name_dict.items():
    #         plt.hist(stat_matrix[:, idx].cpu().numpy())
    #         plt.title(name)
    #         plt.show()

    # plot_stats(torch.cat(saved_global_gts), GLOBAL_IDXS)
    # plot_stats(torch.cat(saved_route_gts), ROUTE_IDXS)

    return best_model


def evaluate_sim_randomness(sim, dl, batch_size, n_steps):
    """Compare the recorded outputs in the dataset to the outputs we get
        re-running the simulator on these inputs."""
    n_sample_routes = batch_size * 10
    n_sample_stops = 5 * n_sample_routes
    candidate_routes, costs = dl.get_routes_and_costs()
    loss_fn = nn.MSELoss()

    cum_global_loss = 0
    cum_route_loss = 0
    cum_stop_loss = 0

    for step in tqdm(range(n_steps)):
        # sample batch_size systems
        log.debug("sampling batch")
        (scenarios, global_dbs), (sample_route_idxs, route_dbs), \
            (sample_stop_idxs, stop_dbs) = \
            dl.sample_batch(batch_size, n_sample_routes, n_sample_stops)

        global_dbs = normalize(global_dbs, GLOBAL_RANGES)
        route_dbs = normalize(route_dbs, ROUTE_RANGES)
        stop_dbs = normalize(stop_dbs, STOP_RANGES)
        batch_routes = [[candidate_routes[ri] for ri in scenario]
                        for scenario in scenarios]
        
        glbl_sims = torch.zeros((len(scenarios), len(GLOBAL_IDXS)), 
                                device=DEVICE)
        max_n_routes = max([len(scenario) for scenario in scenarios])
        route_sims = torch.zeros((len(scenarios), max_n_routes, 
                                  len(ROUTE_IDXS)), device=DEVICE)
        max_n_stops = max([max([len(rr) for rr in routes])
                           for routes in batch_routes])
        stop_sims = torch.zeros((len(scenarios), max_n_routes, max_n_stops,
                                 len(STOP_IDXS)), device=DEVICE)

        for si, (scenario, routes) in enumerate(zip(scenarios, batch_routes)):
            scen_costs = costs[scenario]
            scen_freqs = sim.capacity_to_frequency(scen_costs)
            stop_info, global_info = sim.run(routes, scen_freqs, 
                                             fast_demand_assignment=False)
            for name, idx in GLOBAL_IDXS.items():                                         
                glbl_sims[si, idx] = global_info[name]
            
            for name, idx in ROUTE_IDXS.items():
                for ri in range(len(routes)):
                    route_sims[si, ri, idx] = stop_info[name][ri].sum()

            for name, idx in STOP_IDXS.items():
                for ri, route in enumerate(routes):
                    stop_sims[si, ri, :len(route), idx] = \
                        torch.tensor(stop_info[name][ri], device=DEVICE)

        glbl_sims = normalize(glbl_sims, GLOBAL_RANGES)
        route_sims = normalize(route_sims, ROUTE_RANGES)
        stop_sims = normalize(stop_sims, STOP_RANGES)

        glbl_loss = loss_fn(global_dbs, glbl_sims)
        cum_global_loss = (cum_global_loss * step + glbl_loss) / (step + 1)
        print("average global error:", cum_global_loss)

        route_sims = route_sims[sample_route_idxs[0],
                                sample_route_idxs[1]]
        route_loss = loss_fn(route_dbs, route_sims)
        cum_route_loss = (cum_route_loss * step + route_loss) / (step + 1)
        print("average route error:", cum_route_loss)


def test_proxy_as_greedy_planner(sim, model, candidate_routes, route_costs, 
                                 budget, batch_size=32):
    routes_are_included = torch.zeros(len(candidate_routes), dtype=bool, 
                                      device=DEVICE)
    # make sure this is a copy of the original budget, just in case
    remaining_budget = float(budget)
    model.eval()
    are_valid_choices = ~routes_are_included & \
                        (route_costs <= remaining_budget)
    env_rep = sim.get_env_rep_for_nn(device=DEVICE, one_hot_node_feats=True)
    cdt_routes_tensor, cdt_routes_mask = \
        get_tensor_from_varlen_lists(candidate_routes, device=DEVICE, 
                                     add_dummy=True)
    route_reps = \
        sim.get_route_reps_for_nn(candidate_routes, route_costs, device=DEVICE)

    scenario_so_far = []
    pbar = tqdm()
    while are_valid_choices.any():

        # pick the next estimated best route
        # batched version of the below; needs too much memory.
        scenarios = []
        for cdt_idx, is_valid in enumerate(are_valid_choices):
            if not is_valid:
                continue
            scenarios.append(scenario_so_far + [cdt_idx])
            
        ests = []
        for ii in range(0, len(scenarios), batch_size):
            batch_scenarios = scenarios[ii:ii + batch_size]
            with torch.no_grad():
                if type(model) is SimProxy:
                    batch_route_tensor = cdt_routes_tensor[[batch_scenarios]]
                    batch_stop_mask = cdt_routes_mask[[batch_scenarios]]
                    batch_costs = route_costs[[batch_scenarios]]
                    global_ests, route_ests, _ = \
                        model(env_rep, batch_route_tensor, batch_costs, 
                              batch_stop_mask)
                elif type(model) is GraphSimProxy:
                    batch_route_reps = [[route_reps[ii] for ii in scenario]
                                        for scenario in batch_scenarios]
                    global_ests, route_ests, _ = \
                        model(env_rep, batch_route_reps)

            # ests.append(route_ests[:, ROUTE_IDXS["satisfied demand"]].sum())
            ests.append(global_ests[:, GLOBAL_IDXS["saved time"]].sum())
        
        global_ests = torch.stack(ests)
        best_quality, best_idx = global_ests.max(0)
        best_cdt_idx = scenarios[best_idx][-1]

        pbar.update(1)

        # add the best candidate to the scenario
        scenario_so_far.append(best_cdt_idx)
        # update loop variables
        routes_are_included[best_cdt_idx] = True
        remaining_budget -= route_costs[best_cdt_idx]
        are_valid_choices = (~routes_are_included) & \
                            (route_costs <= remaining_budget)

    pbar.close()

    # quality_range = ROUTE_RANGES[:, ROUTE_IDXS["satisfied demand"]]
    saved_time_range = GLOBAL_RANGES[:, GLOBAL_IDXS["saved time"]]
    est_saved_time = denormalize(best_quality, saved_time_range)
    print("estimated saved time:", est_saved_time)

    with torch.no_grad():
        if type(model) is SimProxy:
            routes_tensor = cdt_routes_tensor[[[scenario_so_far]]]
            stop_mask = cdt_routes_mask[[[scenario_so_far]]]
            costs = route_costs[[[scenario_so_far]]]
            global_ests, route_ests, _ = \
                model(env_rep, routes_tensor, costs, stop_mask)
        elif type(model) is GraphSimProxy:
            route_reps = [route_reps[ii] for ii in scenario_so_far]
            global_ests, route_ests, _ = \
                model(env_rep, [route_reps])

        # global_ests, route_ests, _ = model([scenario_so_far])
    est_sat_dem = route_ests[0, ROUTE_IDXS["satisfied demand"]].sum()
    sat_dem_range = ROUTE_RANGES[:, ROUTE_IDXS["satisfied demand"]]
    est_sat_dem = denormalize(est_sat_dem, sat_dem_range)
    print("estimated satisfied demand:", est_sat_dem)

    scenario_costs = route_costs[scenario_so_far]
    freqs = sim.capacity_to_frequency(scenario_costs)
    routes = [candidate_routes[ri] for ri in scenario_so_far]
    per_stop, glbl = sim.run(routes, freqs, fast_demand_assignment=False)
    true_sat_dem = sum([ps.sum() for ps in per_stop["satisfied demand"]])
    print("true saved time:", glbl["saved time"])
    print("true satisfied demand:", true_sat_dem)

    return routes, freqs 

def main():
    # get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('sim_config', help="The simulator configuration file.")
    parser.add_argument('db', help="The path to the database file.")
    weight_grp = parser.add_mutually_exclusive_group()
    weight_grp.add_argument('--test', 
        help="neural network weights to test in planning mode")
    weight_grp.add_argument('-w', '--weights', 
        help="neural network weights to start with")
    parser.add_argument("--lr", type=float, default=0.001, 
                        help="the learning rate")
    parser.add_argument("--n_steps", type=int, default=50,
        help="the number of steps (batches) to train for")
    parser.add_argument("-b", "--batch_size", type=int, default=32,
        help="the number of scenarios in each batch.")
    # TODO try this
    # parser.add_argument('--gtfs',
    #     help="If provided, the routes defined in the gtfs directory will be " \
    #          "included.")
    parser.add_argument("--decay", type=float, default=0.1, 
        help="the weight decay")
    parser.add_argument("--nrgl", type=int, default=2,
        help="depth of route graph encoder")
    parser.add_argument("--nrl", type=int, default=1, 
        help="depth of individual route encoder")
    parser.add_argument("--ncl", type=int, default=1,
        help="depth of route context encoder")
    parser.add_argument("--n_global_l", type=int, default=2, 
        help="depth of global context encoder")
    parser.add_argument("--nheads", type=int, default=8, 
        help="number of heads for multi-head components")
    parser.add_argument("--embed", type=int, default=16, 
        help="embedding dimension")
    parser.add_argument("--mb", type=float, default=30000, 
        help="maximum budget")
    parser.add_argument("--logdir", default="proxy_logs", 
        help="dir for tensorboard logging")
    parser.add_argument('--run_name', '--rn',
        help="Name of this run, for logging purposes.")
    parser.add_argument('--cpu', action='store_true',
        help="If true, run on the CPU.")
    parser.add_argument('--log', help="Specify the logging level to use.")
    args = parser.parse_args()

    if args.log:
        level = getattr(log, args.log.upper(), None)
        if not isinstance(level, int):
            raise ValueError(f"Invalid log level: {level}")
        log.basicConfig(level=level, format="%(asctime)s %(message)s")

    if args.cpu:
        global DEVICE
        DEVICE = torch.device("cpu")
    
    global GLOBAL_RANGES
    GLOBAL_RANGES = GLOBAL_RANGES.to(device=DEVICE)
    global ROUTE_RANGES
    ROUTE_RANGES = ROUTE_RANGES.to(device=DEVICE)
    global STOP_RANGES
    STOP_RANGES = STOP_RANGES.to(device=DEVICE)

    # assemble the simulator
    sim = TimelessSimulator(args.sim_config, filter_nodes=True, 
                            # stops_path="/localdata/ahollid/laval/gtfs/stops.txt"
                            )
    dl = DataLoader(args.db, "saved time", "satisfied demand", 
                    "satisfied demand")

    # evaluate_sim_randomness(sim, dl, args.batch_size, args.n_steps)

    routes, costs = dl.get_routes_and_costs()

    # assemble the model from the arguments
    env_rep = sim.get_env_rep_for_nn(device=DEVICE, one_hot_node_feats=True)
    route_reps = \
        sim.get_route_reps_for_nn(routes, costs, device=DEVICE)

    # model = SimProxy(env_rep, args.embed, args.nheads, args.nrl, args.ncl)
    # model = SimpleSimProxy(env_rep.stop_data.num_nodes, args.embed, args.nrl)
    # model = SimpleSeqSimProxy(env_rep.stop_data.num_nodes, args.embed, 
    #                           args.nheads, args.nrl, args.n_global_l)
    # model = SimpleGraphSimProxy(env_rep.stop_data.num_nodes, args.embed, 
    #                             args.nrgl, args.n_global_l)
    # model = SimpleSimProxySeparateRoutes(
    #     env_rep.stop_data.num_nodes, args.embed, args.nrl, 
    #     args.n_global_l)
    # total_params = sum(pp.numel() for pp in model.parameters())
    # print('total simple2 params:', total_params)
    model = GraphSimProxy(env_rep, args.embed, args.nheads, args.nrgl)
    # total_params = sum(pp.numel() for pp in model.parameters())
    # print('total graph params', total_params)
    # model = StopDecoder(sim.get_env_rep_for_nn(device=DEVICE), args.embed, 
    #                                            args.nheads, args.nsl)

    if args.weights is not None:
        model.load_state_dict(torch.load(args.weights))

    if args.test is None:
        model = model.to(DEVICE)

        log_path = Path(args.logdir)
        if args.run_name is None:
            run_name = datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
        else:
            run_name = args.run_name
        summary_writer = SummaryWriter(log_path / run_name)

        # run train with the arguments (the core argument makes it faster)
        model = train(dl, model, env_rep, route_reps, args.n_steps, 
                      args.batch_size, args.lr, args.decay, summary_writer)

        log.info("training complete.  Saving model.")
        output_dir = Path("proxy_outputs")
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        torch.save(model.backbone.state_dict(), 
                   output_dir / (run_name + 'backbone.pt'))
        torch.save(model.state_dict(), 
                   output_dir / (run_name + '.pt'))
    else:
        model.backbone.load_state_dict(torch.load(args.test))
        model = model.to(device=DEVICE)

    print("evaluating model as a planner...")
    real_budget = 28200
    routes, freqs = \
        test_proxy_as_greedy_planner(sim, model, routes, costs, real_budget)
    # save an image of the best routes and frequencies
    image = sim.get_network_image(routes, freqs)
    if args.test is None:
        summary_writer.add_image('City with routes', image, 0)
    else:
        plt.imshow(image.transpose(1, 2, 0))
        plt.show()


if __name__ == "__main__":
    main()
