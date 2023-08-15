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
import logging as log
import math
from collections import namedtuple

import numpy as np
import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.nn import (GATv2Conv, GCNConv, MessagePassing, SGConv,
                                BatchNorm, GraphNorm)
from torch_utils import (aggregate_dense_conns, cat_var_size_tensors,
                         floyd_warshall, get_batch_tensor_from_routes,
                         get_indices_from_mask, get_tensor_from_varlen_lists, 
                         get_update_at_mask, get_variable_slice_mask, 
                         reconstruct_all_paths, get_route_edge_matrix,
                         get_route_leg_times
)
from trunc_normal import TruncatedNormal

from world.transit import RouteRep
from simulation.citygraph_dataset import \
    DEMAND_KEY, STOP_KEY, STREET_KEY, ROUTE_KEY, CityGraphData

Q_FUNC_MODE = "q function"
PLCY_MODE = "policy"
GFLOW_MODE = "gflownet"

# use this in place of -float('inf') to avoid nans in some badly-behaved
 # backprops
TORCH_FMIN = torch.finfo(torch.float).min
TORCH_FMAX = torch.finfo(torch.float).max

GREEDY = False

MLP_DEFAULT_DROPOUT = 0.5
ATTN_DEFAULT_DROPOUT = 0.1
DEFAULT_NONLIN = nn.GELU


PlanResults = namedtuple(
    "PlanResults", 
    ["routes", "freqs", "stop_logits", "route_logits", "freq_logits", 
    "stops_tensor", "routes_tensor", "freqs_tensor", "stop_est_vals", 
    "route_est_vals", "freq_est_vals"]
    )

RouteGenResults = namedtuple(
    "GenRouteResults", ["routes", "route_descs", "logits", "selections", 
                        "est_vals"]
    )

RouteChoiceResults = namedtuple(
    "RouteChoiceResults", ["logits", "selections", "est_vals"]
    )

FreqChoiceResults = namedtuple(
    "FreqChoiceResults", ["freqs", "logits", "selections", "est_vals"]
    )

class BatchNormHelper(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    
    def forward(self, features):
        # swap the batch and sequence dimensions, since batchnorm expects
        # batch dimension to be first
        act = features.permute(1, 2, 0)
        act = self.bn(act)
        # swap them back!
        return act.permute(2, 0, 1)


# Backbone encoder modules
class GraphEncoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim


class KoolGraphEncoder(GraphEncoder):
    def __init__(self, in_dim, enc_dim, n_heads=8, n_layers=3, ff_dim=512):
        super().__init__(enc_dim)
        self.linear_embedding = nn.Linear(in_dim, enc_dim, bias=True)
        enc_layer = nn.TransformerEncoderLayer(enc_dim, n_heads, ff_dim)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)

    def forward(self, node_features, adjacency_matrix=None, *args, **kwargs):
        """Adjacency matrix is treated as binary!"""
        node_features = self.linear_embedding(node_features)
        if adjacency_matrix is not None:
            if type(adjacency_matrix) is not torch.Tensor:
                adjacency_matrix = torch.tensor(adjacency_matrix, 
                    dtype=torch.bool)
        node_features = node_features[:, None, :]
        encoded_feats = self.transformer(node_features, adjacency_matrix > 0)
        return encoded_feats.squeeze(dim=1)


class DemandEncoder(nn.Module):
    def __init__(self, node_dim, edge_dim, embed_dim, 
                 dropout=MLP_DEFAULT_DROPOUT):
        super().__init__()
        self.nonlin = DEFAULT_NONLIN()
        self.dropout = nn.Dropout(dropout)
        # this is the floor
        fwd_dim = embed_dim // 2
        self.fwd_demand_gcn = \
            EdgeGraphNetLayer(embed_dim, node_dim, edge_dim, 
                              fwd_dim, flow="source_to_target")
        # this way, fwd_dim + rev_dim == enc_dim
        rev_dim = embed_dim - fwd_dim
        self.reverse_demand_gcn = \
            EdgeGraphNetLayer(embed_dim, node_dim, edge_dim, 
                              rev_dim, flow="target_to_source")
    
    def forward(self, demand_data):
        fwd_node_feats, fwd_edge_feats = \
            self.fwd_demand_gcn(demand_data.x, demand_data.edge_index,
                                demand_data.edge_attr)
        rvs_node_feats, rvs_edge_feats = \
            self.reverse_demand_gcn(demand_data.x, demand_data.edge_index,
                                    demand_data.edge_attr)
        node_feats = torch.cat((fwd_node_feats, rvs_node_feats), dim=-1)
        edge_feats = torch.cat((fwd_edge_feats, rvs_edge_feats), dim=-1)
        return node_feats, edge_feats


class DualGcnEncoder(GraphEncoder):
    def __init__(self, env_rep, embed_dim, dropout=MLP_DEFAULT_DROPOUT):
        super().__init__(embed_dim)
        self.stop_embedding = nn.Sequential(
            nn.Linear(env_rep.stop_data.num_features, embed_dim),
            # nn.Dropout(dropout),
            # DEFAULT_NONLIN()
        )
        self.nonlin = DEFAULT_NONLIN()
        self.dropout = nn.Dropout(dropout)
        self.init_demand_gcn = DemandEncoder(env_rep.demand_data.num_features,
            env_rep.demand_data.edge_attr.shape[1], embed_dim, dropout)
        
        # build the GCN to interface between demand and stop features
        self.combine_gcn = EdgeGraphNetLayer(embed_dim, embed_dim, 1)

    def forward(self, env_rep, *args, **kwargs):
        return self._combine_stop_and_demand_info(env_rep)

    def _combine_stop_and_demand_info(self, env_rep):
        """Adjacency matrix is treated as binary!"""
        stop_data = copy.copy(env_rep.stop_data)
        stop_features = self.stop_embedding(env_rep.stop_data.x)
        stop_data.x = stop_features

        demand_data = copy.copy(env_rep.demand_data)
        dx, _ = self.init_demand_gcn(demand_data)
        demand_data.x = self.dropout(self.nonlin(dx))

        # combine the stop and demand features
        all_feats = torch.cat((stop_data.x, demand_data.x))

        comb_size = stop_data.num_nodes + demand_data.num_nodes
        comb_adj_mat = torch.zeros((comb_size, comb_size), 
                                    device=env_rep.stop_data.x.device)
        comb_idxs = env_rep.basin_edges.clone()
        comb_idxs[0] += stop_data.num_nodes
        comb_adj_mat[comb_idxs[0], comb_idxs[1]] = 1
        comb_adj_mat[comb_idxs[1], comb_idxs[0]] = 1
        comb_index = torch.stack(torch.where(comb_adj_mat))
        comb_attr = env_rep.basin_weights.repeat(2, 1)

        all_out_feats, _ = self.combine_gcn(all_feats, comb_index, comb_attr)
        out_stop_feats = all_out_feats[:stop_features.shape[-2]]
        out_demand_feats = all_out_feats[stop_features.shape[-2]:]
        return self.dropout(self.nonlin(out_stop_feats)), \
            self.dropout(self.nonlin(out_demand_feats))


class DualGcnTransformerEncoder(DualGcnEncoder):
    def __init__(self, env_rep, enc_dim, n_heads=8, n_transformer_layers=3):
        super().__init__(env_rep, enc_dim, dropout=MLP_DEFAULT_DROPOUT)

        enc_layer = nn.TransformerEncoderLayer(enc_dim, n_heads, 512,
                                               ATTN_DEFAULT_DROPOUT,
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                 n_transformer_layers)
        n_stops = env_rep.stop_data.num_nodes
        self.tnsfmr_adj_mat = torch.zeros((n_stops, n_stops), dtype=bool, 
                                          device=env_rep.stop_data.x.device)
        for ii, jj in env_rep.stop_data.edge_index.t():
            self.tnsfmr_adj_mat[ii, jj] = True            

    def forward(self, env_rep, *args, **kwargs):
        """Adjacency matrix is treated as binary!"""
        intermediate = self._combine_stop_and_demand_info(env_rep)
        # add self-connections
        selfconns = torch.eye(intermediate.shape[0], dtype=bool, 
                              device=intermediate.device)
        self.tnsfmr_adj_mat.logical_and_(selfconns.logical_not())

        # add a batch dimension
        intermediate = intermediate[None]
        encoded_feats = self.transformer(intermediate, self.tnsfmr_adj_mat) 
        encoded_feats = encoded_feats.squeeze(dim=0)
        return encoded_feats

# Node scorer modules

class KoolNextNodeScorer(nn.Module):
    # This follows Kool et al 2019
    def __init__(self, embed_dim, clip_size=10):
        """
        embed_dim: the dimension of the embeddings.
        softmax_temp: the temperature of the softmax distribution over nodes.
            default is 1 (no change to energies).
        clip_size: maximum absolute value of scores that can be returned.
        """
        super().__init__()
        self.feature_dim = embed_dim
        self.clip_size = clip_size

        self.node_embedding_projector = nn.Linear(embed_dim, embed_dim*3,
                                                  bias=False)
        self.context_layer2_projector = nn.Linear(embed_dim, embed_dim)

        self.attn_embeddings = None

    def precompute(self, node_vecs):
        embeddings = self.node_embedding_projector(node_vecs)
        self.attn_embeddings = embeddings.chunk(3, dim=-1)
    
    def forward(self, context, mask=None):
        """
        context: the context vector specific to this step.
        mask: a mask that is True for invalid choices.
        """
        # context = context.reshape(1, -1)
        k_layer1, v_layer1, k_layer2 = self.attn_embeddings
        # perform the attention operations

        if k_layer1.ndim == 3: # and context.ndim == 2:
            # there's a batch dimension in the embeddings, so add an 
            # appropriate dimension to the context tensor
            context = context[:, None, :]

        layer1_attn = torch.matmul(context, k_layer1.transpose(-2, -1))
        layer1_attn = layer1_attn / np.sqrt(k_layer1.size(-1))
        layer1_scores = torch.softmax(layer1_attn, dim=-1)
        context_layer2 = torch.matmul(layer1_scores, v_layer1)
        context_layer2 = self.context_layer2_projector(context_layer2)
        layer2_attn = torch.matmul(context_layer2, k_layer2.transpose(-2, -1))
        scores = layer2_attn / np.sqrt(k_layer2.size(-1))
        
        if self.clip_size:
            # limit the range of magnitudes of the energies with a tanh.
            scores = self.clip_size * torch.tanh(scores)

        if scores.dim() == 3:
            scores = scores.squeeze(-2)
    
        if mask is not None:
            # keep masked elements as -inf, so they'll have probability 0.
            scores[mask] = TORCH_FMIN
        
        return scores

    def reset(self):
        self.attn_embeddings = None


class MlpActionScorer(nn.Module):
    def __init__(self, embed_dim, n_layers=2, dropout=MLP_DEFAULT_DROPOUT, 
                 yes_and_no_outs=False):
        super().__init__()
        self.embed_dim = embed_dim
        out_dim = 2 if yes_and_no_outs else 1
        self.net = get_mlp(n_layers, embed_dim, dropout, 
                           in_dim=embed_dim * 2, out_dim=out_dim, bias=False)

    def forward(self, action_descs, state_vec, valid_actions_bool):
        """
        action_descs: 1 x n_routes or batch_size x n_routes tensor
        state_vec: batch_size x embed_dim tensor of state descriptors
        valid_actions_bool: batch_size x n_routes boolean tensor that is True
            for routes that are valid, False otherwise.  If None, ignored.
        """
        batch_size = len(state_vec)
        if action_descs.ndim == 2:
            # add a batch dimension
            action_descs = action_descs[None]
        if action_descs.shape[0] == 1 and batch_size > 1:
            action_descs = action_descs.tile(batch_size, 1, 1)
        
        scores = self.score_actions(action_descs, state_vec)

        # set invalid actions to have value -inf, so they won't be chosen
        if valid_actions_bool is not None:
            if scores.shape[-1] == 1:
                scores[~valid_actions_bool] = TORCH_FMIN
            elif scores.shape[-1] == 2:
                invalid_actions_bool = ~valid_actions_bool.squeeze(-1)
                scores[..., 0][invalid_actions_bool] = 0
                scores[..., 1][invalid_actions_bool] = TORCH_FMIN
        return scores

    def score_actions(self, action_vecs, state_vec):
        """
        action_vecs: batch_size x max_n_actions x embed_dim tensor
        state_vec: batch_size x embed_dim tensor

        Returns batch_size x len(action_vecs) tensor
        """
        state_vec = state_vec[:, None, :]
        state_vec = state_vec.expand(-1, action_vecs.shape[-2], -1)
        in_feats = torch.cat((action_vecs, state_vec), dim=-1)
        scores = self.net(in_feats)
        return scores


class RoutesEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, node_descs, routes_tensor, padding_mask, **encode_args):
        # build route tensor and mask
        old_shape = None
        nodes_have_batch = node_descs.ndim > 2
        if nodes_have_batch:
            batch_size = node_descs.shape[0]
            nodes_per_batch = node_descs.shape[-2]
            node_descs = node_descs.reshape(-1, node_descs.shape[-1])

            if routes_tensor.ndim == 2:
                # give the routes tensor and pad mask a batch dim too
                sizes = (batch_size,) + (-1,) * routes_tensor.ndim
                routes_tensor = routes_tensor[None]
                routes_tensor = routes_tensor.expand(sizes)

            if padding_mask.ndim == 2:
                # give the routes tensor and pad mask a batch dim too
                sizes = (batch_size,) + (-1,) * padding_mask.ndim
                padding_mask = padding_mask[None]
                padding_mask = padding_mask.expand(sizes)

        if routes_tensor.ndim > 2:
            old_shape = routes_tensor.shape
            batch_size = old_shape[0]
            if nodes_have_batch:
                # add offsets to each batch element to the right set of nodes
                offsets = torch.arange(0, batch_size * nodes_per_batch, 
                                       nodes_per_batch, 
                                       device=node_descs.device)
                reshape_dims = (-1,) + (1,) * (len(old_shape) - 1)
                routes_tensor = routes_tensor + offsets.reshape(reshape_dims)
            # flatten multiple batch dimensions 
            routes_tensor = routes_tensor.reshape(-1, routes_tensor.shape[-1])
            padding_mask = padding_mask.reshape(-1, routes_tensor.shape[-1])

        # cut out invalid rows (padding mask is true everywhere), then
         # add them back
        routes_tensor = routes_tensor.to(device=node_descs.device)
        padding_mask = padding_mask.to(device=node_descs.device)
        enc_shape = (routes_tensor.shape[0], node_descs.shape[-1])
        valid_seqs = ~padding_mask.all(dim=1)
        # here's the problem, at the below line.
        routes_tensor = routes_tensor[valid_seqs]
        padding_mask = padding_mask[valid_seqs]

        # get node features for routes
        route_node_feats = node_descs[routes_tensor]
        encode_out = self.encode(route_node_feats, padding_mask, **encode_args)
        if type(encode_out) is tuple:
            valid_enc = encode_out[0]
        else:
            valid_enc = encode_out
        enc = torch.zeros(enc_shape, device=node_descs.device)
        enc[valid_seqs] = valid_enc
        # remove the sequence dimension in the seq-len-1 encoding
        enc = enc.squeeze(1)
        if old_shape is not None:
            enc = enc.reshape(old_shape[:-1] + (enc.shape[-1],))
        if type(encode_out) is tuple:
            return (enc,) + encode_out[1:]
        else:
            return enc

    def encode(self, route_seqs, padding_mask):
        raise NotImplementedError

    
class MeanRouteEncoder(RoutesEncoder):
    def encode(self, route_seqs, padding_mask):
        enc = mean_pool_sequence(route_seqs, padding_mask)

        return enc


class MaxRouteEncoder(RoutesEncoder):
    def encode(self, route_seqs, padding_mask):
        unsq_pad = padding_mask.unsqueeze(-1)
        enc = route_seqs * ~unsq_pad + TORCH_FMIN * unsq_pad
        enc, _ = route_seqs.max(dim=1)
        return enc


class LatentAttnRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_heads=8, n_layers=2, 
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        self.encoder = LatentAttentionEncoder(embed_dim, 1, n_heads, n_layers,
                                              dropout)

    def encode(self, route_seqs, padding_mask):
        # TODO somehow also encode the inter-stop times
        enc = self.encoder(route_seqs, padding_mask=padding_mask, 
                           embed_pos=True).squeeze(1)

        return enc


class TransformerRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_heads, n_layers, 
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, 4*embed_dim,
                                               dropout=dropout, 
                                               batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        # self.final_encoder = LatentAttentionEncoder(embed_dim, 1, n_heads, 1,
        #                                             dropout)

    def encode(self, route_seqs, padding_mask, return_nodes=False):
        sinseq = get_sinusoid_pos_embeddings(route_seqs.shape[1], 
                                             route_seqs.shape[2])
        sinseq = sinseq[None]
        route_seqs = route_seqs + sinseq.to(device=route_seqs.device)
        tfed = self.transformer(route_seqs, src_key_padding_mask=padding_mask)
        route_descs = mean_pool_sequence(tfed, padding_mask)
        if return_nodes:
            return route_descs, tfed
        else:
            return route_descs
        # encoding = self.final_encoder(tfed, padding_mask).squeeze(1)
        # return encoding


class ConvRouteEncoder(RoutesEncoder):
    def __init__(self, embed_dim, n_layers):
        super().__init__()
        layers = sum([[nn.Conv1d(embed_dim, embed_dim, 3), 
                       DEFAULT_NONLIN(),
                       nn.AvgPool1d(2)] for _ in range(n_layers)], start=[])
        # applied to a boolean, this should act like "or" over each pooled 
         # block
        self.mask_pooler = nn.MaxPool1d(2**n_layers)
        self.encoder = nn.Sequential(*layers)

    def encode(self, route_seqs, padding_mask):
        # swap sequence and channel dims, since that's what conv1d expects
        route_seqs = route_seqs.permute(0, 2, 1)
        conved = self.encoder(route_seqs)
        # swap sequence and channel dims back
        conved = conved.permute(0, 2, 1)
        
        # compute new padding mask
        valid_mask = (~padding_mask[:, None]).to(dtype=float)
        pooled_valid_mask = self.mask_pooler(valid_mask)
        pooled_valid_mask = pooled_valid_mask[:, 0].to(dtype=bool)
        # due to padding, the shapes might not match perfectly
        pooled_valid_mask = pooled_valid_mask[:, :conved.shape[1]]
        pooled_pad_mask = ~pooled_valid_mask

        # do mean-pooling over pooled sequences
        pooled = mean_pool_sequence(conved, pooled_pad_mask)
        return pooled


class EncodedState:
    def __init__(self, global_desc, route_descs):
        self.global_desc = global_desc
        self.route_desc = route_descs

    @property
    def batch_size(self):
        if self.global_desc.ndim == 2:
            return self.global_desc.shape[0]
        elif self.global_desc.ndim == 1:
            return 1
        else:
            raise ValueError()


class FullGraphEncodedState(EncodedState):
    def __init__(self, global_desc, route_descs, node_descs):
        super().__init__(global_desc, route_descs)
        self.node_descs = node_descs


class BudgetEmbedder(nn.Module):
    def __init__(self, embed_dim, max_budget):
        super().__init__()
        self.budget_embedder = nn.Linear(1, embed_dim)
        self.budget_scale = max_budget
    
    def forward(self, batch_rmng_budgets):
        scaled_budgets = (batch_rmng_budgets * 2 / self.budget_scale) - 1
        if scaled_budgets.ndim == 1:
            # add a feature dimension
            scaled_budgets = scaled_budgets[:, None]
        return self.budget_embedder(scaled_budgets)


class EnvironmentEncoder(nn.Module):
    def __init__(self, stop_node_in_dim, stop_edge_in_dim, dmd_node_in_dim,
                 dmd_edge_in_dim, basin_edge_in_dim, embed_dim, n_layers):
        super().__init__()
        self.embed_dim = embed_dim
        # initial embeddings for nodes and edges
        self.stop_node_embedder = nn.Linear(stop_node_in_dim, embed_dim)
        self.stop_edge_embedder = nn.Linear(stop_edge_in_dim, embed_dim)
        self.dmd_node_embedder = nn.Linear(dmd_node_in_dim, embed_dim)
        self.dmd_edge_embedder = nn.Linear(dmd_edge_in_dim, embed_dim)
        self.basin_edge_embedder = nn.Linear(basin_edge_in_dim, embed_dim)

        # self.net = EdgeGraphNet(n_layers, embed_dim, dropout, recurrent=True)
        self.net = SimplifiedGcn(n_layers, embed_dim)

    def forward(self, env_rep):
        # embed the demand features and "spread" them across demand edges
        stop_node_descs = self.stop_node_embedder(env_rep.stop_data.x)
        stop_edge_descs = self.stop_edge_embedder(env_rep.stop_data.edge_attr)
        dmd_node_descs = self.dmd_node_embedder(env_rep.demand_data.x)
        dmd_edge_descs = self.dmd_edge_embedder(env_rep.demand_data.edge_attr)
        bsn_edge_descs = self.basin_edge_embedder(env_rep.basin_weights)

        num_stops = env_rep.stop_data.num_nodes
        dmd_edge_idx = env_rep.demand_data.edge_index + num_stops
        basin_idxs = env_rep.basin_edges.clone()
        basin_idxs[0] += num_stops
        all_node_descs = torch.cat((stop_node_descs, dmd_node_descs), dim=-2)
        edge_idx = torch.cat((env_rep.stop_data.edge_index, dmd_edge_idx, 
                              basin_idxs, basin_idxs.flip(dims=(0,))), dim=-1)
        edge_attrs = torch.cat((stop_edge_descs, dmd_edge_descs,
                                bsn_edge_descs, bsn_edge_descs))
        
        # right now, no edge features.
        data = Data(x=all_node_descs, edge_index=edge_idx)
        if self.net.gives_edge_features:
            data.edge_attr = edge_attrs
            return self.net(data)
        else:
            node_descs = self.net(data)
            return node_descs, None

    @property
    def edge_key_order(self):
        # TODO route keys aren't used by this model right now
        return (STREET_KEY, DEMAND_KEY)


class FullGraphStateEncoder(nn.Module):
    def __init__(self, stop_node_in_dim, dmd_node_in_dim, route_edge_in_dim, 
                 dmd_edge_in_dim, basin_edge_in_dim, embed_dim, n_heads, 
                 n_layers, max_budget=28200, dropout=MLP_DEFAULT_DROPOUT, 
                 recurrent=True):
        super().__init__()
        self.embed_dim = embed_dim
        # initial embeddings for nodes and edges
        self.stop_node_embedder = nn.Linear(stop_node_in_dim, embed_dim)
        self.dmd_node_embedder = nn.Linear(dmd_node_in_dim, embed_dim)
        self.dmd_edge_embedder = nn.Linear(dmd_edge_in_dim, embed_dim)
        self.route_edge_embedder = nn.Linear(route_edge_in_dim, embed_dim)
        self.basin_edge_embedder = nn.Linear(basin_edge_in_dim, embed_dim)

        self.budget_embedder = BudgetEmbedder(embed_dim, max_budget)
        self.route_gcn = GraphNetBase('edge graph', n_layers, embed_dim, 
                                  dropout=dropout, recurrent=recurrent)

        # learned embedding for the global state when no routes are present
        init_global_desc = nn.Parameter(torch.randn((1, embed_dim)))
        self.register_parameter(name="init_global_desc", 
                                param=init_global_desc)

    def forward(self, env_rep, batch_route_reps, batch_rmng_budgets):
        """
        route_edge_indices: a n_routes list of 2 x route_len tensors
        batch_route_reps: a n_routes list of route_len x route_dim tensors
        batch_rmng_budgets: a tensor of remaning budgets for the batch

        Returns:
        -- a list of route descriptor tensors
        -- a tensor of demand edge descriptors
        """
        # embed the demand features and "spread" them across demand edges
        demand_data = copy.copy(env_rep.demand_data)
        demand_nodes = self.dmd_node_embedder(demand_data.x)
        dmd_edge_feats = self.dmd_edge_embedder(demand_data.edge_attr)
        device = env_rep.stop_data.x.device

        # embed the stop nodes
        stop_nodes = self.stop_node_embedder(env_rep.stop_data.x)
        all_node_feats = torch.cat((stop_nodes, demand_nodes), dim=-2)

        dmd_edge_idx = demand_data.edge_index + env_rep.stop_data.num_nodes
        datas = []

        log.debug("assembling graph batch")
        for route_reps in batch_route_reps:
            # assemble all route and basin edges into a single edge collection
            if len(route_reps) == 0:
                all_edge_idxs = dmd_edge_idx
                all_edge_feats = dmd_edge_feats
            else:
                cat_route_edge_indices = torch.cat(
                    [rr.edge_index for rr in route_reps], dim=1)
                # select only those basin edges connecting to stops on routes
                used_stops = cat_route_edge_indices.unique()
                basin_idxs = env_rep.basin_edges.clone()
                is_on_route = \
                    (basin_idxs[1][None] == used_stops[:, None]).any(dim=0)
                basin_idxs = basin_idxs[:, is_on_route]
                basin_idxs[0] += env_rep.stop_data.num_nodes
                all_edge_idxs = \
                    torch.cat((cat_route_edge_indices, basin_idxs, 
                               basin_idxs.flip(dims=(0,)), dmd_edge_idx), 
                               dim=1)
                # embed the edge features
                routes_edge_in_feats = [rr.edge_attr for rr in route_reps]
                scen_edge_feats = torch.cat(routes_edge_in_feats, dim=0)
                scen_edge_feats = self.route_edge_embedder(scen_edge_feats)
                basin_weights = env_rep.basin_weights[is_on_route]
                basin_feats = self.basin_edge_embedder(basin_weights)
                all_edge_feats = \
                    torch.cat((scen_edge_feats, basin_feats, basin_feats,
                            dmd_edge_feats), dim=0)

            # combine edges into a single data/batch object
            data = Data(x=all_node_feats, edge_index=all_edge_idxs,
                        edge_attr=all_edge_feats)
            datas.append(data)
        
        batch = Batch.from_data_list(datas)
        # - apply the edge graph network to get node features and edge features
        log.debug("running graph batch through GCN")
        out_node_feats, out_edge_feats = self.route_gcn(batch)

        # aggregate route features into route descriptors
        log.debug("aggregating GCN outputs")
        batch_size = len(batch_route_reps)
        max_n_routes = max([len(rr) for rr in batch_route_reps])
        route_descs = torch.zeros((batch_size, max_n_routes, self.embed_dim),
                                  device=device)                                  
        global_desc = torch.zeros((batch_size, self.embed_dim), device=device)
        be_idx = 0
        for bi, route_reps in enumerate(batch_route_reps):
            if len(route_reps) == 0:
                global_desc[bi] = self.init_global_desc
                
            # stop_feats = []
            n_batch_edges = datas[bi].num_edges
            scen_edge_feats = out_edge_feats[be_idx:be_idx + n_batch_edges]
            re_idx = 0
            for ri, route_rep in enumerate(route_reps):
                n_edges = route_rep.edge_index.shape[1]
                route_edge_feats = scen_edge_feats[re_idx:re_idx + n_edges]
                global_desc[bi] += route_edge_feats.sum(dim=-2)
                route_descs[bi, ri] = route_edge_feats.mean(dim=-2)
                re_idx += n_edges

                # # compute the per-route-stop features
                # edge_idx = route_rep.edge_index
                # is_from = edge_idx[0][:, None] == route_rep.route[None]
                # is_to = edge_idx[1][:, None] == route_rep.route[None]
                # is_on_stop = is_from | is_to
                # route_stop_feats = \
                #     (route_rep.edge_attr[:, None] * is_on_stop[..., None]).mean(dim=0)
                # stop_feats.append(route_stop_feats)
            be_idx += n_batch_edges

        global_desc /= env_rep.stop_data.num_nodes

        global_desc = global_desc + self.budget_embedder(batch_rmng_budgets)

        log.debug("done encoder forward pass")

        # aggregate route edge features into a descriptor for each route
        num_nodes = stop_nodes.shape[0] + demand_nodes.shape[0]
        out_node_feats = \
            out_node_feats.reshape((batch_size, num_nodes, -1))
        node_descs = out_node_feats[:, :env_rep.stop_data.num_nodes]
        return FullGraphEncodedState(global_desc, route_descs, node_descs)


class TransformerEncoder(nn.Module):
    # TODO this is out-of-step with the current encoder interface
    """This class represents an encoder for a whole transit system."""
    def __init__(self, env_rep, embed_dim, n_heads, n_route_layers, 
                 n_ctxt_layers, dropout=ATTN_DEFAULT_DROPOUT, min_cost=0, 
                 max_cost=273):
        super().__init__()
        self.min_cost = min_cost
        self.max_cost = max_cost
        self.graph_encoder = \
            DualGcnEncoder(env_rep, embed_dim, MLP_DEFAULT_DROPOUT)

        # self.route_encoder = TransformerRouteEncoder(embed_dim, n_heads,
        #                                              n_route_layers, dropout)
        self.route_encoder = LatentAttnRouteEncoder(embed_dim, n_heads, 
                                                    n_route_layers, dropout)
        # self.route_encoder = ConvRouteEncoder(embed_dim, n_route_layers)
        
        hidden_dim = 4 * embed_dim
        self.cost_embedder = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.Dropout(MLP_DEFAULT_DROPOUT),
            DEFAULT_NONLIN(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(MLP_DEFAULT_DROPOUT),
            DEFAULT_NONLIN(),
            nn.Linear(hidden_dim, embed_dim)
        )        

        tf_layer = nn.TransformerEncoderLayer(embed_dim, n_heads, hidden_dim,
                                              dropout, batch_first=True)
        self.context_encoder = nn.TransformerEncoder(tf_layer, n_ctxt_layers)

        self.global_aggr = LatentAttentionEncoder(embed_dim, n_heads=n_heads,
                                                  n_layers=1, dropout=dropout)

        # a learned placeholder for the empty state, when no routes exist
        init_state = nn.Parameter(torch.randn(1, embed_dim))
        self.register_parameter(name="init_state", 
            param=init_state)

    def forward(self, env_rep, batch_routes_tensor, batch_costs, 
                route_pad_mask, scenario_pad_mask):
        # compute node embeddings over the graph
        node_descs = self.encode_nodes(env_rep)
        route_descs = self.encode_routes(node_descs, batch_routes_tensor, 
                                         batch_costs, route_pad_mask)
        ctxt_route_descs, global_desc = \
            self.get_global_info_descs(route_descs, scenario_pad_mask)
        return ctxt_route_descs, global_desc

    def encode_nodes(self, env_rep):
        # just the node features, not the demand features
        return self.graph_encoder(env_rep)[0]

    def encode_routes(self, node_descs, batch_routes_tensor, batch_costs, 
                      route_pad_mask=None):
        # compute individual embeddings for each route
        route_descs = \
            self.route_encoder(node_descs, batch_routes_tensor, route_pad_mask)
            
        # bring costs into range [-1, 1]
        center = (self.max_cost + self.min_cost) / 2
        norm = (self.max_cost - self.min_cost) / 2
        batch_costs = (batch_costs - center) / norm

        # embed the route costs in the route descriptions
        if batch_routes_tensor.ndim == 3:
            batch_size = batch_routes_tensor.shape[0]
        else:
            batch_size = 1

        if batch_costs.ndim == 2 and batch_costs.shape[0] > batch_size:
            # expand the route descs for the cost embedding
            batch_size = batch_costs.shape[0]
            route_descs = route_descs.expand(batch_size, -1, -1)

        # add a feature dimension
        batch_costs = batch_costs[..., None]
        cost_cat = torch.cat((route_descs, batch_costs), dim=-1)
        route_descs_w_cost = self.cost_embedder(cost_cat)
        return route_descs_w_cost

    def get_global_info_descs(self, route_descs, scenario_pad_mask):
        """Compute a global descriptor, and route descriptors with global
            context."""
        is_empty_scenario = scenario_pad_mask.all(dim=1)
        if is_empty_scenario.all():
            batch_is = self.init_state.expand(len(is_empty_scenario), -1)
            return route_descs, batch_is
        valid_route_descs = route_descs[~is_empty_scenario]
        valid_spm = scenario_pad_mask[~is_empty_scenario]
        valid_ctxt_route_descs = self.context_encoder(valid_route_descs, 
            src_key_padding_mask=valid_spm)
        ctxt_route_descs = torch.zeros(route_descs.shape, 
                                       device=route_descs.device)
        ctxt_route_descs[~is_empty_scenario] = valid_ctxt_route_descs
        # global_desc = self.global_aggr(ctxt_route_descs, scenario_pad_mask)
        # global_desc = mean_pool_sequence(ctxt_route_descs, scenario_pad_mask)
        global_desc = ctxt_route_descs.sum(dim=-2) / 100
        global_desc[is_empty_scenario] = self.init_state
        return ctxt_route_descs, global_desc.squeeze(1)


class NodeSumActionEncoder(nn.Module):
    def __init__(self, embed_dim, dropout=MLP_DEFAULT_DROPOUT):
        super().__init__()
        self.embed_dim = embed_dim
        self.route_cost_embedder = nn.Sequential(
            nn.Linear(embed_dim + 1, embed_dim),
            nn.Dropout(dropout),
            DEFAULT_NONLIN()
        )

    def forward(self, encoded_state, route_reps):
        """
        encoded_state: an encoded_state object
        route_reps: list of stop-node-idx sequences, or batch_size list
            of lists (in this case, all lists must represent the same 
            collection of routes, but with potentially different 
            representations)
        """
        batch_encoded_nodes = encoded_state.node_descs
        if batch_encoded_nodes.ndim == 2:
            batch_encoded_nodes = batch_encoded_nodes[None]

        batch_size = batch_encoded_nodes.shape[0]
        if type(route_reps[0]) is RouteRep:
            route_reps = [route_reps] * batch_size

        routes = [rr.route for rr in route_reps[0]]
        routes_tensor, _ = get_tensor_from_varlen_lists(routes)
        node_seqs = batch_encoded_nodes[:, routes_tensor]        
        seq_aggr = node_seqs.sum(dim=-2) / 100

        costs = torch.tensor([[rr.norm_cost for rr in rrs]
                              for rrs in route_reps], device=node_seqs.device)
        route_ins = torch.cat((seq_aggr, costs[..., None]), dim=-1)
        return self.route_cost_embedder(route_ins)
        

class NodeSeqActionEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads, n_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.route_cost_embedder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.Dropout(MLP_DEFAULT_DROPOUT),
            DEFAULT_NONLIN()
        )
        # self.inner_enc = ConvRouteEncoder(embed_dim, n_layers)
        self.inner_enc = LatentAttnRouteEncoder(embed_dim, n_heads, n_layers,
                                                ATTN_DEFAULT_DROPOUT)
        
    def forward(self, encoded_state, route_reps):
        """
        encoded_state: an encoded_state object
        route_reps: list of stop-node-idx sequences, or batch_size list
            of lists (in this case, all lists must represent the same 
            collection of routes, but with potentially different 
            representations)
        """
        log.debug("encoding actions")
        batch_encoded_nodes = encoded_state.node_descs
        if batch_encoded_nodes.ndim == 2:
            batch_encoded_nodes = batch_encoded_nodes[None]

        batch_size = batch_encoded_nodes.shape[0]
        if type(route_reps[0]) is RouteRep:
            route_reps = [route_reps] * batch_size

        routes = [rr.route for rr in route_reps[0]]
        routes_tensor, pad_mask = get_tensor_from_varlen_lists(routes)

        dev = batch_encoded_nodes.device
        enc = self.inner_enc(batch_encoded_nodes, routes_tensor, pad_mask)
        costs = torch.tensor([[rr.norm_cost for rr in rrs]
                              for rrs in route_reps], device=dev)
        enc += self.route_cost_embedder(costs[..., None])
        return enc


class Encoder(nn.Module):
    def __init__(self, embed_dim, state_encoder, action_encoder):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder

    # TODO do we need a 'forward'?
    def encode_state(self, env_rep, batch_route_reps, batch_rmng_budgets):
        return self.state_encoder(env_rep, batch_route_reps, 
                                  batch_rmng_budgets)

    def encode_actions(self, state_encoding, action_route_reps):
        return self.action_encoder(state_encoding, action_route_reps)

    def load_state_encoder_weights(self, weights_path, freeze=True):
        self.state_encoder.load_state_dict(torch.load(weights_path))
        if freeze:
            for param in self.state_encoder.parameters():
                param.requires_grad_(False)


class SimplestEncoder(Encoder):
    def __init__(self, n_routes, embed_dim, n_layers, max_budget=28200,
                 dropout=MLP_DEFAULT_DROPOUT):
        super().__init__(embed_dim, None, None)
        self.budget_embedder = BudgetEmbedder(embed_dim, max_budget)
        self.nonlin = DEFAULT_NONLIN()
        self.fwd_embedder = get_mlp(n_layers, embed_dim, dropout, 
                                    in_dim=n_routes)
        self.n_routes = n_routes

    def encode_state(self, env_rep, batch_state_routereps, batch_rmng_budgets):
        batch_size = len(batch_state_routereps)
        dev = env_rep.stop_data.x.device
        # TODO why -2 here?
        in_tensor = torch.full((batch_size, self.n_routes), -2, device=dev)
        for bi, route_reps in enumerate(batch_state_routereps):
            for route_rep in route_reps:
                in_tensor[bi, route_rep.route_idx] = 1
        enc = self.fwd_embedder(in_tensor)
        enc = enc + self.budget_embedder(batch_rmng_budgets)

        return EncodedState(enc, in_tensor)
                
    def encode_actions(self, batch_state_encoding, action_route_reps):
        batch_size = batch_state_encoding.batch_size
        if type(action_route_reps[0]) is RouteRep:
            action_route_reps = [action_route_reps] * batch_size

        dev = batch_state_encoding.global_desc.device
        in_tensor = batch_state_encoding.route_desc.clone()
        act_enc = torch.eye(self.n_routes, device=dev, dtype=bool)
        batch_act_enc = in_tensor[:, None, :].repeat(1, self.n_routes, 1)
        batch_act_enc[:, act_enc] = 2

        act_idxs = [rr.route_idx for rr in action_route_reps[0]]
        batch_act_enc = batch_act_enc[:, act_idxs, :]
        return self.fwd_embedder(batch_act_enc)


class SimpleEncoder(Encoder):
    def __init__(self, n_stops, embed_dim, n_route_layers, n_global_layers,
                 max_budget, dropout=MLP_DEFAULT_DROPOUT):
        super().__init__(embed_dim, None, None)
        self.n_stops = n_stops
        self.nonlin = DEFAULT_NONLIN()
        self.budget_embedder = BudgetEmbedder(embed_dim, max_budget)
        self.route_embedder = get_mlp(n_route_layers, embed_dim, dropout,
                                      in_dim=n_stops)
        self.global_projector = get_mlp(n_global_layers, embed_dim, dropout)
        self.action_embedder = get_mlp(n_global_layers, embed_dim, dropout,
            in_dim=embed_dim*2)

        # learned embedding for the global state when no routes are present
        init_global_desc = nn.Parameter(torch.randn((1, embed_dim)))
        self.register_parameter(name="init_global_desc", 
                                param=init_global_desc)

    def encode_state(self, env_rep, batch_route_reps, batch_rmng_budgets):
        budget_enc = self.budget_embedder(batch_rmng_budgets)
        if len(batch_route_reps[0]) == 0:
            route_descs = torch.zeros((0, self.embed_dim), 
                                      device=self.init_global_desc.device)
            batch_size = len(batch_route_reps)
            global_desc = self.init_global_desc.expand((batch_size, -1))
            return EncodedState(global_desc + budget_enc, route_descs)

        one_hot_routes, pad_mask = \
            self._encode_one_hot_routes(batch_route_reps)
        route_descs = self.route_embedder(one_hot_routes)
        route_descs[pad_mask] = 0

        global_desc = route_descs.sum(dim=-2) / 100
        return EncodedState(global_desc + budget_enc, route_descs)

    def encode_actions(self, state_encoding, action_route_reps):
        if type(action_route_reps[0]) is RouteRep:
            one_hot_routes, pad_mask = \
                self._encode_one_hot_routes(action_route_reps)
        else:
            one_hot_routes, pad_mask = \
                self._encode_one_hot_routes(action_route_reps[0])

        route_descs = self.route_embedder(one_hot_routes)
        route_descs[pad_mask] = 0
        if state_encoding.batch_size > 1 and route_descs.shape[0] == 1:
            new_shape = \
                (state_encoding.batch_size,) + (-1,) * (route_descs.ndim - 1)
            route_descs = route_descs.expand(new_shape)

        global_desc = state_encoding.global_desc
        if global_desc.ndim == 1:
            # add a batch dimension
            global_desc = global_desc[None]

        max_n_routes = pad_mask.shape[-1]
        tiled_glbl = global_desc[:, None, :].repeat(1, max_n_routes, 1)
        route_head_in = torch.cat((tiled_glbl, route_descs), dim=-1)
        action_descs = self.action_embedder(route_head_in)
        return action_descs

    def _encode_one_hot_routes(self, batch_route_reps):
        if type(batch_route_reps[0]) is RouteRep:
            # add a batch list "wrapper"
            batch_route_reps = [batch_route_reps]
        dev = batch_route_reps[0][0].device
        max_n_routes = max([len(rrs) for rrs in batch_route_reps])
        batch_size = len(batch_route_reps)
        one_hot_routes = \
            torch.full((batch_size, max_n_routes, self.n_stops), -1,
                       device=dev)
        pad_mask = torch.zeros(one_hot_routes.shape[:-1], device=dev,
                               dtype=bool)
        for si, scenario_route_reps in enumerate(batch_route_reps):
            pad_mask[si, len(scenario_route_reps):] = True
            for ri, route_rep in enumerate(scenario_route_reps):
                one_hot_routes[si, ri, route_rep.route] = 1

        return one_hot_routes, pad_mask


class SequentialSelector(nn.Module):
    def __init__(self, embed_dim, max_seq_len, softmax_temp):
        super().__init__()
        self.embed_dim = embed_dim
        self.scorer = KoolNextNodeScorer(embed_dim)
        self.softmax_temp = softmax_temp
        if max_seq_len is None:
            self.max_seq_len = float("inf")
        else:
            self.max_seq_len = max_seq_len

        self.max_seq_len = max_seq_len
        bl_hidden_dim = embed_dim
        self.value_est_net = nn.Sequential(
            # value_net_backbone
            nn.Linear(embed_dim, bl_hidden_dim),
            DEFAULT_NONLIN(),
            nn.Linear(bl_hidden_dim, bl_hidden_dim),
            DEFAULT_NONLIN(),
            nn.Linear(bl_hidden_dim, 1)
        )
        # TODO why bias=False here?  Try without.
        self.fixed_context_projector = \
            nn.Linear(embed_dim, embed_dim, bias=False)

        # a learned placeholder for ending the sequence
        halt_placeholder = nn.Parameter(torch.randn(1, embed_dim))
        self.register_parameter(name="halt_placeholder", 
            param=halt_placeholder)

    def reset(self):
        self.scorer.reset()

    def select(self, context, mask, greedy=False, rollout=None):
        scores = self.scorer(context, mask.clone().detach())

        # bias the halt action to have a-priori probability 1 / max seq len
        n_actions = (~mask).sum(dim=-1)
        length_mods = torch.zeros(scores.shape, device=scores.device)
        length_mods[:, -1] = torch.log(n_actions / self.max_seq_len)
        scores += length_mods

        # perform the selection
        selections, logits = select(scores, greedy, self.softmax_temp, rollout)
        est_val = self.value_est_net(context)
        return selections, logits, est_val


class DemandBasedRouteGenerator(SequentialSelector):
    def __init__(self,
                 embed_dim,
                 softmax_temp=1,
                 max_n_route_stops=100, 
                 n_attn_heads=8,
                 dropout=ATTN_DEFAULT_DROPOUT,
                 ):
        super().__init__(embed_dim, max_n_route_stops, softmax_temp)
        # a learned placeholder for ending the route
        route_end_placeholder = nn.Parameter(torch.randn(1, embed_dim))
        self.register_parameter(name="route_end_placeholder", 
                                param=route_end_placeholder)
        first_choice_ctxt = nn.Parameter(torch.randn(1, embed_dim))
        self.register_parameter(name="first_choice_ctxt", 
                                param=first_choice_ctxt)
        second_choice_ctxt = nn.Parameter(torch.randn(1, embed_dim))
        self.register_parameter(name="second_choice_ctxt", 
                                param=second_choice_ctxt)

        self.route_tf = nn.TransformerEncoderLayer(embed_dim, n_attn_heads, 
                                                   embed_dim, batch_first=True)

        # a learned placeholder for ending the route
        route_desc_attn = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.register_parameter(name="route_desc_attn", param=route_desc_attn)

        self.seed_projector = nn.Linear(embed_dim, embed_dim, bias=False)

        # TODO why bias=False here?  Try without.
        self.context_attn = \
            nn.MultiheadAttention(embed_dim, n_attn_heads, dropout, bias=False,
                                  batch_first=True)
        self.stop_pos_attn = \
            nn.MultiheadAttention(embed_dim, n_attn_heads, dropout, 
                                  bias=False, batch_first=True, kdim=embed_dim,
                                  vdim=embed_dim)
        self.stop_pos_scorer = nn.Sequential(
            DEFAULT_NONLIN(),
            nn.Linear(embed_dim, 1)
        )

        self.sinseq = get_sinusoid_pos_embeddings(max_n_route_stops, embed_dim)

    def precompute(self, embedded_nodes, env_rep):
        # add the "halt" choice
        self.choices_tnsr = torch.cat((embedded_nodes, self.halt_placeholder))
        self.scorer.precompute(self.choices_tnsr)
        n_choices = self.choices_tnsr.shape[0]
        self.inter_node_demands = torch.zeros((n_choices, n_choices), 
                                              dtype=bool, 
                                              device=embedded_nodes.device)
        self.inter_node_demands[:-1, :-1] = env_rep.inter_node_demands
        avg_node_vec = embedded_nodes.mean(dim=-2).squeeze()
        self.fixed_context = self.fixed_context_projector(avg_node_vec)[None]
        if env_rep.stop_adj_mat.is_sparse:
            self.node_adjacencies = env_rep.stop_adj_mat.to_dense() > 0
        else:
            self.node_adjacencies = env_rep.stop_adj_mat > 0


    def reset(self):
        super().reset()
        self.choices_tnsr = None
        self.fixed_context = None

    @property
    def n_nodes(self):
        return self.choices_tnsr.shape[0] - 1

    @property
    def halt_idx(self):
        return self.n_nodes

    @property
    def n_actions(self):
        return self.choices_tnsr.shape[0]

    @property
    def device(self):
        return self.choices_tnsr.device

    def forward(self, greedy=False, batch_size=1, seeds=None, **kwargs):
        """
        greedy: If false, pick nodes randomly weighted by their
            scores.  If true, pick the node with the highest score.
        """
        # check that combination of arguments is valid
        if seeds is not None:
            # incorporate the seeds in the fixed context
            assert batch_size == seeds.shape[0]
            proj_seeds = self.seed_projector(seeds)
            fixed_context = self.fixed_context + proj_seeds
        else:
            fixed_context = self.fixed_context
        fixed_context = fixed_context.expand(batch_size, -1)

        # objects that will be used during the selection loop
        dev = self.device
        route_desc_attn = expand_batch_dim(self.route_desc_attn, batch_size)
        n_outgoing_edges = self.inter_node_demands.sum(dim=-1)

        # storage for intermediates and outputs of the selection loop
        batch_routes = [[] for _ in range(batch_size)]
        selections = []
        slct_logits = []
        pos_logits = []
        mask = torch.zeros((batch_size, self.n_actions), dtype=bool, 
                           device=dev)
        # don't pick edges with no outgoing demand
        mask[:] = n_outgoing_edges == 0
        # require each route to have at least one pair of nodes
        mask[:, -1] = True
        log.info("generating route loop")
        
        while not torch.all(mask[:, :-1]):
            # build a tensor of the nodes in the route, plus the route-end
            routes_tnsr, routes_mask = self.build_routes_tensor(batch_routes)
            routes_tnsr += self.route_tf(routes_tnsr,
                                         src_key_padding_mask=routes_mask)

            # compute the step context from the existing routes.
            step_context, _ = \
                self.context_attn(route_desc_attn, routes_tnsr, routes_tnsr,
                                  key_padding_mask=routes_mask)
            # remove the "sequence" dimension and add to fixed context
            context = fixed_context + step_context.squeeze(1)
            if len(batch_routes[0]) == 0:
                # the first step is special, since we're picking from all nodes
                context += self.first_choice_ctxt
            elif len(batch_routes[0]) == 1:
                # the second step is also special: pick from all nodes with
                 # demand to or from the first node
                context += self.second_choice_ctxt

            log.debug("selecting next node")
            selection, logit, _ = self.select(context, mask, greedy)
            selections.append(selection)
            slct_logits.append(logit)
                        
            # determine the locations where each stop will be inserted
            # run multi-head attention between the chosen node feature and the
             # route tensor, giving a feature for each node in the route.
            log.debug("picking stop position for node")
            node_desc = self.choices_tnsr[selection][:, None]
            # TODO make this another Kool scorer?  Would have to reset it each time.
            node_v_route_enc, _ = \
                self.stop_pos_attn(node_desc, routes_tnsr, routes_tnsr,
                                   key_padding_mask=routes_mask)
            # compute a score for inserting before each node in the route
            stop_pos_input = routes_tnsr + node_v_route_enc
            stop_pos_scores = self.stop_pos_scorer(stop_pos_input).squeeze(-1)

            # choose the insertion position for each route's new node
            step_pos_logits = torch.zeros(batch_size, device=dev)
            for ri, route in enumerate(batch_routes):
                pos_logit = 0
                ni = selection[ri].item()
                if len(route) == 0:
                    route.append(ni)
                elif ni != self.halt_idx:
                    scores = stop_pos_scores[ri]
                    # mask out scores for invalid positions
                    vpm = valid_pos_matrices[ri][:, ni]
                    scores[torch.where(~vpm)] = TORCH_FMIN
                    scores[len(route) + 1:] = TORCH_FMIN
                    # choose the insertion position
                    pos, pos_logit = select(scores, greedy, self.softmax_temp)
                    route.insert(pos, ni)
                
                step_pos_logits[ri] = pos_logit
            pos_logits.append(step_pos_logits)
                        
            log.debug("updating mask")
            # mask out nodes already on the route
            valid_pos_matrices = []
            for ri, route in enumerate(batch_routes):
                if len(route) == 1:
                    # ignore the distance mask when picking the second node
                    vpm = self._get_valid_insertion_matrix(route, False)
                else:
                    vpm = self._get_valid_insertion_matrix(route)
                valid_pos_matrices.append(vpm)
                # mask in all nodes that have a valid insertion location
                mask[ri, :-1] = ~(vpm.any(dim=0))
                # mask out all nodes that are already on the route
                mask[ri, route] = True

            # mask out all nodes in batch elems where halt was chosen
            mask[selection == self.halt_idx, :-1] = True
            # mask out all nodes where max route length is reached
            route_lens = torch.tensor([len(rr) for rr in batch_routes],
                                      device=dev)
            mask[route_lens == self.max_seq_len, :-1] = True
            # unmask the halt action when route has 2 or more nodes
            mask[route_lens > 1, -1] = False

        log.info("Route-gen loop is done. Assembling the output logits")
        out_logits = torch.zeros((batch_size, self.max_seq_len), device=dev)
        halted = [False for _ in batch_routes]
        # could speed this up by building a reverse dict for each route
        for sl, ss, pl in zip(slct_logits, selections, pos_logits):
            for ri, route in enumerate(batch_routes):
                if ss[ri] == self.halt_idx:
                    # it's the halt choice
                    if not halted[ri] and len(route) < self.max_seq_len:
                        # first time halt was chosen, so assign the halt logit
                        out_logits[ri, len(route)] = sl[ri]
                        halted[ri] = True
                else:
                    pos = route.index(ss[ri])
                    # assign the sum of the node and position logits
                    out_logits[ri, pos] = sl[ri] + pl[ri]

        # compute the route descriptors
        routes_tnsr, routes_mask = self.build_routes_tensor(batch_routes)
        routes_tnsr += self.route_tf(routes_tnsr, 
                                     src_key_padding_mask=routes_mask)
    
        route_descs, _ = \
            self.context_attn(route_desc_attn, routes_tnsr, routes_tnsr,
                              key_padding_mask=routes_mask)
        # remove the "sequence" dimension
        route_descs.squeeze(1)

        # return the routes, route-order stop logits, and route descriptors
        return RouteGenResults(routes=batch_routes, route_descs=route_descs,
                               logits=out_logits, 
                               selections=None, est_vals=None)

    def build_routes_tensor(self, batch_routes):
        # build a tensor of the nodes in the route, plus the route-end
        device = self.choices_tnsr.device
        routes_tnsr = torch.zeros((len(batch_routes), self.max_seq_len,
                                   self.embed_dim), device=device)
        # mask out the placeholder locations in the route tensor
        routes_mask = torch.ones((len(batch_routes), self.max_seq_len),
                                  dtype=bool, device=device)
        for bi, route in enumerate(batch_routes):
            routes_tnsr[bi, :len(route)] = self.choices_tnsr[route]
            if len(route) < self.max_seq_len:
                # we only need the end placeholder if the route isn't max len
                routes_tnsr[bi, len(route)] = self.route_end_placeholder
            routes_mask[bi, :len(route) + 1] = False
        # add sinusoidal sequence info
        routes_tnsr += self.sinseq.to(device)
        return routes_tnsr, routes_mask

    def _get_valid_insertion_matrix(self, route, use_adjacency=True):
        """Given a route with R stops in an environment with N nodes, this
        function returns a R x N binary matrix V, such that V[i,j] is True 
        if i is a valid location on the route to insert node j."""
        valid_poss = torch.zeros((len(route) + 1, self.n_nodes), 
                                  dtype=bool, device=self.device)
        after_first_dm_to_nodes = valid_poss.clone()
        before_last_dm_from_nodes = valid_poss.clone()

        dm_to_nodes = self.inter_node_demands[route, :-1]
        after_first_dm_to_nodes[1:] = (dm_to_nodes.cumsum(dim=0) > 0)
        valid_poss[after_first_dm_to_nodes] = True

        rvs_dm_from_nodes = self.inter_node_demands[:-1, route[::-1]].T
        rvs_cum_dm_from_nodes = rvs_dm_from_nodes.cumsum(dim=0) > 0
        before_last_dm_from_nodes[:-1] = rvs_cum_dm_from_nodes.flip(0)
        valid_poss[before_last_dm_from_nodes] = True
        
        if use_adjacency:
            nodes_inrange_before = torch.zeros(valid_poss.shape, dtype=bool,
                                               device=self.device)
            nodes_inrange_before[:len(route)] = \
                self.node_adjacencies[:, route].T
            nodes_inrange_after = self.node_adjacencies[route]
            nodes_inrange_before[1:len(route) + 1] |= nodes_inrange_after
            valid_poss &= nodes_inrange_before

        return valid_poss


class DummyPregenRouteReturner(nn.Module):
    def __init__(self, routes) -> None:
        super().__init__()
        self.routes = routes
        max_len_route = max([len(route) for route in self.routes])
        routes_shape = (len(self.routes), max_len_route)
        self.routes_tensor = torch.zeros(routes_shape, dtype=int)
        self.routes_pad_mask = torch.zeros(routes_shape, dtype=bool)
        for ri, route in enumerate(routes):
            self.routes_tensor[ri, :len(route)] = torch.tensor(route)
            self.routes_pad_mask[ri, len(route):] = True

    def forward(self, *args, **kwargs):
        # encode the routes
        return RouteGenResults(routes=self.routes, logits=None,
                               route_descs=None, selections=None, 
                               est_vals=None)
    
    def reset(self):
        self.route_descs = None
    
    def precompute(self, *args, **kwargs):
        pass


class ContinuousGaussianActor(nn.Module):
    def __init__(self, n_layers, embed_dim, in_dim=None,
                 min_action=None, max_action=None, min_logstd=None, 
                 max_logstd=None, bias=True):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        self.min_logstd = min_logstd
        self.max_logstd = max_logstd
        self.mean_and_std_net = get_mlp(n_layers, embed_dim, in_dim=in_dim,
                                        out_dim=2, bias=bias)

    def forward(self, inpt, pick_max=False):
        means_and_stds = self.mean_and_std_net(inpt)
        means = means_and_stds[..., 0]
        if self.min_action is not None or self.max_action is not None:
            means = means.clamp(self.min_action, self.max_action)

        # stds must be positive, so exponentiate them
        logstds = means_and_stds[..., 1]
        if self.min_logstd is not None or self.max_logstd is not None:
            logstds = logstds.clamp(self.min_logstd, self.max_logstd)
        stds = torch.exp(logstds)

        if self.min_action is not None or self.max_action is not None:
            dstrb = TruncatedNormal(means, stds, self.min_action, 
                                    self.max_action)
        else:
            dstrb = torch.distributions.Normal(means, stds)

        if pick_max:
            # take the action at the mean of the distribution
            actions = means
        else:
            # sample from the distribution and limit to valid range
            actions = dstrb.sample()

        log_probs = dstrb.log_prob(actions)
        return actions, log_probs


class ContinuousGaussFixedStdActor(nn.Module):
    def __init__(self, n_layers, embed_dim, in_dim=None,
                 min_action=None, max_action=None, fixed_std=None, bias=True):
        super().__init__()
        self.min_action = min_action
        self.max_action = max_action
        if fixed_std is None:
            if min_action is not None and max_action is not None:
                self.fixed_std = (max_action - min_action) / 5
            else:
                raise ValueError(
                    "Must provide fixed_std or min_action and max_action")
        else:
            self.fixed_std = fixed_std
            
        self.mean_net = get_mlp(n_layers, embed_dim, in_dim=in_dim, out_dim=1, 
                                bias=bias)

    def forward(self, inpt, greedy=False):
        means = self.mean_net(inpt).squeeze(-1)
        means = means.clamp(self.min_action, self.max_action)
        stds = torch.full_like(means, self.fixed_std)
    
        dstrb = TruncatedNormal(means, stds, self.min_action, self.max_action)
        if greedy:
            # take the action at the mean of the distribution
            actions = means
        else:
            # sample from the distribution and limit to valid range
            actions = dstrb.sample()

        log_probs = dstrb.log_prob(actions)
        return actions, log_probs, stds


class FrequencySelector(nn.Module):
    def __init__(self, embed_dim, min_frequency_Hz=1/7200, 
                 max_frequency_Hz=1/60):
        """default min_frequency_Hz is one bus every two hours.
           default max_frequency_Hz is one bus every minute."""
        super().__init__()
        self.mean_and_std_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            DEFAULT_NONLIN(),
            nn.Linear(embed_dim, embed_dim),
            DEFAULT_NONLIN(),
            nn.Linear(embed_dim, 2)
        )

        self.freq_embedder = nn.Sequential(
            nn.Linear(1, embed_dim),
            DEFAULT_NONLIN()
        )
        assert min_frequency_Hz < max_frequency_Hz
        # approximately 1.44 Hz
        self.min_logstd = -1
        self.min_action = self._freq_Hz_to_action(min_frequency_Hz)
        self.max_action = self._freq_Hz_to_action(max_frequency_Hz)
        # wide enough to give a uniform distribution
        self.max_logstd = np.log((self.max_action - self.min_action) * 10000)


    def forward(self, route_descriptor, greedy=False, rollout_freqs=None,
                min_frequency_Hz=None, max_frequency_Hz=None):
        assert not greedy or rollout_freqs is None, \
            "greedy frequency selection is incompatible with rollout!"
        # compute the parameters of the gaussians from which to sample freqs
        normal_param_tensor = self.mean_and_std_net(route_descriptor)
        # stop the mean from being lower than the minimum allowed action
        if min_frequency_Hz:
            min_action = self._freq_Hz_to_action(min_frequency_Hz)
        else:
            min_action = self.min_action
        if max_frequency_Hz:
            max_action = self._freq_Hz_to_action(max_frequency_Hz)
        else:
            max_action = self.max_action
        assert min_action < max_action

        means = normal_param_tensor[..., 0].clamp(min_action, max_action)
        # clamp the std devs in a sensible range for numerical stability
        logstds = normal_param_tensor[..., 1].clamp(self.min_logstd, 
                                                    self.max_logstd)
        # stds must be positive, so exponentiate them
        stds = torch.exp(logstds)
        dstrb = torch.distributions.Normal(means, stds)
        if greedy:
            # greedily take the action at the center of the distribution
            actions = means
        elif rollout_freqs is None:
            # sample the action from the distribution
            actions = dstrb.sample()
        else:
            # take the dictated action
            actions = self._freq_Hz_to_action(rollout_freqs)

        # enforce the minimum allowable frequency for a route that's included        
        actions.clamp_(self.min_action, self.max_action)
        log_probs = dstrb.log_prob(actions)

        freqs_Hz = self._action_to_freq_Hz(actions)
        descriptors = self.freq_embedder(actions[:, None])
        return freqs_Hz, log_probs, descriptors

    def _freq_Hz_to_action(self, freq_Hz):
        # the "action" is the log of the per-hour frequency.
        freq_hourly = freq_Hz * 3600
        if type(freq_hourly) is torch.Tensor:
            return torch.log(freq_hourly)
        else:
            return np.log(freq_hourly)

    def _action_to_freq_Hz(self, action):
        if type(action) is torch.Tensor:
            freq_hourly = torch.exp(action)
        else:
            freq_hourly = np.exp(action)
        return freq_hourly / 3600


class FeatureNorm(nn.Module):
    def __init__(self, momentum, dim):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('running_mean', torch.zeros(dim))
        self.register_buffer('running_var', torch.zeros(dim))
        self.register_buffer('avg_count', torch.tensor(1))
        self.register_buffer('initialized', torch.tensor(False))
        self.new_mean = None
        self.new_var = None
        self.new_count = 0
        self.frozen = False

    def freeze(self):
        self.frozen = True

    def extra_repr(self) -> str:
        return f'mean={self.running_mean}, var={self.running_var}, ' \
            f'initialized={self.initialized}, count={self.avg_count}'

    def forward(self, xx):
        assert xx.shape[-1] == self.running_mean.shape[-1], \
            "input tensor has the wrong dimensionality!"

        x_size = xx.shape[0]
        if x_size == 0:
            # the tensor is empty, so we don't need to do anything.
            return xx
        
        if not self.initialized or (self.training and not self.frozen):
            old_shape = xx.shape
            xx = xx.reshape(-1, xx.shape[-1])
            x_mean = xx.mean(0).detach()
            x_var = xx.var(0).detach()
            # we're done with the need for flattened x, so reshape it
            xx = xx.reshape(old_shape)

        if self.training and not self.frozen:
            if self.new_mean is None:
                self.new_mean = x_mean
                self.new_var = x_var
                self.new_count = x_size
            else:
                # update new-samples mean
                nmrtr = (self.new_mean * self.new_count + x_mean * x_size)
                updated_count = self.new_count + x_size
                updated_mean = nmrtr / updated_count
                
                # update new-samples variance using the variance-combining formula:
                # v_c = n_1(v_1 + (m_1 - m_c)^2) + n_2 * (v_2 + (m_1 - m_2)^2) 
                #        / (n_1 + n_2)
                mean_diff_1 = (self.new_mean - updated_mean) ** 2
                mean_diff_2 = (x_mean - updated_mean) ** 2
                prev_part = self.new_count * (self.new_var + mean_diff_1)
                sample_part = x_size * (x_var + mean_diff_2)
                self.new_var = (prev_part + sample_part) / updated_count

                # update count of new samples
                self.new_count = updated_count            

        if not self.initialized:
            # we don't have any running statistics yet, so just normalize by
             # minibatch statistics.
            shift = x_mean
            denom = x_var.sqrt()
        
        else:
            # avoid division by 0
            shift = self.running_mean
            denom = self.running_var.sqrt()

        # avoid division by nan or 0. nans will occur in variance if batch 
         # size is 1.
        denom[(denom == 0) | denom.isnan()] = 1
        out = (xx - shift) / denom
        return out

    def update(self):
        if self.frozen:
            # do nothing
            return
        
        if self.new_mean is not None:
            if not self.initialized:
                # just set the initial statistics
                self.running_mean[...] = self.new_mean
                self.running_var[...] = self.new_var
                self.avg_count = torch.tensor(self.new_count)
                self.initialized[...] = True
            else:
                # update running statistics
                # scale update size in proportion to how big the sample is
                alpha = self.momentum * self.new_count / self.avg_count
                # if new count is *much* bigger than old, don't let alpha 
                # be greater than 1...that would make things wierd
                alpha = min(alpha, 1.0)
                updated_mean = alpha * self.new_mean + \
                    (1 - alpha) * self.running_mean
                
                # implements the variance-combing formula, assuming
                 # n_1 / (n_1 + n_2) = alpha, n_2 / (n_1 + n_2) = 1 - alpha
                t1 = self.new_var + (self.new_mean - updated_mean) ** 2
                t2 = self.running_var + (self.running_mean - updated_mean) ** 2
                self.running_var[...] = alpha * t1 + (1 - alpha) * t2

                # update the running mean *after* using it to compute the new
                 # running variance
                self.running_mean[...] = updated_mean
                self.avg_count = self.momentum * self.new_count + \
                    (1 - self.momentum) * self.avg_count

            self.new_mean = None
            self.new_var = None
            self.new_count = 0


class NodepairDotScorer(nn.Module):
    def __init__(self, embed_dim, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.query_transform = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.key_transform = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, flat_node_vecs, batch_n_nodes=None):
        queries = self.query_transform(flat_node_vecs)
        queries /= math.sqrt(self.embed_dim)
        keys = self.key_transform(flat_node_vecs)
        dev = flat_node_vecs.device

        if batch_n_nodes is not None:
            # fold the queries and keys into a batch dimension
            max_n_nodes = max(batch_n_nodes)
            batch_size = len(batch_n_nodes)
            folded_queries = torch.zeros((batch_size, max_n_nodes, 
                                          self.embed_dim), device=dev)
            folded_keys = folded_queries.clone()
            for bi, num_nodes in enumerate(batch_n_nodes):
                folded_queries[bi, :num_nodes, :queries.shape[-1]] = \
                    queries[:num_nodes]
                folded_keys[bi, :num_nodes, :keys.shape[-1]] = \
                    keys[:num_nodes]
        
        else:
            folded_queries = queries[None].contiguous()
            folded_keys = keys[None].contiguous()
        
        # because the folded tensors are zero everywhere there is no real
         # input, we don't need to mask anything.
        dot_prod = torch.bmm(folded_queries, folded_keys.transpose(-2, -1))
        return dot_prod


class RouteScorer(nn.Module):
    def __init__(self, embed_dim, nonlin_type, dropout, n_mlp_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.dropout = dropout
        n_extra_feats = 6
        self.out_mlp = get_mlp(n_mlp_layers, embed_dim * 2, nonlin_type, 
                               dropout, in_dim=embed_dim + n_extra_feats, 
                               out_dim=1)
        self.extras_norm = FeatureNorm(0.0, n_extra_feats)
    
    def forward(self, state, node_descs, route_idxs, route_time, 
                node_padding_mask=None):
        # assemble route sequences
        route_gather_idxs = route_idxs * (route_idxs > -1)
        route_gather_idxs = \
            route_gather_idxs.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        route_seqs = node_descs.gather(1, route_gather_idxs)

        # pass sequences through encoder
        route_desc = self.encode(node_descs, route_seqs, node_padding_mask, 
                                 route_idxs == -1)

        # pass encoding and features through MLP
        existing_times = state.total_route_time / state.n_routes_to_plan

        extra_feats = torch.stack((route_time, existing_times), dim=1)
        extra_feats = torch.cat((extra_feats, state.cost_weights, 
                                 state.get_n_routes_features()), dim=-1)
        # normalize extra features
        extra_feats = self.extras_norm(extra_feats)
        in_vec = torch.cat((route_desc, extra_feats), dim=1)
        return self.out_mlp(in_vec)

    def encode(self, node_descs, route_seqs, node_pad_mask=None, 
               route_pad_mask=None):
        raise NotImplementedError()


class RouteZeroScorer(RouteScorer):
    def forward(self, node_descs, *args, **kwargs):
        batch_size = node_descs.shape[0]
        return torch.zeros((batch_size, 1), device=node_descs.device)


class RouteMeanScorer(RouteScorer):
    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask):
        return mean_pool_sequence(route_seqs, route_pad_mask)
    

class RouteLatentScorer(RouteScorer):
    def __init__(self, n_heads, n_layers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = LatentAttentionEncoder(self.embed_dim, n_heads=n_heads,
                                              n_layers=n_layers, 
                                              dropout=self.dropout)
        
    def encode(self, node_descs, route_seqs, node_pad_mask, route_pad_mask):
        query = mean_pool_sequence(node_descs, node_pad_mask)[:, None]
        return self.encoder(route_seqs, query, route_pad_mask, True).squeeze(1)


class RouteGenBatchState:
    def __init__(self, graph_data, n_routes_to_plan, route_data_list, 
                 extra_nodepair_feats, valid_terms_mat, cost_weights, 
                 mean_stop_time, symmetric_routes, min_route_len=2, 
                 max_route_len=None):
        self._inner_init(graph_data, n_routes_to_plan, min_route_len, 
                         max_route_len, route_data_list, extra_nodepair_feats, 
                         valid_terms_mat, cost_weights, mean_stop_time, 
                         symmetric_routes)
        self.is_demand_float = (self.graph_data.demand > 0).to(torch.float32)
        
    def _inner_init(self, graph_data, n_routes_to_plan, min_route_len, 
                    max_route_len, route_data_list, extra_nodepair_feats, 
                    valid_terms_mat, cost_weights, mean_stop_time, 
                    symmetric_routes):
        # do initialization needed to make properties work
        self.graph_data = graph_data
        self.n_routes_to_plan = n_routes_to_plan
        self.route_data_list = route_data_list
        self.nodes_per_scenario = \
            torch.tensor([data.num_nodes for data in route_data_list],
                         device=graph_data[STOP_KEY].x.device)
        if max_route_len is None:
            self.max_route_len = self.nodes_per_scenario.clone()
        elif isinstance(max_route_len, torch.Tensor):
            self.max_route_len = max_route_len
        else:
            self.max_route_len = torch.full_like(self.nodes_per_scenario, 
                                                 max_route_len)
        self.min_route_len = torch.full_like(self.max_route_len, min_route_len)
        self.extra_nodepair_feats = extra_nodepair_feats
        self.base_valid_terms_mat = valid_terms_mat
        self.cost_weights = cost_weights
        self.mean_stop_time = mean_stop_time
        self.symmetric_routes = symmetric_routes
        self.total_route_time = torch.zeros_like(self.nodes_per_scenario, 
                                                 dtype=torch.float32)
        # this initializes route data
        self.clear_routes()

    def clear_routes(self):
        # for rd in self.route_data_list:
        #     rd.edge_index = rd.edge_index[:, :0]
        #     if rd.edge_attr is not None:
        #         rd.edge_attr = rd.edge_attr[:0]

        directly_connected = torch.eye(self.max_n_nodes, device=self.device, 
                                       dtype=bool)
        self.directly_connected = \
            directly_connected.repeat(self.batch_size, 1, 1)
        self.routes = [[] for _ in range(self.batch_size)]        
        self.route_mat = torch.full((self.batch_size, self.max_n_nodes, 
                                     self.max_n_nodes), 
                                    float('inf'), device=self.device)
        self.valid_terms_mat = self.base_valid_terms_mat.clone()
        self.total_route_time[...] = 0

    def add_new_routes(self, batch_new_routes,
                       only_routes_with_demand_are_valid=False, 
                       invalid_directly_connected=False):
        """Takes a tensor of new routes. The first dimension is the batch"""
        # incorporate new routes into the route graphs
        # add new routes to the route matrix.
        new_route_mat = get_route_edge_matrix(
            batch_new_routes, self.graph_data.drive_times, 
            self.mean_stop_time, self.symmetric_routes)
        self.route_mat = torch.minimum(self.route_mat, new_route_mat)
        route_lengths = (batch_new_routes > -1).sum(dim=-1)

        batch_idxs, route_froms, route_tos = \
            torch.where(self.route_mat < float('inf'))
        _, counts = torch.unique(batch_idxs, return_counts=True)
        counts = counts.tolist()
        edge_idxs = torch.stack((route_froms, route_tos))
        edge_times = self.route_mat[batch_idxs, route_froms, route_tos]

        for bi, route_data in enumerate(self.route_data_list):
            count = counts[0]
            counts = counts[1:]
            route_idxs = edge_idxs[:, :count]
            edge_idxs = edge_idxs[:, count:]
            # route_data.edge_index = route_idxs
            # route_data.edge_attr = edge_times[:count][:, None]
            edge_times = edge_times[count:]
            self.directly_connected[bi, route_idxs[0], route_idxs[1]] = True

            for route, length in zip(batch_new_routes[bi], route_lengths[bi]):
                if length <= 1:
                    # this is an invalid route
                    log.warn('invalid route!')
                    continue
                self.routes[bi].append(route[:length])

        # allow connection to any node 'upstream' of a demand dest, or
         # 'downstream' of a demand src.
        if only_routes_with_demand_are_valid:
            connected_T = self.nodes_are_connected(2).transpose(1, 2)
            connected_T = connected_T.to(torch.float32)
            valid_upstream = self.is_demand_float.bmm(connected_T)
            self.valid_terms_mat[valid_upstream.to(bool)] = True
            valid_downstream = connected_T.bmm(self.is_demand_float)
            self.valid_terms_mat[valid_downstream.to(bool)] = True

        if invalid_directly_connected:
            self.valid_terms_mat[self.directly_connected] = False

        leg_times = get_route_leg_times(batch_new_routes, 
                                        self.graph_data.drive_times,
                                        self.mean_stop_time)
        total_new_time = leg_times.sum(dim=(1,2))

        if self.symmetric_routes:
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = get_route_leg_times(batch_new_routes, 
                                                   transpose_dtm,
                                                   self.mean_stop_time)
            total_new_time += return_leg_times.sum(dim=(1,2))
            self.valid_terms_mat = self.valid_terms_mat & \
                self.valid_terms_mat.transpose(1, 2)
        
        self.total_route_time += total_new_time

    @property
    def node_covered_mask(self):
        have_out_paths = self.directly_connected.any(dim=1)
        if self.symmetric_routes:
            are_covered = have_out_paths
        else:
            are_covered = have_out_paths & self.directly_connected.any(dim=2)
        return are_covered

    @property
    def demand(self):
        return self.graph_data.demand

    @property
    def drive_times(self):
        return self.graph_data.drive_times

    @property
    def n_routes_so_far(self):
        nrsf = len(self.routes[0])
        return torch.full((self.batch_size,), nrsf, dtype=torch.float32,
                          device=self.device)

    @property
    def n_routes_left_to_plan(self):
        return self.n_routes_to_plan - self.n_routes_so_far
    
    def get_n_routes_features(self):
        so_far = self.n_routes_so_far
        left = self.n_routes_left_to_plan
        both = torch.stack((so_far, left), dim=-1)
        return (both + 1).log()

    def nodes_are_connected(self, n_transfers=2):
        dircon_float = self.directly_connected.to(torch.float32)
        connected = dircon_float
        for _ in range(n_transfers):
            # connected by 2 or fewer transfers
            connected = connected.bmm(dircon_float)
        return connected.bool()

    @property
    def device(self):
        return self.graph_data[STOP_KEY].x.device

    @property
    def batch_size(self):
        return len(self.route_data_list)
    
    @property
    def max_n_nodes(self):
        return max(self.nodes_per_scenario)
    

class RouteGeneratorBase(nn.Module):
    def __init__(self, backbone_net, mean_stop_time_s, 
                 embed_dim, n_nodepair_layers, nonlin_type=DEFAULT_NONLIN, 
                 dropout=MLP_DEFAULT_DROPOUT, symmetric_routes=True,
                 fixed_frequency_Hz=0.00074,
                 only_routes_with_demand_are_valid=False):
        """nonlin_type: a nonlin, or the string name of a nonlin, like 'ReLU' 
            or 'LeakyReLU' or 'Tanh'"""
        super().__init__()
        # make this a parameter of the model so it will be kept in state_dict
        # mst = torch.tensor(mean_stop_time_s, dtype=torch.float32)
        self.mean_stop_time_s = mean_stop_time_s

        self.dropout = dropout
        if type(nonlin_type) is str:
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.embed_dim = embed_dim
        self.n_nodepair_layers = n_nodepair_layers
        self.backbone_net = backbone_net
        self.symmetric_routes = symmetric_routes
        self.fixed_freq = fixed_frequency_Hz
        self.only_routes_with_demand_are_valid = \
            only_routes_with_demand_are_valid

        self.stop_logits_comb_mode = None

        n_edge_feats = 10
        # + 2 for the cost weights, + 2 for the two n_routes features
        self.full_nodepair_dim = n_edge_feats + embed_dim * 2

        # edge feats: demand, drive time, has route, route time, has street,
         # street time, has 1-transfer path, has 2-transfer path, has any path,
         # shortest transit path time, is same node
        self.edge_norm = FeatureNorm(0, n_edge_feats)
        self.node_norm = FeatureNorm(0, 8)
        self.cost_weight_norm = FeatureNorm(0, 2)

    @property
    def edge_key_order(self):
        return (DEMAND_KEY, ROUTE_KEY)

    def update_and_freeze_feature_norms(self):
        for mod in self.modules():
            if isinstance(mod, FeatureNorm):
                mod.update()
                mod.freeze()

    def plan(self, *args, **kwargs):
        return self.forward_oldenv(*args, **kwargs)        

    def setup_planning(self, graph_data, n_routes_to_plan, min_route_len, 
                       max_route_len, cost_weights):
        # compute the valid terminals matrices
        batch_size = graph_data.num_graphs
        dev = graph_data[STOP_KEY].x.device
        max_n_nodes = graph_data.demand.shape[-1]
        if self.only_routes_with_demand_are_valid:
            valid_terms_mat = graph_data.demand > 0
        else:
            valid_terms_mat = ~torch.eye(max_n_nodes, device=dev, dtype=bool)
            valid_terms_mat = valid_terms_mat.repeat(batch_size, 1, 1)

        if not isinstance(graph_data, Batch):
            graph_data = Batch.from_data_list([graph_data])

        batch_size = graph_data.num_graphs
        max_n_nodes = graph_data.demand.shape[-1]
 
        # embed the data
        log.debug("embedding graph")
        # first, apply normalization to input features
        norm_stops_x = self.node_norm(graph_data[STOP_KEY].x)
        
        # assemble route data objects
        route_data_list = []
        for dd in graph_data.to_data_list():
            node_idxs = torch.arange(dd.num_nodes, device=dev)
            index = torch.combinations(node_idxs, with_replacement=True).T
            index = torch.cat((index, index.flip(0)), dim=1)
            # coalesce to remove duplicate self-connections
            rd = Data(norm_stops_x[:dd.num_nodes], index).coalesce()
            route_data_list.append(rd)
            norm_stops_x = norm_stops_x[dd.num_nodes:]

        # build the cost_weights tensor from the provided dict
        cost_weights_list = []
        for key in sorted(cost_weights.keys()):
            if type(cost_weights[key]) is torch.Tensor:
                cw = cost_weights[key].to(dev)
                if cw.ndim == 0:
                    cw = cw[None]
            else:
                cw = torch.tensor(cost_weights[key], device=dev)[None]

            cost_weights_list.append(cw)
        cost_weights = torch.stack(cost_weights_list, dim=1)
        if cost_weights.shape[0] == 1:
            cost_weights = cost_weights.expand(graph_data.num_graphs, -1)
        if cost_weights.shape[0] > batch_size:
            cost_weights = cost_weights[:batch_size]
        # finally, normalize the cost weights
        cost_weights = self.cost_weight_norm(cost_weights ** 3)

        # return this as a state
        return RouteGenBatchState(graph_data, n_routes_to_plan, route_data_list,
                                  None, valid_terms_mat, cost_weights,
                                  self.mean_stop_time_s, self.symmetric_routes,
                                  min_route_len, max_route_len)

    def step(self, state, greedy, chunk_size=1):
        log.debug("stepping")
        route_batch = Batch.from_data_list(state.route_data_list)
        # get edge features and assign them to the batch
        if type(self.backbone_net) in [EdgeGraphNet, GraphAttnNet]:
            edge_features_square = self._get_edge_features(state)
            for ef, rd in zip(edge_features_square, state.route_data_list):
                rd.edge_attr = ef[rd.edge_index[0], rd.edge_index[1]]
            
        route_batch = Batch.from_data_list(state.route_data_list)

        # run GNN forward
        if self.backbone_net.gives_edge_features:
            route_node_embeds, _ = self.backbone_net(route_batch)
        else:
            route_node_embeds = self.backbone_net(route_batch)
 
        # do routing and choosing (different subclasses do it differently)
        log.debug("get routes")
        batch_new_routes, route_logits, stop_logits = self._get_routes(
            chunk_size, state, route_node_embeds, edge_features_square, greedy)
        
        # update the state
        state.add_new_routes(
            batch_new_routes, self.only_routes_with_demand_are_valid)
        return state, route_logits, stop_logits

    def _get_edge_features(self, state):
        # edge feats: demand, drive time, has route, route time, has street,
         # street time, has 1-transfer path, has 2-transfer path, has any path,
         # shortest transit path time, is same node
        edge_feature_parts = [state.demand, state.drive_times]
        # has a direct route
        has_direct_route = state.route_mat.isfinite()
        edge_feature_parts.append(has_direct_route)
        # # time on direct route 
        # finite_direct_times = get_update_at_mask(state.route_mat, 
        #                                          ~has_direct_route)
        # edge_feature_parts.append(finite_direct_times)
        # has street
        has_street = state.graph_data.street_adj.isfinite()
        edge_feature_parts.append(has_street)
        # street time
        street_times = get_update_at_mask(state.graph_data.street_adj,
                                          ~has_street)
        edge_feature_parts.append(street_times)
        # has 1-transfer path, 2-transfer path, any path, shortest transit
        # compute route shortest path lengths
        _, _, times = floyd_warshall(state.route_mat, True)
        is_path = times < float('inf')
        edge_feature_parts.append(is_path)
        # times of existing paths
        times[~is_path] = 0
        edge_feature_parts.append(times)
        is_direct_path = state.directly_connected
        flt_is_direct_path = is_direct_path.to(torch.float32)
        flt_upto_1trnsfr_path = flt_is_direct_path.bmm(flt_is_direct_path)
        flt_upto_2trnsfr_path = flt_upto_1trnsfr_path.bmm(flt_is_direct_path)
        upto_1trnsfr_path = flt_upto_1trnsfr_path.bool()
        is_1trnsfr_path = upto_1trnsfr_path ^ is_direct_path
        edge_feature_parts.append(is_1trnsfr_path)
        upto_2trnsfr_path = flt_upto_2trnsfr_path.bool()
        is_2trnsfr_path = upto_2trnsfr_path ^ upto_1trnsfr_path
        edge_feature_parts.append(is_2trnsfr_path)

        # is same node
        eye = torch.eye(state.max_n_nodes, device=state.device)
        eye = eye.expand(state.batch_size, -1, -1)
        edge_feature_parts.append(eye)
        edge_feature_square = torch.stack(edge_feature_parts, dim=-1)

        return self.edge_norm(edge_feature_square)

    def forward_oldenv(self, env_rep, n_routes, batch_size=1, greedy=False, 
                n_chunks=None):
        # assemble the data from the env rep into a batch
        data = CityGraphData()
        data[STOP_KEY].x = env_rep.stop_data.x
        data[STOP_KEY].pos = env_rep.stop_data.pos
        data[STREET_KEY].edge_index = env_rep.stop_data.edge_index
        data[STREET_KEY].edge_attr = env_rep.stop_data.edge_attr
        data.demand = env_rep.inter_node_demands

        # compute all shortest paths
        _, nexts, times = floyd_warshall(env_rep.stop_data.street_time_matrix, 
                                         return_raw_tensors=True)
        assert (nexts >= 0).all(), "the graph is disconnected!"
        data.drive_times = times.squeeze(0)
        data.nexts = nexts.squeeze(0)
        batch = Batch.from_data_list([data] * batch_size)

        return self._forward_helper(batch, n_routes, n_chunks, greedy)

    def forward(self, graph_data, n_routes, min_route_len, max_route_len, 
                cost_weights, greedy=False):
        return self._forward_helper(graph_data, n_routes, cost_weights, greedy,
                                    min_route_len, max_route_len)

    def _forward_helper(self, graph_data, n_routes, cost_weights, greedy=False,
                        min_route_len=2, max_route_len=None, n_chunks=None):
        # do a full rollout
        state = self.setup_planning(graph_data, n_routes, min_route_len, 
                                    max_route_len, cost_weights)

        if n_chunks is None:
            n_chunks = n_routes

        r_chunk_size = int(n_routes / n_chunks)
        all_stop_logits = []
        all_route_logits = []

        log.debug("starting route-generation loop")
        n_routes_left = n_routes
        for ci in range(n_chunks):
            if ci == n_chunks - 1:
                # the last chunk is just however many are left
                r_chunk_size = n_routes - ci * r_chunk_size
            
            state, route_logits, stop_logits = \
                self.step(state, greedy, r_chunk_size)
            if route_logits is not None:
                all_route_logits.append(route_logits)
            if stop_logits is not None:
                all_stop_logits.append(stop_logits)
            n_routes_left -= r_chunk_size

        # assemble the logits into single tensors
        if len(all_route_logits) > 0:
            route_logits = torch.cat(all_route_logits, dim=1)
        else:
            route_logits = None
        stop_logits = self.combine_stop_logits(all_stop_logits)
            
        if self.fixed_freq is not None:
            freqs = [[self.fixed_freq for _ in range(len(rs))]
                     for rs in state.routes]
        else:
            freqs = None

        routes_tensor = get_batch_tensor_from_routes(state.routes, state.device)
        # TODO make this return a named tuple with names that actually make 
         # sense
        result = PlanResults(
            routes=state.routes, freqs=freqs, stop_logits=stop_logits,
            route_logits=route_logits, freq_logits=None, stops_tensor=None, 
            routes_tensor=routes_tensor, freqs_tensor=None, 
            stop_est_vals=None, route_est_vals=None, freq_est_vals=None
        )
        return result
    
    @staticmethod
    def fold_node_descs(node_descs, state):
        # fold the node description vectors so that the batch dimension
         # is separate from the nodes-in-a-batch-elem dimension
        folded_node_descs = torch.zeros((state.batch_size, state.max_n_nodes, 
                                         node_descs.shape[-1]), 
                                         device=node_descs.device)
        node_pad_mask = torch.zeros((state.batch_size, state.max_n_nodes), 
                                    dtype=bool, device=node_descs.device)
        for bi, num_nodes in enumerate(state.nodes_per_scenario):
            folded_node_descs[bi, :num_nodes, :node_descs.shape[-1]] = \
                node_descs[:num_nodes]
            node_pad_mask[bi, num_nodes:] = True
        
        return folded_node_descs, node_pad_mask

    @staticmethod
    def get_node_pair_descs(node_descs):
        n_nodes = node_descs.shape[-2]
        exp_dst_nodes = node_descs[:, None].expand(-1, n_nodes, -1, -1)
        exp_src_nodes = exp_dst_nodes.permute(0, 2, 1, 3)
        return torch.cat((exp_src_nodes, exp_dst_nodes), dim=-1)

    def set_invalid_scores_to_fmin(self, scores, valid_terminals_mat):
        scores_holder = torch.full_like(scores, TORCH_FMIN)
        no_valid_options = valid_terminals_mat.sum(dim=(1,2)) == 0
        if no_valid_options.any():
            # no valid options, so make all terminal pairs equally likely
            off_diag = ~torch.eye(scores.shape[-1], device=scores.device, 
                                  dtype=bool)
            where_nvo = torch.where(no_valid_options)[0]
            scores_holder[where_nvo[:, None], off_diag] = 0

        # valid_terminals_mat keeps changing, so clone so pytorch doesn't 
         # complain during backprop
        vtm = valid_terminals_mat.clone()
        scores_holder[vtm] = scores[vtm]

        return scores_holder

    def combine_stop_logits(self, all_stop_logits):
        if len(all_stop_logits) > 0:
            if self.stop_logits_comb_mode == "stack":
                stop_logits = torch.stack(all_stop_logits, dim=1)
            elif self.stop_logits_comb_mode == "cat":
                stop_logits = cat_var_size_tensors(all_stop_logits, dim=1)
            else:
                raise ValueError("stop_logits_comb_mode must be cat or stack")
        else:
            stop_logits = None
        return stop_logits

    def get_biased_scores(self, scores, bias, are_connected):
        # no bias for demand that's already covered
        # bias = get_update_at_mask(bias, are_connected)
        scores += bias
        # # set scores for invalid connections to 0
        scores = scores * ~are_connected

        return scores


class ShortestPathRouteGenerator(RouteGeneratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nodepair_scorer = \
            get_mlp(self.n_nodepair_layers, self.embed_dim, 
                    in_dim=self.full_nodepair_dim, out_dim=1)

    def _get_routes(self, n_routes, graph_data, valid_terms_mat, are_connected,
                    folded_node_descs, nodepair_descs, greedy):
        nexts, stop_logits = \
            self.find_shortest_paths(graph_data, nodepair_descs, greedy)                    
        seqs, _ = reconstruct_all_paths(nexts)
        np_scores = self.nodepair_scorer(nodepair_descs).squeeze(-1)
        np_scores = self.get_biased_scores(
            np_scores, graph_data.demand, are_connected)
        path_scores = aggregate_dense_conns(seqs, np_scores[..., None], 'sum')
        path_scores = \
            self.set_invalid_scores_to_fmin(path_scores, valid_terms_mat)

        # flatten the src and dst dimensions
        path_scores = path_scores.reshape(graph_data.num_graphs, -1)
        seqs = seqs.flatten(1, 2)
        
        flat_slctd_idxs, route_logits = \
            select(path_scores, scores_as_logits=not greedy, 
                   n_selections=n_routes)

        # gather new routes
        # add a sequence dimension
        exp_shape = (-1, -1, seqs.shape[-1])
        exp_idxs = flat_slctd_idxs[..., None].expand(*exp_shape)
        batch_new_routes = seqs.gather(1, exp_idxs)

        return batch_new_routes, route_logits, stop_logits

    def find_shortest_paths(self, graph_data, *args, **kwargs):
        # the true shortest paths, and no stop logits
        return graph_data.nexts, None


class WeightedShortestPathRouteGenerator(ShortestPathRouteGenerator):
    """Same as shortest path router, but learns to find the shortest paths."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.routing_nodepair_weighter = ContinuousGaussianActor(
            self.n_nodepair_layers, self.embed_dim, self.full_nodepair_dim,
            min_action=-10, max_action=10, min_logstd=-10, max_logstd=3)
        self.stop_logits_comb_mode = "stack"

    def find_shortest_paths(self, graph_data, nodepair_descs, greedy):
        # find the weighted shortest paths
        edge_weights, log_probs = \
            self.routing_nodepair_weighter(nodepair_descs)
        edge_weights = edge_weights.sigmoid().squeeze(-1)
        weighted_edge_costs = graph_data.street_adj * edge_weights
        _, nexts, _ = floyd_warshall(weighted_edge_costs, True)
        return nexts, log_probs


class PathCombiningRouteGenerator(RouteGeneratorBase):
    def __init__(self, *args, mask_used_paths=True, n_scorelenfn_layers=2, 
                 scorelenfn_hidden_dim=8, n_halt_layers=2, n_halt_heads=4,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.nodepair_scorer = get_mlp(self.n_nodepair_layers, self.embed_dim,
                                       self.nonlin_type, self.dropout,
                                       in_dim=self.full_nodepair_dim, 
                                       out_dim=1)
        self.time_norm = FeatureNorm(0, 1)
        if n_scorelenfn_layers == 0:
            self.nodepair_updater = None
        else:
            self.nodepair_updater = get_mlp(
                n_scorelenfn_layers, scorelenfn_hidden_dim, self.nonlin_type, 
                self.dropout, in_dim=2, out_dim=1)
        self.stop_logits_comb_mode = "cat"

        assert self.only_routes_with_demand_are_valid is False, 'not supported'
        self.mask_used_paths = mask_used_paths

        self.path_scorer = nn.Sequential(
            FeatureNorm(0, 7),
            get_mlp(n_scorelenfn_layers, scorelenfn_hidden_dim, 
                    self.nonlin_type, self.dropout, in_dim=7, out_dim=1)
        )
        self.halt_scorer = RouteLatentScorer(n_halt_heads, n_halt_layers,
                                             self.embed_dim, self.nonlin_type,
                                             self.dropout)

    def _get_routes(self, n_routes, state, node_descs, extra_nodepair_feats, 
                    greedy):
        # fold the node features
        batch_size = state.graph_data.num_graphs
        max_n_nodes = max(state.nodes_per_scenario)
        node_descs, node_pad_mask = self.fold_node_descs(node_descs, state)
            
        nodepair_descs = self.get_node_pair_descs(node_descs)

        # concatenate the extra nodepair features if they're there
        if extra_nodepair_feats is not None:
            nodepair_descs = torch.cat((extra_nodepair_feats, nodepair_descs), 
                                       dim=-1)
        
        # give these to nodepair scorer
        np_scores = self.nodepair_scorer(nodepair_descs).squeeze(-1)

        if not hasattr(state.graph_data, '_seqs'):
            # compute the shortest paths and store them in the graph_data object
             # so we don't need to recompute them for each route.
            seqs, _ = reconstruct_all_paths(state.graph_data.nexts)
            state.graph_data._seqs = seqs
        else:
            seqs = state.graph_data._seqs

        path_scores = aggregate_dense_conns(seqs, np_scores[..., None], 'sum')
        path_scores.squeeze_(-1)
        path_scores = \
            self.set_invalid_scores_to_fmin(path_scores, state.valid_terms_mat)

        routes = []
        all_logits = []
        max_n_nodes = nodepair_descs.shape[1]
        batch_size = state.graph_data.num_graphs
        batch_idxs = torch.arange(batch_size, device=np_scores.device)
        for ii in range(n_routes):
            route, logits = self._assemble_route(state, node_descs, seqs, 
                                                 path_scores, np_scores, greedy, 
                                                 node_pad_mask)
            routes.append(route)
            all_logits.append(logits)

            if ii == n_routes - 1:
                # don't do the below on the last iteration
                break

            if self.mask_used_paths:
                path_scores = \
                    self.set_invalid_scores_to_fmin(path_scores, 
                                                    state.valid_terms_mat)
            else:
                # update np_scores based on new routes
                mask_shape = (batch_size, max_n_nodes+1, max_n_nodes+1)
                new_mask = torch.zeros(mask_shape, device=np_scores.device,
                                       dtype=bool)
                for ii in range(route.shape[1] - 1):
                    new_mask[batch_idxs, route[:, ii], route[:, ii+1:]] = True
                new_mask = new_mask[:, :-1, :-1]
                np_scores = get_update_at_mask(np_scores, new_mask)
        
        all_logits = cat_var_size_tensors(all_logits, dim=1)
        routes = torch.stack(routes, dim=-2)

        # prune extra route sequence padding
        max_route_len = (routes > -1).sum(dim=-1).max()
        routes = routes[..., :max_route_len]
        return routes, all_logits, None
    
    def _update_path_scores(self, state, base_path_scores, new_path_lens, 
                            new_drive_times):
        """Update the path scores based on the new path lengths and drive times.
        """
        is_valid = base_path_scores > TORCH_FMIN
        n_route_feats = state.get_n_routes_features()
        existing_route_times = state.total_route_time / state.n_routes_to_plan
        update_in = torch.stack((base_path_scores, new_drive_times), dim=-1)

        cost_weights = state.cost_weights
        n_cw = cost_weights.shape[-1]
        n_rf = n_route_feats.shape[-1]
        update_in = nn.functional.pad(update_in, (0, 1 + n_cw + n_rf))
        for _ in range(update_in.ndim - cost_weights.ndim):
            cost_weights = cost_weights.unsqueeze(1)
            n_route_feats = n_route_feats.unsqueeze(1)
            existing_route_times = existing_route_times.unsqueeze(1)

        update_in[..., -(n_cw + n_rf):-n_cw] = n_route_feats
        update_in[..., -n_cw:] = cost_weights
        update_in[..., -1] = existing_route_times

        update_in = update_in[is_valid]

        updated_path_scores_wo_infs = self.path_scorer(update_in).squeeze(-1)
        updated_path_scores = torch.full_like(base_path_scores, TORCH_FMIN)
        updated_path_scores[is_valid] = updated_path_scores_wo_infs

        return updated_path_scores

    def _assemble_route(self, state, node_descs, path_seqs, path_scores, 
                        nodepair_embeds, greedy, node_pad_mask=None):
        """Here, 'valid terms' means a pair of nodes whose shortest-path is 
           not yet part of any route."""
        log.debug("assembling route")
        batch_size = state.drive_times.shape[0]
        dev = nodepair_embeds.device

        # initialize the routes
        path_lens = (path_seqs > -1).sum(dim=-1)
        updated_path_scores = self._update_path_scores(
            state, path_scores, path_lens, state.drive_times)
        # don't choose segments greater than the max route length
        updated_path_scores = get_update_at_mask(updated_path_scores,
            path_lens > state.max_route_len[:, None, None], TORCH_FMIN)
        flat_path_scores = updated_path_scores.reshape(batch_size, -1)

        # scores, flat_idxs = flat_path_scores.max(1)
        flat_idxs, start_logits = select(flat_path_scores, not greedy)

        # gather new routes
        exp_shape = (-1, -1, path_seqs.shape[-1])
        exp_idxs = flat_idxs[..., None].expand(*exp_shape)
        start_routes = path_seqs.flatten(1, 2).gather(1, exp_idxs)
        start_routes.squeeze_(-2)
        max_n_nodes = state.drive_times.shape[1]
        routes = torch.full((batch_size, max_n_nodes), -1, device=dev)
        routes[:, :start_routes.shape[-1]] = start_routes
        route_lens = path_lens.flatten(1, 2).gather(1, flat_idxs).squeeze(-1)

        # mark chosen paths as invalid
        batch_idxs = torch.arange(batch_size)

        # pad each node axis with a row/column of zeros, so we can safely index
         # with -1.
        padding = (0,1, 0,1, 0,0)
        drive_times = torch.nn.functional.pad(state.drive_times, padding)
        nodepair_embeds = torch.nn.functional.pad(nodepair_embeds, padding)
        are_on_routes = torch.zeros((batch_size, max_n_nodes + 1), 
                                      device=dev).bool()
        are_on_routes[batch_idxs[:, None], routes] = True
        # make padding column False
        are_on_routes[:, -1] = False

        times_from_start = torch.zeros((batch_size, max_n_nodes), device=dev)
        init_tfs = drive_times[batch_idxs[:, None], routes[:, 0:1], routes]
        times_from_start[:, :init_tfs.shape[-1]] = init_tfs

        is_done = torch.zeros(batch_size, device=dev).bool()
        logits = start_logits.squeeze(-1)
        
        while not is_done.all():
            getext_args = (state, routes, times_from_start, route_lens, 
                           are_on_routes, path_seqs, nodepair_embeds, 
                           drive_times, path_scores, path_lens)

            next_path_scores = self._get_extension_scores(
                *getext_args, before_or_after='after')
            last_node = routes.gather(1, route_lens[:, None] - 1).squeeze(-1)

            # repeat with to/from swapped
            prev_path_scores = self._get_extension_scores(
                *getext_args, before_or_after='before')
            ext_scores = torch.cat(
                (prev_path_scores, next_path_scores), dim=-1)
            # no valid options, so force it to be done
            is_done = is_done | (ext_scores == TORCH_FMIN).all(dim=-1)

            route_times = times_from_start.max(dim=-1)[0]
            if self.symmetric_routes:
                route_times *= 2
            
            # choose to halt
            halt_scores = self.halt_scorer(state, node_descs, routes, 
                                           route_times, node_pad_mask)
            halt_scores[is_done] = TORCH_FMAX
            # don't halt if the route is too short
            halt_scores[route_lens < state.min_route_len] = TORCH_FMIN
            continue_scores = -halt_scores
            cont_or_halt = torch.cat((continue_scores, halt_scores), dim=-1)
            halt, corh_logit = select(cont_or_halt, not greedy)
            corh_logit = corh_logit.squeeze(1) * ~is_done
            # update doneness
            is_done = is_done | halt.squeeze(1).bool()

            chosen_ext, ext_logits = select(ext_scores, not greedy)
            chosen_ext = chosen_ext.squeeze(-1)
            ext_logits = ext_logits.squeeze(-1)
            # set logits and choices to 0 for already-done routes
            ext_logits = ext_logits * ~is_done
            chosen_ext = chosen_ext * ~is_done

            # add the chosen part to the route
            n_prevs = prev_path_scores.shape[-1]
            # chose_prev = (chosen_ext < n_prevs + 1) & ~is_done
            chose_prev = (chosen_ext < n_prevs) & ~is_done
            chose_next = ~(chose_prev | is_done)
            done_idxs = is_done * -1
            best_starts = chose_prev * (chosen_ext) + \
                            chose_next * last_node + done_idxs
            first_node = routes[:, 0].clone()
            best_ends = chose_prev * first_node + \
                        chose_next * (chosen_ext - n_prevs) + done_idxs

            # -1 to account for the terminal at the joining end
            new_part_lens = path_lens[batch_idxs, best_starts, best_ends] - 1

            new_parts = torch.full_like(routes, -1)
            np = path_seqs[batch_idxs, best_starts, best_ends].clone()
            new_parts[:, :path_seqs.shape[-1]] = np

            # trim the first nodes of new parts that are next choices
            new_parts[chose_next, :-1] = new_parts[chose_next, 1:]
            new_parts[chose_next, -1] = -1
            # trim the last nodes of new parts that are previous choices
            max_idx = new_parts.shape[-1] - 1
            last_node_locs = new_part_lens * chose_prev + max_idx * ~chose_prev
            new_parts.scatter_(1, last_node_locs[:, None], -1)
            
            # first part is current route if we chose to add next OR halt
            first_part_lens = new_part_lens * chose_prev + \
                              route_lens * ~chose_prev
            first_parts = routes * ~chose_prev[:, None] + \
                          new_parts * chose_prev[:, None]
            first_part_mask = get_variable_slice_mask(
                routes, dim=1, tos=first_part_lens)

            # second part is zeros if we chose to halt
            scnd_part_lens = route_lens * chose_prev + \
                             new_part_lens * chose_next
            scnd_parts = new_parts * chose_next[:, None] + \
                         routes * chose_prev[:, None]

            # set the new first part of the route
            routes = get_update_at_mask(routes, first_part_mask, first_parts)

            # set the new second part of the route
            new_route_lens = first_part_lens + scnd_part_lens
            scnd_part_mask = get_variable_slice_mask(
                routes, dim=1, froms=first_part_lens, tos=new_route_lens)
            scnd_part_len_mask = get_variable_slice_mask(
                routes, dim=1, tos=scnd_part_lens)
            routes[scnd_part_mask] = scnd_parts[scnd_part_len_mask]

            # update times_from_start
            # first, times_from_start for the new route's first part
            new_part_tfs = drive_times[batch_idxs[:, None], 
                                        best_starts[:, None], new_parts]
            first_part_tfs = times_from_start * ~chose_prev[:, None] + \
                             new_part_tfs * chose_prev[:, None]
            times_from_start[first_part_mask] = first_part_tfs[first_part_mask]

            # then, times_from_start for the new route's second part
            np_end_idxs = (new_part_lens - 1) * (new_part_lens > 0)
            np_end_tfs = new_part_tfs.gather(1, np_end_idxs[:, None])
            next_old_tfs = times_from_start + np_end_tfs                
            cur_end_tfs = times_from_start.gather(1, route_lens[:, None] - 1)
            next_new_part_tfs = new_part_tfs + cur_end_tfs
            scnd_part_tfs = next_new_part_tfs * chose_next[:, None] + \
                            next_old_tfs * chose_prev[:, None]
            times_from_start[scnd_part_mask] = scnd_part_tfs[scnd_part_len_mask]

            # update route and tracking variables
            are_on_routes[batch_idxs[:, None], routes] = True
            # keep padding column False
            are_on_routes[:, -1] = False
            # update route lengths
            route_lens = new_route_lens
                
            logits += ext_logits + corh_logit

        # combine all logits
        logits = logits[:, None]

        log.debug("new route assembled!")
        return routes, logits

    def _get_extension_scores(self, state, routes, times_from_start, route_lens, 
                              are_on_routes, all_paths, nodepair_embeds, 
                              drive_times, path_scores, path_lens,
                              before_or_after='after'):
        log.debug("get extension scores")
        assert before_or_after in ['before', 'after']
        lens_wo_term = route_lens - 1
        final_times = times_from_start.gather(1, lens_wo_term[:, None])
        times_to_end = (times_from_start - final_times).abs()
        if before_or_after == 'after':
            # terminal is last node
            mask = get_variable_slice_mask(routes, 1, froms=lens_wo_term)
            routes_but_terminal = get_update_at_mask(routes, mask, -1)[:, :-1]
            terminal_nodes = routes.gather(1, lens_wo_term[:, None]).squeeze(-1)
            times_to_end = get_update_at_mask(times_to_end, mask, 0)[:, :-1]
        else:
            # terminal is first node
            routes_but_terminal = routes[:, 1:].clone()
            terminal_nodes = routes[:, 0]
            times_to_end = times_to_end[:, 1:]
            # swap from- and to-node axes so indexing the first node axis
             # by the terminal gives data on edges/paths *to* the terminal,
             # instead of *from* the terminal.
            drive_times = drive_times.transpose(-2, -1)
            nodepair_embeds = nodepair_embeds.transpose(-2, -1)
            path_scores = path_scores.transpose(-2, -1)
            path_lens = path_lens.transpose(-2, -1)
            all_paths = all_paths.transpose(-3, -2)

        # get possible extensions 
        batch_idxs = torch.arange(routes.shape[0], device=routes.device)
        extension_paths = all_paths[batch_idxs, terminal_nodes].clone()
        # ignore the terminal itself in the extensions
        extension_paths[extension_paths == terminal_nodes[:, None, None]] = -1

        times_from_end = drive_times[batch_idxs, terminal_nodes]
        # batch_size x route_len - 1 x n_nodes + 1
        times_on_route = times_from_end[:, None] + times_to_end[:, :, None]
        
        # compute scores for all edges from each node on route
        embeds_on_route = nodepair_embeds[batch_idxs[:, None], 
                                          routes_but_terminal]

        if self.nodepair_updater is None:
            # use the raw node-pair scores
            from_route_edge_scores = embeds_on_route
        else:
            # use a neural network to compute scores and lengths
            norm_times = self.time_norm(times_on_route.flatten(0, 2)[:, None])
            norm_times = norm_times.reshape_as(times_on_route)
            log.debug("updating edge scores...")
            descs = torch.stack((embeds_on_route, norm_times), dim=-1)
            # route_len - 1 x n_nodes + 1
            from_route_edge_scores = self.nodepair_updater(descs).squeeze(-1)
            log.debug("edge scores updated")

        from_route_edge_scores *= (routes_but_terminal > -1)[..., None]
        # sum over nodes on route to get "score" of visiting each node before/
         # after the current route.
        from_route_node_scores = from_route_edge_scores.sum(dim=-2)
        # compute scores for each possible following sequence
        # ignore first node of each sequence, since it's the terminal
        from_route_pathnode_scores = \
            from_route_node_scores[batch_idxs[:, None, None], extension_paths]
        # zero locations corresponding to "dummy" extension path nodes
        from_route_pathnode_scores[extension_paths == -1] = 0
        from_route_path_scores = from_route_pathnode_scores.sum(dim=-1)

        # add score of route from end node to each node not on route, and add
         # current score
        innate_scores = path_scores[batch_idxs, terminal_nodes]
        ext_scores = from_route_path_scores + innate_scores

        # apply final transformer of lengths
        new_lens = path_lens[batch_idxs, terminal_nodes] + route_lens[:, None]
        # max over the stops already on the route, and drop the dummy row
        new_drive_times = times_on_route.max(dim=-2)[0][..., :-1]
        ext_scores = self._update_path_scores(state, ext_scores, new_lens, 
                                              new_drive_times)

        revisits = are_on_routes[batch_idxs[:, None, None], extension_paths]
        revisits = revisits.any(dim=-1)
        is_empty_path = path_lens[batch_idxs, terminal_nodes] == 0
          

        invalid = revisits | is_empty_path | \
            (new_lens > state.max_route_len[:, None])
        ext_scores = get_update_at_mask(ext_scores, invalid, TORCH_FMIN)

        log.debug("extension scores gotten")

        return ext_scores


class UnbiasedPathCombiner(RouteGeneratorBase):
    def __init__(self, *args, n_heads=1, n_encoder_layers=1, 
                 n_selection_attn_layers=1, **kwargs):
        super().__init__(*args, **kwargs)
        n_extra_path_feats = 2
        self.path_extras_norm = FeatureNorm(0.0, n_extra_path_feats)
        # self.path_encoder = MeanRouteEncoder()
        self.path_encoder = MaxRouteEncoder()
        # self.path_encoder = LatentAttnRouteEncoder(self.embed_dim, 
        #                                            n_heads, n_encoder_layers,
        #                                            self.dropout)
        self.path_encoder = TransformerRouteEncoder(self.embed_dim, 
                                                    n_heads, n_encoder_layers,
                                                    self.dropout)
        self.path_mlp = get_mlp(
            3, self.embed_dim * 2, self.nonlin_type, self.dropout, 
            in_dim=self.embed_dim + n_extra_path_feats, 
            out_dim=self.embed_dim)

        init_ctxt = nn.Parameter(torch.randn(1, self.embed_dim))
        self.register_parameter(name="init_ctxt", param=init_ctxt)

        self.n_selection_layers = n_selection_attn_layers
        self.kv_embedder = nn.Linear(
            self.embed_dim, self.embed_dim * (2 * n_selection_attn_layers + 1)
        )

        n_extra_global_feats = 3
        self.global_extras_norm = FeatureNorm(0.0, n_extra_global_feats)
        init_dim = self.embed_dim * 2 + n_extra_global_feats
        query_embedders = \
            [nn.Linear(init_dim, self.embed_dim)]
        query_embedders += [nn.Linear(self.embed_dim, self.embed_dim)
                            for _ in range(n_selection_attn_layers - 1)]
        self.query_embedders = nn.ModuleList(query_embedders)

        self.halt_scorer = get_mlp(2, self.embed_dim * 2, self.nonlin_type, 
                                   self.dropout, in_dim=self.embed_dim,
                                   out_dim=1)

    def _get_routes(self, n_routes, state, node_descs, extra_nodepair_feats,
                    greedy):
        """
        n_routes: int
        node_descs: batch_size x n_nodes x node_feat_dim
        state: batch_size x n_nodes x n_nodes x state_dim
        greedy: bool
        returns: batch_size x n_routes x route_len, all_logits, None
        """
        log.debug("get routes")
        # doesn't support chunk sizes greater than 1
        assert n_routes == 1

        if not hasattr(state.graph_data, '_seqs'):
            # compute the shortest paths and store them in the graph_data object
             # so we don't need to recompute them for each route.
            seqs, _ = reconstruct_all_paths(state.graph_data.nexts)
            state.graph_data._seqs = seqs
        else:
            seqs = state.graph_data._seqs
        
        # fold node descriptors
        node_descs, node_pad_mask = self.fold_node_descs(node_descs, state)

        # encode the paths
        # with torch.no_grad():
        path_encs = self.encode_path(node_descs, seqs, state.drive_times)
        path_attn_embeds = self.kv_embedder(path_encs)
        
        # do all the key and value embeddings at once
        base_ext_mask = ~state.valid_terms_mat.clone()
        ext_mask = base_ext_mask.flatten(1, 2)
        
        # set the initial context
        # TODO later, maybe add cost component weights
        n_routes_so_far = torch.full_like(state.total_route_time,
                                          state.n_routes_so_far)
        n_routes_left = torch.full_like(state.total_route_time,
                                        state.n_routes_left_to_plan)
        global_feats = torch.stack(
            (state.total_route_time, n_routes_so_far, n_routes_left),
            dim=-1)
        global_feats = self.global_extras_norm(global_feats)
        # get average node descriptor
        avg_node = mean_pool_sequence(node_descs, node_pad_mask)
        # concatenate the above
        global_feats = torch.cat((global_feats, avg_node), dim=-1)
        init_ctxt = self.init_ctxt.expand(state.batch_size, -1)
        context = torch.cat((global_feats, init_ctxt), dim=-1)

        # set up loop-tracking variables and other needed values
        route = torch.full((state.batch_size, state.max_n_nodes), -1, 
                            device=state.device)
        is_done = torch.zeros(state.batch_size, device=state.device).bool()
        logits = None
        candidates = seqs.flatten(1, 2)
        cdt_times = state.drive_times.flatten(1, 2)
        first_iter = True
        batch_idxs = torch.arange(state.batch_size, device=state.device)
        relevant_attn_embeds = path_attn_embeds.flatten(1, 2)
        node_on_route = torch.zeros((state.batch_size, state.max_n_nodes + 1), 
                                    device=state.device).bool()
        route_times = torch.zeros(state.batch_size, device=state.device)

        # assemble a route!
        while not is_done.all():
            # compute scores vs current context and pick an extension path
            out_context, ext_probs = self._get_extension_scores(
                context, relevant_attn_embeds, ext_mask)
            
            if not first_iter:
                # decide whether to halt using context
                halt_score = self.halt_scorer(out_context)
                halt_score[is_done] = TORCH_FMAX
                cont_score = -halt_score
                cont_or_halt = torch.cat((cont_score, halt_score), dim=-1)
                halt, corh_logit = select(cont_or_halt, not greedy)
                # zero out logits where we've already halted
                corh_logit = corh_logit.squeeze(1) * ~is_done
                # update doneness
                is_done = is_done | halt.squeeze(1).bool()
                logits += corh_logit

            if greedy:
                selection = ext_probs.argmax(dim=-1)
            else:
                selection = ext_probs.multinomial(1)
            step_logit = ext_probs.gather(1, selection).log().squeeze(1)
            step_logit = step_logit + ~is_done
            if first_iter:
                logits = step_logit
            else:
                logits += step_logit + corh_logit

            # update the routes
            route_lens = (route > -1).sum(-1)
            candidate_lens = (candidates > -1).sum(-1)
            chosen_ext_lens = candidate_lens.gather(1, selection).squeeze(1)
            # enforce no extension if is_done
            chosen_ext_lens[is_done] = 0
            # compute mask for new stops on the routes
            insert_mask = get_variable_slice_mask(
                route, dim=1, froms=route_lens, 
                tos=route_lens + chosen_ext_lens)
            # select the chosen extensions
            exp_sel = selection[..., None].expand(-1, -1, candidates.shape[-1])
            chosen_seqs = candidates.gather(1, exp_sel).squeeze(1)
            # insert the chosen extensions
            select_mask = get_variable_slice_mask(
                chosen_seqs, dim=1, tos=chosen_ext_lens)
            route[insert_mask] = chosen_seqs[select_mask]
            # update which nodes are on the route
            node_on_route[batch_idxs[:, None], chosen_seqs] = True
            # make padding column False
            node_on_route[:, -1] = False
            route_times += cdt_times.gather(1, selection).squeeze(1)

            # TODO introspect on the below

            # update the extension mask based on the current route
            # make only segments continuing from this one valid
            route_lens += chosen_ext_lens
            end_nodes = route.gather(1, route_lens[..., None] - 1).squeeze()
            relevant_attn_embeds = path_attn_embeds[batch_idxs, end_nodes]
            ext_mask = base_ext_mask[batch_idxs, end_nodes]
            # trim the first node of each candidate, since it's the last node
             # on the current route
            candidates = seqs[batch_idxs, end_nodes][..., 1:]
            cdt_times = state.drive_times[batch_idxs, end_nodes]
            # invalidate segments that revisit stops on the route
            revisits = node_on_route[batch_idxs[:, None, None], candidates]
            revisits = revisits.any(-1)
            ext_mask = ext_mask | revisits

            # determine necessary halts
            must_halt = ext_mask.all(-1)
            is_done = is_done | must_halt
            
            # update the context vector and manual descriptors based on the 
             # current route
            route_descs = self.encode_path(node_descs, route, 
                                           route_times)
            context = torch.cat((global_feats, route_descs), dim=-1)            
            first_iter = False

        # prune extra route sequence padding
        max_route_len = (route > -1).sum(dim=-1).max()
        route = route[:, :max_route_len]

        route_lens = (route > -1).sum(-1)

        return route[:, None], logits[:, None], None

    def _get_extension_scores(self, context_vec, kv_embeds, key_mask=None):
        # compute scores vs current context and pick an extension path
        # add query "sequence" dimension
        context_vec = context_vec[..., None, :]
        path_chunks = kv_embeds.chunk(2 * self.n_selection_layers + 1, dim=-1)
        keys = path_chunks[::2]
        values = path_chunks[1::2] + (None,)

        # TODO implement multi-head attention
        for query_embedder, lk, lv in zip(self.query_embedders, keys, values):
            query = query_embedder(context_vec)
            scores = torch.bmm(query, lk.transpose(-1, -2)).squeeze(1)
            scores /= self.embed_dim**0.5
            if key_mask is not None:
                scores[key_mask] = TORCH_FMIN
            scores = torch.nn.functional.softmax(scores, dim=-1)
            if lv is not None:
                # the final layer scores are just the selection probabilities,
                 # so don't use them to update the context
                context_vec = torch.bmm(scores[:, None], lv)

        return context_vec.squeeze(1), scores
    
    def encode_path(self, node_descs, path, path_times):
        if path.ndim == 2:
            added_dim = True
            # add a route dimension
            path = path[:, None]
            path_times = path_times[:, None]
        else:
            added_dim = False
        path_mask = path == -1
        enc = self.path_encoder(node_descs, path, path_mask)

        path_lens = (~path_mask).sum(-1)
        extra_feats = torch.stack((path_times, path_lens), dim=-1)
        extra_feats = self.path_extras_norm(extra_feats)
        mlp_in = torch.cat((enc, extra_feats), dim=-1)
        path_desc = self.path_mlp(mlp_in)
        if added_dim:
            path_desc.squeeze_(1)
        return path_desc


class NodeWalker(RouteGeneratorBase):
    def __init__(self, *args, n_heads=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_node_scorer = get_mlp(3, self.embed_dim * 2, 
                                         in_dim=self.embed_dim, out_dim=1)
        self.next_node_scorer = get_mlp(3, self.embed_dim * 2, out_dim=1)
        self.next_node_attn = nn.MultiheadAttention(
            self.embed_dim, n_heads, self.dropout, batch_first=True)
        halt_desc = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.register_parameter(name="halt_desc", param=halt_desc)

    def _get_routes(self, n_routes, state, node_descs, extra_nodepair_feats,
                    greedy):
        """
        n_routes: int
        node_descs: batch_size x n_nodes x node_feat_dim
        state: batch_size x n_nodes x n_nodes x state_dim
        greedy: bool
        returns: batch_size x n_routes x route_len, all_logits, None
        """
        log.debug("get routes")
        # doesn't support chunk sizes greater than 1
        assert n_routes == 1

        # fold node descriptors
        node_descs, _ = self.fold_node_descs(node_descs, state)

        # pick first node
        node_scores = self.start_node_scorer(node_descs).squeeze(-1)
        cur_node, logits = select(node_scores, not greedy)
        route = cur_node
        cur_node = cur_node.squeeze(-1)

        # set up loop-tracking variables and other needed values
        is_done = torch.zeros(state.batch_size, device=state.device).bool()
        batch_idxs = torch.arange(state.batch_size, device=state.device)
        node_on_route = torch.zeros((state.batch_size, state.max_n_nodes + 1), 
                                    device=state.device).bool()
        adj_mat = state.graph_data.street_adj
        is_adj = adj_mat.isfinite() & (adj_mat > 0)
        halt_desc = self.halt_desc.expand(state.batch_size, -1, -1)

        # assemble a route!
        while not is_done.all():
            # compute scores vs current context and pick an extension path
            # assemble query vector: adjacent nodes and the halt option
            is_valid = is_adj[batch_idxs, cur_node] & ~node_on_route[..., :-1]
            is_valid = is_valid & ~is_done[:, None]
            valid_idxs = get_indices_from_mask(is_valid, dim=-1)
            cdt_feats = node_descs[batch_idxs[:, None], valid_idxs]
            if route.shape[-1] > 1:
                # don't enable halt option unless we have at least two stops
                cdt_feats = torch.cat((cdt_feats, halt_desc), dim=-2)
                valid_idxs = nn.functional.pad(valid_idxs, (0, 1), value=True)

            # select a node
            route_feats = node_descs[batch_idxs[:, None], route]
            route_pad_mask = route == -1
            cdt_descs, _ = self.next_node_attn(
                cdt_feats, route_feats, route_feats, 
                key_padding_mask=route_pad_mask, need_weights=False)
            cdt_descs = torch.cat((cdt_descs, cdt_feats), dim=-1)
            scores = self.next_node_scorer(cdt_descs).squeeze(-1)
            scores = get_update_at_mask(scores, valid_idxs == -1, TORCH_FMIN)
            cur_node, next_logit = select(scores, not greedy)
            cur_node = cur_node.squeeze(-1)

            # update route, node_on_route, logits, is_done
            if route.shape[-1] > 1:
                chose_halt = cur_node == cdt_feats.shape[-2] - 1
                is_done = is_done | chose_halt
                cur_node = get_update_at_mask(cur_node, chose_halt, -1)
                next_logit[chose_halt] = 0.0
                
            route = torch.cat((route, cur_node[:, None]), dim=1)
            logits += next_logit
            node_on_route[batch_idxs, cur_node] = True

        return route[:, None], logits, None


class BeamSearchRouteGenerator(ShortestPathRouteGenerator):
    def _get_routes(self, n_routes, graph_data, valid_terms_mat, are_connected,
                    folded_node_descs, nodepair_descs, greedy):
        nodepair_scores = self.nodepair_scorer(nodepair_descs).squeeze(-1)
        nodepair_scores = self.get_biased_scores(
            nodepair_scores, graph_data.demand, are_connected)

        seqs, scores = find_best_routes(nodepair_scores, graph_data.street_adj)

        scores = self.set_invalid_scores_to_fmin(scores, valid_terms_mat)
        # flatten the src and dst dimensions
        scores = scores.reshape(graph_data.num_graphs, -1)
        seqs = seqs.flatten(1, 2)
        
        flat_slctd_idxs, route_logits = \
            select(scores, scores_as_logits=not greedy, n_selections=n_routes)

        # gather new routes
        # add a sequence dimension
        exp_shape = (-1, -1, seqs.shape[-1])
        exp_idxs = flat_slctd_idxs[..., None].expand(*exp_shape)
        batch_new_routes = seqs.gather(1, exp_idxs)

        return batch_new_routes, route_logits, None


class LearnedRoutingGenerator(RouteGeneratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # plus 1 in in_dim for the candidate route distance
        self.nodepair_scorer = get_mlp(self.n_nodepair_layers, self.embed_dim, 
                                       in_dim=self.full_nodepair_dim, 
                                       out_dim=1)
        self.score_length_fn = nn.Sequential(
            get_mlp(2, 4, in_dim=2, out_dim=1)
        )
        # TODO try different numbers of heads
        self.nodepair_query_transform = nn.Linear(self.full_nodepair_dim, 
                                                  self.embed_dim)
        self.node_kv_transform = nn.Linear(self.embed_dim * 2, 
                                           self.embed_dim * 2)
        self.query_time_transform = nn.Linear(1, self.embed_dim)
        self.kv_dist_transform = nn.Linear(1, self.embed_dim * 2)
        node_dim = self.embed_dim * 2 + 1
        self.next_stop_attn = nn.MultiheadAttention(
            self.full_nodepair_dim + 1, self.n_heads, ATTN_DEFAULT_DROPOUT, 
            batch_first=True, kdim=node_dim, vdim=node_dim)
        self.next_stop_scorer = get_mlp(self.n_nodepair_layers, self.embed_dim, 
                                        out_dim=1)
        self.attn_dropout = nn.Dropout(ATTN_DEFAULT_DROPOUT)

        self.stop_logits_comb_mode = "cat"

    def plan_routes(self, node_descs, nodepair_descs, street_adj, drive_times,
                    greedy=False, srcs_and_dsts=None):
        """
        srcs_and_dsts: a 2xN tensor of source and destination nodes.
        """
        batch_size = node_descs.shape[0]
        max_n_nodes = node_descs.shape[1]
        dev = node_descs.device
        if srcs_and_dsts is None:
            # do all pairs
            ones = torch.ones((max_n_nodes, max_n_nodes), device=dev)
            srcs, dsts = torch.where(ones)
            srcs = srcs.expand(batch_size, -1)
            dsts = dsts.expand(batch_size, -1)
        else:
            srcs = srcs_and_dsts[..., 0]
            dsts = srcs_and_dsts[..., 1]

        # flatten batch and source-dest into a single sequence
        # TODO how to do this properly?  Make use of graph batch structure?
        # but we need to index into nodepair_descs appropriately...
        # if we assume there are the same number of srcs and dsts for all 
         # scenarios, does that help?  Probably, yes.
        n_routes = srcs.shape[-1]
        batch_idxs = torch.arange(batch_size, device=dev)
        # set up initial sequence (source nodes)
        cur_nodes = srcs.clone()
        seqs = cur_nodes[..., None]
        path_times = torch.zeros((batch_size, n_routes, 1, 1), device=dev)
        end_pl = torch.zeros_like(path_times)
        all_logits = torch.zeros_like(path_times[..., 0])
        route_scores = torch.zeros((batch_size, n_routes), device=dev)
        np_dim = nodepair_descs.shape[-1]
        np_scores = self.nodepair_scorer(nodepair_descs)

        query_vecs = self.nodepair_query_transform(nodepair_descs)
        kv_vecs = self.node_kv_transform(node_descs)

        # while current node is not goal node:
        while (cur_nodes != dsts).any():
            # get next node options
            maybe_nexts = street_adj[batch_idxs[:, None], cur_nodes]
            is_valid_next = maybe_nexts.isfinite() & (maybe_nexts > 0)
            # TODO prevent going back to previous node
            on_route_idxs = seqs.clone()
            on_route_idxs[on_route_idxs == -1] = 0
            is_valid_next.scatter_(2, on_route_idxs, 0)
            valid_idxs = get_indices_from_mask(is_valid_next, dim=-1)
            # preappend the dests, because we can always choose to go direct.
            valid_idxs = torch.cat((dsts[..., None], valid_idxs), dim=-1)

            # get edge descriptors from all nodes on route to candidate nodes
            exp_bi = batch_idxs[:, None, None]
            # clone to avoid overwriting street_adj
            next_times = \
                street_adj[exp_bi, cur_nodes[..., None], valid_idxs].clone()

            # set correct distance for going direct to dest from current node
            straighttodst_dist = drive_times[batch_idxs[:, None], cur_nodes,
                                             valid_idxs[..., 0]]
            next_times[..., 0] = straighttodst_dist.clone()
            
            # set invalid next-times to -1 to avoid infinities
            next_times[valid_idxs == -1] = -1
            # add a feature dimension
            next_times = next_times[..., None]

                # limit choices to those that are closer than current node???

            # compute scores for edges from each node on sequence to candidates
             # this uses the edge descriptors from each node on the sequence to
             # the new node.
            all_to_nexts_times = path_times[..., None, :] + \
                next_times[..., None, :, :]
            # attach path lengths to nodes on route, instead of seq aug
             # eg. first node gets 0, 2nd node gets its dist over route from 0,
             # 3rd gets its dist over route from 0, etc.
            exp_seqs = seqs[..., None]
            exp_vi = valid_idxs[..., None, :]
            base_np_scores = np_scores[exp_bi[..., None], exp_seqs, exp_vi]
            score_input = torch.cat((base_np_scores, all_to_nexts_times), 
                                    dim=-1)
            # TODO do some kind of batch norm here for the length features?
            pre_shape = score_input.shape
            flat_score_input = score_input.view(-1, 2)
            out_np_scores = self.score_length_fn(flat_score_input)
            out_np_scores = out_np_scores.squeeze(-1).reshape(pre_shape[:-1])
            # edges to invalid options get 0 score
            next_pad_mask = valid_idxs == -1
            out_np_scores = get_update_at_mask(out_np_scores, 
                                               next_pad_mask[:, :, None])
            # sum along the sequence dimension
            sofar_to_new_scores = out_np_scores.sum(dim=-2)
            # accumulate the computed edge scores

            # compute attention between next node options and current sequence
            # reverse path lengths for edge features to end of route so far
            q_pathlen_embed = self.query_time_transform(next_times)
            current_queries = query_vecs[exp_bi, valid_idxs, dsts[..., None]]
            full_queries = current_queries + q_pathlen_embed

            kv_pathlen_embed = self.kv_dist_transform(path_times)
            current_kvs = kv_vecs[exp_bi, seqs]
            full_kvs = current_kvs + kv_pathlen_embed

            # compute attention!
            # scale by square root of feature dimension
            full_queries = full_queries * full_queries.shape[-1] ** -0.5
            flat_queries = full_queries.flatten(end_dim=1)
            flat_kvs = full_kvs.flatten(end_dim=1)
            keys, values = flat_kvs.chunk(2, dim=-1)
            # compute dot products
            attn_weights = torch.bmm(flat_queries, keys.transpose(1, 2))
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_dropout(attn_weights)
            attn_outs = torch.bmm(attn_weights, values)

            # apply an MLP to the attn's output node descs to get their scores
            subsequent_scores = self.next_stop_scorer(attn_outs).squeeze(-1)
            flat_sofar_scores = sofar_to_new_scores.flatten(end_dim=1)
            # batch and src,dst dimensions are flattened into one in scores
            scores = flat_sofar_scores + subsequent_scores
            # set scores at padding locations to fmin
            flat_nextpadmask = next_pad_mask.flatten(end_dim=1)
            scores = get_update_at_mask(scores, flat_nextpadmask, TORCH_FMIN)

            # sample or pick max, depending on greedy
            next_valid_nodes, logits = select(scores, not greedy)
            next_valid_nodes = next_valid_nodes.reshape((batch_size, n_routes))
            next_nodes = valid_idxs.gather(-1, next_valid_nodes[..., None])
            next_nodes = next_nodes.squeeze(-1)

            # force choosing destination if it's an option, and staying at
             # destination when there
            # force next nodes to be the goal node if there are no valid nodes
            no_valids = ~is_valid_next.any(dim=-1) 
            assert (next_nodes[no_valids] == dsts[no_valids]).all()
            # if already at dest, stay there
            already_at_dst = cur_nodes == dsts
            next_nodes = get_update_at_mask(next_nodes, already_at_dst, dsts)

            # update path lengths
            gather_nvn = next_valid_nodes[..., None]
            next_times = next_times.gather(-2, gather_nvn[..., None])
            # update the path lengths
            path_times = path_times + next_times
            # add a new zero at the end for the new final node
            path_times = torch.cat((path_times, end_pl), dim=-2)

            # add the chosen node desc to the sequence
            next_node_descs = node_descs[batch_idxs[:, None], next_nodes]
            next_node_descs.unsqueeze_(-2)

            # update the estimated score of the routes so far
            chosen_sofar_scores = \
                sofar_to_new_scores.gather(2, gather_nvn).squeeze(-1)
            chosen_sofar_scores[already_at_dst] = 0
            route_scores += chosen_sofar_scores

            # update current nodes and index sequences
            cur_nodes = next_nodes

            # update sequence, setting dummy elements of the sequence to -1
            next_seq_elem = get_update_at_mask(next_nodes, already_at_dst, -1)
            seqs = torch.cat((seqs, next_seq_elem[..., None]), dim=-1)

            # store the logits
            logits = logits.reshape((batch_size, n_routes))
            # erase logits where they don't affect the choice
            logits = get_update_at_mask(logits, already_at_dst)
            all_logits = torch.cat((all_logits, logits[..., None]), dim=-1)

        # return the node sequences, route scores, and logits of the choices
        if srcs_and_dsts is None:
            # reshape outputs to be a square of src x dst
            seqs = seqs.reshape(batch_size, max_n_nodes, max_n_nodes, -1)
            route_scores = \
                route_scores.reshape(batch_size, max_n_nodes, max_n_nodes)
            all_logits = \
                all_logits.reshape(batch_size, max_n_nodes, max_n_nodes, -1)

        return seqs, route_scores, all_logits


class RoutingFirstRouteGenerator(LearnedRoutingGenerator):
    def _get_routes(self, n_routes, graph_data, valid_terms_mat, are_connected,
                    folded_node_descs, nodepair_descs, greedy):
        if self.only_routes_with_demand_are_valid:
            # plan only valid terms
            srcs_and_dsts = [torch.stack(torch.where(s_vtm), dim=-1)[None]
                             for s_vtm in valid_terms_mat]
            srcs_and_dsts = cat_var_size_tensors(srcs_and_dsts)
        else:
            # plan a route for every terminal pair
            srcs_and_dsts = None

        seqs, scores, stop_logits = \
            self.plan_routes(folded_node_descs, nodepair_descs, 
                             graph_data.street_adj, graph_data.drive_times, 
                             greedy, srcs_and_dsts)

        # pick among the planned routes according to their scores
        log.debug("selecting routes")
        if self.only_routes_with_demand_are_valid:
            # if the src and dst are the same, they are invalid.
            is_invalid = srcs_and_dsts[..., 0] == srcs_and_dsts[..., 1]
            scores = get_update_at_mask(scores, is_invalid, TORCH_FMIN)
        else:
            scores = self.set_invalid_scores_to_fmin(scores, valid_terms_mat)
            batch_size = graph_data.num_graphs
            # flatten the src and dst dimensions
            scores = scores.reshape(batch_size, -1)
            seqs = seqs.flatten(1, 2)
            stop_logits = stop_logits.flatten(1, 2)

        flat_slctd_idxs, route_logits = \
            select(scores, scores_as_logits=not greedy, n_selections=n_routes)

        # gather new routes
        # add a sequence dimension
        exp_shape = (-1, -1, seqs.shape[-1])
        exp_idxs = flat_slctd_idxs[..., None].expand(*exp_shape)
        batch_new_routes = seqs.gather(1, exp_idxs)

        if stop_logits is not None:
            # logits are 1 shorter than seqs, since no logit for start node
            stop_logits = stop_logits.gather(1, exp_idxs[..., :-1])

        return batch_new_routes, route_logits, stop_logits


class TermsFirstRouteGenerator(LearnedRoutingGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO tweak these parameters
        self.term_scorer = get_mlp(2, self.full_nodepair_dim, out_dim=1)
        
    def _get_routes(self, n_routes, graph_data, valid_terms_mat, are_connected,
                    folded_node_descs, nodepair_descs, greedy):
        scores = self.term_scorer(nodepair_descs)
        scores = self.set_invalid_scores_to_fmin(scores, valid_terms_mat)

        # pick r_chunk_size new routes from the set of demand edges
        log.debug("selecting routes")
        batch_size = graph_data.num_graphs
        flat_scores = scores.reshape(batch_size, -1)
        flat_slctd_idxs, route_logits = \
            select(flat_scores, scores_as_logits=not greedy, 
                    n_selections=n_routes)

        max_n_nodes = valid_terms_mat.shape[-1]
        srcs = flat_slctd_idxs // max_n_nodes
        dsts = flat_slctd_idxs % max_n_nodes
        srcs_and_dsts = torch.stack((srcs, dsts), dim=-1)

        batch_new_routes, _, stop_logits = \
            self.plan_routes(folded_node_descs, nodepair_descs, 
                             graph_data.street_adj, graph_data.drive_times, 
                             greedy, srcs_and_dsts)

        return batch_new_routes, route_logits, stop_logits


# Planner classes that incorporate all of the needed components.
class PlannerBase(nn.Module):
    def __init__(self, encoder, freq_selector):
        super().__init__()
        self.encoder = encoder
        self.freq_selector = freq_selector

    def forward(self, env_rep, route_reps, budget, greedy=False, **kwargs):
        return self.plan(env_rep, route_reps, budget, greedy, **kwargs)

    def plan(self):
        raise NotImplementedError()

    def set_pretrained_state_encoder(self, weights_path, freeze=True):
        self.encoder.load_state_encoder_weights(weights_path, freeze)

    @property
    def embed_dim(self):
        return self.encoder.embed_dim


class SeededRoutesPlanner(PlannerBase):
    def __init__(self, embed_dim, graph_encoder, route_selector, softmax_temp,
                 max_n_route_stops, n_routes_per_ep, n_routes_to_generate):
        self.n_routes_to_generate = n_routes_to_generate
        self.n_routes_per_ep = n_routes_per_ep
        route_generator = DemandBasedRouteGenerator(
            embed_dim, softmax_temp, max_n_route_stops)
        # TODO try with a simple non-sequential frequency chooser.  Might be
        # faster.
        super().__init__(graph_encoder, route_generator, route_selector)

    def forward(self, env_rep, seeds, greedy, base_mask=None):
        self.encode_graph(env_rep)
        return self.plan(seeds, greedy, base_mask=base_mask)

    def plan(self, seeds, greedy, batch_size=1, base_mask=None, **kwargs):
        seeds = seeds.reshape((-1, self.embed_dim))
        # use seeds to generate the routes
        gen_result = self.route_generator(greedy, self.n_routes_to_generate, 
            seeds, dist_mask=base_mask)
        # TODO may we need to force non-zero routes?
        route_index_range = \
            range(0, self.n_routes_to_generate, self.n_routes_per_ep)
        # fold the routes
        folded_routes = [gen_result.routes[ii:ii + self.n_routes_per_ep]
                         for ii in route_index_range]
        folded_valid_route_idxs = \
            [[ri for ri, rs in enumerate(batch_routes) if len(rs) > 0]
             for batch_routes in folded_routes]

        # fold route_descs
        route_descs = self._fold_batch(gen_result.route_descs, batch_size)

        # use route_selector to get freqs for all valid routes
        # first build a tensor selecting all valid routes
        route_idx_tnsr = torch.ones((batch_size, self.n_routes_per_ep + 1),
                                    device=seeds.device, dtype=int)
        # set all indices to termination route
        route_idx_tnsr *= self.n_routes_per_ep
        # insert indices of valid routes
        for bi in range(batch_size):
            idxs = torch.tensor(folded_valid_route_idxs[bi], 
                                device=seeds.device)
            route_idx_tnsr[bi, :len(idxs)] = idxs
        # _, freq_result = self.freq_selector(route_descs, greedy, batch_size, 
        #                                     route_idx_tnsr)

        # assemble and return results
        # fold the generation tensors to have the batch dimension
        stop_logits = self._fold_batch(gen_result.logits, batch_size)
        if gen_result.selections is not None:
            stops_tnsr = self._fold_batch(gen_result.selections, batch_size)
        else:
            stops_tnsr = None
        if gen_result.est_vals is not None:
            stop_est_vals = self._fold_batch(gen_result.est_vals, batch_size)
        else:
            stop_est_vals = None
        # discard frequencies for empty routes
        valid_routes = [[rr for rr in ep_routes if len(rr) > 0] 
                        for ep_routes in folded_routes]
        # freqs = [freq_result.freqs[bi][:len(batch_routes)]
        #          for bi, batch_routes in enumerate(valid_routes)]
        freqs = [[1/600 for _ in er] for er in valid_routes]
        return PlanResults(routes=valid_routes, freqs=freqs,
            stop_logits=stop_logits, route_logits=None, freq_logits=None,
            # freq_logits=freq_result.logits,
            stops_tensor=stops_tnsr, routes_tensor=route_idx_tnsr,
            freqs_tensor=None,
            # freqs_tensor=freq_result.selections,
            stop_est_vals=stop_est_vals, route_est_vals=None, 
            freq_est_vals=None,
            # freq_est_vals=freq_result.est_vals
            )

    def _fold_batch(self, tensor, batch_size):
        """During planning, we roll out all routes of all eps in the batch
           in parallel, which results in 2D tensors where the first dimension
           is batch_size * max_n_routes.  This method takes such a tensor and
           reshapes it into a 3D tensor where the first dimension corresponds
           to the episodes in the batch, and the second to the proposed routes
           in an episode."""
        return tensor.reshape((batch_size, self.n_routes_per_ep, -1))


class NoFreqPlanner(PlannerBase):
    def __init__(self,
                 embed_dim,
                 encoder,
                 learned_function_mode,
                 do_binary_rollout=False,
                ):
        super().__init__(encoder, None)
        # insist we're using fixed routes
        self.action_scorer = MlpActionScorer(embed_dim, 2,
                                             yes_and_no_outs=do_binary_rollout)
        self.learned_function_mode = learned_function_mode
        self.do_binary_rollout = do_binary_rollout

    def plan(self, env_rep, route_reps, budget, greedy, batch_size=1, 
             epsilon=0.0, mask=None, **kwargs):
        if greedy:
            assert epsilon == 0.0, \
                "greedy=True is incompatible with epsilon > 0"
            scores_as_logits = False
        elif self.learned_function_mode == "q function":
            scores_as_logits = False
        elif self.learned_function_mode == "gflownet":
            scores_as_logits = epsilon == 0.0
        elif self.learned_function_mode == "policy":
            scores_as_logits = True

        route_idxs, scores = \
            self.rollout(env_rep, route_reps, budget, batch_size,
                         epsilon, scores_as_logits=scores_as_logits, mask=mask)
        
        max_scen_len = max([len(rr) for rr in route_idxs])
        null_idx = len(route_reps)
        routes_tensor = torch.ones((batch_size, max_scen_len), dtype=int)
        routes_tensor *= null_idx
        batch_routes = []
        for ep_idx in range(batch_size):
            # cut off "halt" selections
            ep_selections = route_idxs[ep_idx]
            ep_routes = [route_reps[ri].route for ri in ep_selections
                         if ri < len(route_reps)]
            batch_routes.append(ep_routes)
            routes_tensor[ep_idx, :len(ep_routes)] = ep_selections

        if self.learned_function_mode == "policy":
            return PlanResults(
                routes=batch_routes, freqs=None,
                stop_logits=None,
                route_logits=scores,
                freq_logits=None,
                stops_tensor=None,
                routes_tensor=routes_tensor,
                freqs_tensor=None,
                stop_est_vals=None,
                route_est_vals=None,
                freq_est_vals=None)
        else:
            return PlanResults(
                routes=batch_routes, freqs=None,
                stop_logits=None,
                route_logits=None,
                freq_logits=None,
                stops_tensor=None,
                routes_tensor=routes_tensor,
                freqs_tensor=None,
                stop_est_vals=None,
                route_est_vals=scores,
                freq_est_vals=None)

    def greedy_flow_plan(self, env_rep, route_reps, budget, n_samples=64, 
                         mask=None):
        # generate n_sample rollouts according to the flow policy
        samples, est_q_values = \
            self.rollout(env_rep, route_reps, budget, n_samples, 
                         scores_as_logits=True, mask=mask)
        # estimate their quality by the inflow to each generated plan
        parent_q_ests = \
            self.estimate_parent_qs(env_rep, route_reps, budget, samples)
        sample_inflows = torch.stack([qe.logsumexp(0) for qe in parent_q_ests])
        best_sample_idx = sample_inflows.argmax()
        # return the highest estimated quality
        best_scenario = samples[best_sample_idx]
        # cut off "halt" selections
        routes = [route_reps[ri].route for ri in best_scenario
                  if ri < len(route_reps)]
        routes_tensor = torch.ones((1, len(best_scenario)), dtype=int)
        routes_tensor[0, :len(best_scenario)] = best_scenario
        best_q_ests = est_q_values[best_sample_idx, :len(best_scenario)]
        return PlanResults(
            routes=routes, freqs=None,
            stop_logits=None,
            route_logits=None,
            freq_logits=None,
            stops_tensor=None,
            routes_tensor=routes_tensor,
            freqs_tensor=None,
            stop_est_vals=None,
            route_est_vals=best_q_ests, freq_est_vals=None)

    def rollout(self, env_rep, route_reps, budget, batch_size=1, epsilon=0.0, 
                scores_as_logits=False, mask=None):
        """
        route_reps: a list of RouteRep objects for the candidate routes
        budget: the budget for the rollout
        epsilon: parameter of epsilon-greedy selection policy
        scores_as_logits: if True, choose actions stochastically according to 
            the flow-based policy where the estimated Q values are treated as 
            the reward flows, as in the GFlowNets paper of Bengio et al. 2021.
            If True, epsilon must not be specified.
        mask: if provided, a n_routes boolean tensor that is True for routes 
            that aren't valid choices
        """
        dev = route_reps[0].device
        n_candidate_routes = len(route_reps)
        if n_candidate_routes == 0:
            # no routes were supplied, so there's nothing to choose from.
            return None, None

        # keep remaining_budget as a 1D batch_size tensor
        if type(budget) in (int, float) or budget.shape[0] == 1:
            remaining_budget = torch.full(batch_size, budget, device=dev)
        else:
            remaining_budget = budget.clone()
        scenarios = [[] for _ in range(batch_size)]
        scores = []

        if self.do_binary_rollout:
            for cdt_idx, route_rep in enumerate(route_reps):
                are_included, act_scores = \
                    self.step(env_rep, route_reps, scenarios, remaining_budget,
                        epsilon, scores_as_logits, mask=mask, 
                        action_cdt_idx=cdt_idx)
                scores.append(act_scores)
                for ep_idx, is_included in enumerate(are_included):
                    if is_included:
                        scenarios[ep_idx].append(cdt_idx)
                        remaining_budget[ep_idx] -= route_rep.cost
            
        else:
            is_valid_action = \
                get_valid_actions_tensor(route_reps, remaining_budget, 
                                        scenarios, mask)
            while is_valid_action.any():
                # choose actions epsilon-greedily
                next_route_idxs, act_scores = \
                    self.step(env_rep, route_reps, scenarios, remaining_budget, 
                             epsilon, scores_as_logits, mask=mask)
                scores.append(act_scores)
                
                # update state
                for ep_idx, route_idx in enumerate(next_route_idxs):
                    if route_idx < n_candidate_routes:
                        scenarios[ep_idx].append(route_idx)
                        remaining_budget[ep_idx] -= route_reps[route_idx].cost

                is_valid_action = \
                    get_valid_actions_tensor(route_reps, remaining_budget, 
                                            scenarios, mask)

        scenarios = [torch.tensor(scenario, device=dev) 
                     for scenario in scenarios]
        scores = torch.stack(scores).t()
        return scenarios, scores

    def step(self, env_rep, cdt_route_reps, batch_states, 
             remaining_budgets, epsilon=0.0, scores_are_logits=False, 
             next_actions=None, mask=None, action_cdt_idx=None):
        """
        env_rep: representation of the environment
        cdt_route_reps: a list of RouteReps representing candidate routes,
            or a batch_size list of lists of RouteReps
        batch_states: batch_size-len list of indices of routes that have been
            chosen so far
        remaining_budgets: batch_size tensor
        epsilon: probability of uniform-random sampling.
        scores_are_logits: if True, treat the scores as action logits and 
            sample the action according to them.  If False, simply take the 
            action with maximum score with probability 1 - epsilon.
        next_actions: if provided, a batch_size tensor indicating what will 
            be chosen in this step, overriding whatever choice would be made
        mask: if provided, a n_routes or batch_size x n_routes boolean tensor 
            that is True for routes that aren't valid choices
        """
        # only one of these arguments should be provided; if more than
         # one is not the default, raise an exception.
        if epsilon > 0:
            assert next_actions is None and not scores_are_logits, \
            "mutually-exclusive arguments provided!"
        
        assert self.do_binary_rollout == (action_cdt_idx is not None), \
            "action candidate index must be provided if and *only* if "\
            "planner is in binary-rollout mode."

        batch_size = len(batch_states)
        dev = env_rep.stop_data.x.device
        batch_idxs = torch.arange(batch_size, device=dev)

        if type(cdt_route_reps[0]) is RouteRep:
            cdt_route_reps = [cdt_route_reps] * batch_size

        n_candidate_routes = len(cdt_route_reps[0])
        state_route_reps = [[cdt_route_reps[bi][ii] for ii in cr]
                            for bi, cr in enumerate(batch_states)]
        state_enc = self.encoder.encode_state(env_rep, state_route_reps, 
                                              remaining_budgets)

        if next_actions is not None:
            is_valid_action = torch.zeros((batch_size, n_candidate_routes),
                                          device=dev, dtype=bool)
            is_valid_action[batch_idxs, next_actions] = True
        else:
            is_valid_action = \
                get_valid_actions_tensor(cdt_route_reps, remaining_budgets, 
                                         batch_states, mask)

        if self.do_binary_rollout:
            if type(action_cdt_idx) is int:
                action_cdt_idx = torch.tensor([action_cdt_idx] * batch_size,
                                              device=dev)
            act_cdts = [[cdrs[action_cdt_idx[ei]]] 
                        for ei, cdrs in enumerate(cdt_route_reps)]
            action_descs = self.encoder.encode_actions(state_enc, act_cdts)
            # select only the validity of the action being considered
            if is_valid_action is not None:
                is_valid_action = is_valid_action[batch_idxs, action_cdt_idx]
                is_valid_action = is_valid_action[:, None]
            if epsilon > 0:
                total_remaining_cost = torch.tensor(
                    [sum([rr.cost for rr in crs[action_cdt_idx[ei]:] 
                          if rr.cost <= rb])
                    for ei, (rb, crs) 
                        in enumerate(zip(remaining_budgets, cdt_route_reps))], 
                    device=dev)

        else:
            action_descs = \
                self.encoder.encode_actions(state_enc, cdt_route_reps)

        if len(action_descs.shape) == 2:
            # add a batch dimension if one is not present
            action_descs = action_descs[None]

        # compute scores
        scores = self.action_scorer(action_descs, state_enc.global_desc, 
                                    is_valid_action)
        if self.do_binary_rollout:
            # remove the dimension for different routes, since we're only
             # considering one route
            scores = scores.squeeze(1)
        else:
            # remove the feature dimension
            scores = scores.squeeze(-1)

        if next_actions is not None and self.do_binary_rollout:
            # pick the right (don't-include, include) prob in chosen_scores
            chosen_scores = scores[batch_idxs, next_actions.to(dtype=int)]
        else:
            if self.do_binary_rollout:
                next_actions = torch.zeros(batch_size, device=dev, dtype=int)
            else:
                next_actions = torch.ones(batch_size, device=dev, dtype=int)
                next_actions *= n_candidate_routes

            chosen_scores = torch.full(batch_size, TORCH_FMIN, device=dev)
            valid_batch_idxs = batch_idxs[is_valid_action.any(dim=1)]
            valid_scores = scores[valid_batch_idxs]
            if scores_are_logits:
                # choose in proportion to probabilities
                if valid_scores.numel() > 0:
                    # some of these options are valid
                    valid_next_route_idxs = \
                        Categorical(logits=valid_scores).sample()
                    next_actions[valid_batch_idxs] = valid_next_route_idxs
                    chosen_scores[valid_batch_idxs] = \
                        scores[valid_batch_idxs, valid_next_route_idxs]
            else:
                # choose greedily
                choices = scores[valid_batch_idxs].argmax(dim=-1)
                next_actions[valid_batch_idxs] = choices
                chosen_scores[valid_batch_idxs] = \
                    scores[valid_batch_idxs, choices]

            if epsilon > 0:
                # randomize with epsilon
                pick_randomly = torch.rand(batch_size, device=dev) < epsilon
                # don't pick randomly if there's nothing to pick from
                pick_randomly &= is_valid_action.any(dim=1)
                if pick_randomly.any():
                    weights = is_valid_action[pick_randomly].to(dtype=float)
                    if self.do_binary_rollout:
                        # probability of picking any route is proportional to 
                         # how much budget is left, so distribution is roughly
                         # uniform over remaining routes
                        epsln_probs = remaining_budgets / total_remaining_cost
                        epsln_probs.clip_(max=1.0)
                        weights *= epsln_probs[pick_randomly]
                        # make weights for not including vs. including
                        weights = torch.cat((1 - weights, weights), dim=-1)

                    next_actions[pick_randomly] = \
                        torch.multinomial(weights, 1).squeeze(1)

        return next_actions, chosen_scores

    def estimate_parent_qs(self, env_rep, route_reps, remaining_budget, 
                           batch_chosen_routes):
        batch_size = len(batch_chosen_routes)
        dev = route_reps[0].device
        route_costs = torch.tensor([rr.cost for rr in route_reps], device=dev)
        route_costs = route_costs[None].tile(batch_size, 1)
        prev_state_routereps = []
        acts_from_parents = []
        for chosen_routes in batch_chosen_routes:
            chosen_route_reps = [route_reps[cr] for cr in chosen_routes]
            for ii in range(len(chosen_routes)):
                prev_chosen = chosen_route_reps[:ii] + chosen_route_reps[ii+1:]
                prev_state_routereps.append(prev_chosen)
                acts_from_parents.append([chosen_route_reps[ii]])

        state_enc = self.encoder.encode_state(env_rep, prev_state_routereps)
        action_descs = self.encoder.encode_actions(state_enc, acts_from_parents)
        if len(action_descs.shape) == 2:
            action_descs = action_descs[None]

        if type(remaining_budget) in (int, float) or \
            remaining_budget.size()[0] == 1:
            remaining_budget = torch.full(batch_size, remaining_budget, 
                                          device=dev)
        else:
            remaining_budget = remaining_budget.clone()
        seq_lens = [len(cr) for cr in batch_chosen_routes]
        exp_batch_idxs = np.cumsum([0] + seq_lens)
        expanded_batch_size = sum(seq_lens)
        
        parent_budgets = torch.zeros(expanded_batch_size, device=dev)
        for ep_idx, chosen_routes in enumerate(batch_chosen_routes):
            # chosen_descs = action_descs[ep_idx, chosen_routes]
            sei = exp_batch_idxs[ep_idx]
            eei = exp_batch_idxs[ep_idx + 1] 
            # action_vecs[sei:eei, 0] = chosen_descs
            parent_budgets[sei:eei] = remaining_budget[ep_idx] + \
                route_costs[ep_idx, chosen_routes]
            # padding_mask[sei:eei, len(chosen_routes):] = True

        # run networks forward to estimate the q values
        q_est = self.action_scorer.score_actions(action_descs, parent_budgets, 
                                                 state_enc.global_desc)
        q_est = q_est.squeeze()
        # reshape for output
        q_est = [q_est[exp_batch_idxs[ep_idx]:exp_batch_idxs[ep_idx+1]]
                 for ep_idx in range(len(batch_chosen_routes))]
        # return a list of tensors, one for each batch element, containing the
         # incoming q estimates from each parent
        return q_est 


# primitive modules


class LatentAttentionEncoder(nn.Module):
    def __init__(self, embed_dim, latent_size=None, n_heads=8, n_layers=2,
                 dropout=ATTN_DEFAULT_DROPOUT):
        super().__init__()
        # the latent embedding.  We don't learn it, but we make it a parameter
         # so that it will be included in the module's state_dict.
        if latent_size is not None:
            latent = nn.Parameter(torch.randn((1, latent_size, embed_dim)))
            self.register_parameter(name="latent", param=latent)
            # self.latent.requires_grad_(False)
        else:
            self.latent = None
        
        self.nonlin = DEFAULT_NONLIN()
        self.attention_layers = nn.ModuleList(
            nn.MultiheadAttention(embed_dim, n_heads, dropout, 
                                  batch_first=True)
            for _ in range(n_layers)
        )

    def forward(self, seq_to_encode, query=None, padding_mask=None, 
                embed_pos=False, seqlen_scale=None, residual=False):
        """
        seq_to_encode: batch_size x sequence length x embed dim tensor of the
            sequence(s) to encode
        padding_mask: batch_size x sequence length binary tensor indicating
            which elements of seq_to_encode are padding elements
        embed_pos: if True, sinusoidal position embeddings will be added to the
            sequence before encoding it.
        seqlen_scale: if provided, scale the encoding proportionally to the
            number of nodes, making the latent encoding a weighted sum rather 
            than a weighted average; the value of seqlen_scale is the 
            proportionality constant.
        residual: if True, the output of each layer is added to the query to
            get the query to the next layer; if False, the output alone is used
            as the query.
        """
        if seq_to_encode is None:
            # can't encode an empty sequence, so the "encoding" is just the
            # latent vector.
            return self.latent

        if seq_to_encode.ndim == 2:
            # there's no batch dimension, so add one
            seq_to_encode = seq_to_encode[None]

        if self.latent is None:
            assert query is not None, \
                "query_base must be provided if there is no latent vector!"
            
        dev = seq_to_encode.device
        batch_size = seq_to_encode.shape[0]
        if self.latent is not None and query is None:
            batch_latent = self.latent.tile(batch_size, 1, 1).to(device=dev)
            if seq_to_encode.shape[1] == 0:
                # can't encode an empty sequence, so the "encoding" is just the
                # latent vector.
                return batch_latent
            encoding = batch_latent
        else:
            encoding = query

        if embed_pos:
            sinseq = get_sinusoid_pos_embeddings(seq_to_encode.shape[1], 
                                                 seq_to_encode.shape[2])
            if seq_to_encode.ndim == 3:
                sinseq = sinseq[None]
            seq_to_encode = seq_to_encode + sinseq.to(device=dev)

        # ignore empty sequences
        if padding_mask is not None:
            seq_is_empty = padding_mask.all(dim=1)
            padding_mask = padding_mask[~seq_is_empty]
            encoding = encoding[~seq_is_empty]
            seq_to_encode = seq_to_encode[~seq_is_empty]
            seq_lens = padding_mask.shape[-1] - padding_mask.sum()
        else:
            seq_lens = torch.ones((batch_size, 1), device=dev)
            seq_lens *= seq_to_encode.shape[-2]

        # encode the input sequences via attention
        for attn_layer in self.attention_layers:
            attn_vec, _ = attn_layer(encoding, seq_to_encode, seq_to_encode,
                                     key_padding_mask=padding_mask)
            if seqlen_scale:
                attn_vec *= seqlen_scale * seq_lens

            if residual:
                encoding = encoding + attn_vec
            else:
                encoding = attn_vec

            if attn_layer is not self.attention_layers[-1]:
                # apply a non-linearity in between attention layers
                encoding = self.nonlin(encoding)
        
        if padding_mask is not None and seq_is_empty.any():
            batch_latent[~seq_is_empty] = encoding
            encoding = batch_latent
        
        return encoding


class EdgeGraphNetLayer(MessagePassing):
    def __init__(self, in_node_dim, in_edge_dim=0, 
                 out_dim=None, hidden_dim=None, bias=True, 
                 nonlin_type=DEFAULT_NONLIN, n_edge_layers=1, 
                 dropout=MLP_DEFAULT_DROPOUT, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        if hidden_dim is None:
            hidden_dim = in_node_dim
        if out_dim is None:
            out_dim = in_node_dim
        in_dim = in_node_dim * 2 + in_edge_dim
        layers = [
            get_mlp(n_edge_layers, hidden_dim, nonlin_type, dropout=dropout, 
                    bias=bias, in_dim=in_dim, out_dim=out_dim),
        ]
        if n_edge_layers == 1:
            layers.append(nonlin_type())
        self.edge_embedder = nn.Sequential(*layers)

    def forward(self, node_features, edge_index, edge_features):
        if edge_features is not None and edge_features.ndim == 1:
            # edge features have one channel, so ensure there's a dim for them
            edge_features = edge_features[:, None]
        edge_features = self.edge_updater(edge_index, x=node_features, 
                                          edge_attr=edge_features)
        size = (node_features.shape[0], node_features.shape[0])
        node_features = self.propagate(edge_index, edge_attr=edge_features,
                                       size=size)

        return node_features, edge_features

    def message(self, edge_attr):
        return edge_attr

    def edge_update(self, x_i, x_j, edge_attr):
        in_vec = torch.cat((x_i, x_j, edge_attr), dim=-1)
        embed = self.edge_embedder(in_vec)
        return embed


class GraphNetBase(nn.Module):
    def __init__(self, n_layers, embed_dim, in_node_dim=None, in_edge_dim=None, 
                 out_dim=None, nonlin_type=DEFAULT_NONLIN, 
                 dropout=ATTN_DEFAULT_DROPOUT, recurrent=False, residual=False,
                 dense=False, n_proj_layers=1, use_norm=True, layer_kwargs={}):
        """nonlin type: nonlinearity, or the string name the non-linearity to 
            use."""
        super().__init__()

        # do some input checking
        assert not (residual and dense)
        assert not (recurrent and dense)

        # set up members
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        self.in_node_dim = embed_dim if in_node_dim is None else in_node_dim
        self.in_edge_dim = in_edge_dim
        self.out_dim = embed_dim if out_dim is None else out_dim
        self.recurrent = recurrent
        self.residual = residual
        self.dense = dense
        self.dropout_rate = dropout
        self.dropout = nn.Dropout(dropout)
        if type(nonlin_type) is str:
            self.nonlin_type = getattr(nn, nonlin_type)
        else:
            self.nonlin_type = nonlin_type
        self.nonlin = self.nonlin_type()
        self.embed_dim = embed_dim
        self.layer_kwargs = layer_kwargs

        self.node_in_proj = nn.Identity()
        self.edge_in_proj = nn.Identity()
        self.node_out_proj = nn.Identity()
        self.edge_out_proj = nn.Identity()

        if recurrent:
            if self.in_node_dim != self.embed_dim:
                # project nodes to embedding dimension
                self.node_in_proj = nn.Linear(self.in_node_dim, self.embed_dim)
            if self.gives_edge_features and self.in_edge_dim != self.embed_dim:
                # project edges to embedding dimension
                self.edge_in_proj = nn.Linear(self.in_edge_dim, self.embed_dim)

        final_nodedim = self._get_final_nodedim()
        if self.out_dim != final_nodedim:
            # projected embedded nodes to output dimension
            self.node_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                         in_dim=final_nodedim, 
                                         out_dim=self.out_dim)
            if self.gives_edge_features:
                self.edge_out_proj = get_mlp(n_proj_layers, self.embed_dim,
                                             in_dim=self._get_final_edgedim(),
                                             out_dim=self.out_dim)

        # regardless of whether the network is recurrent, we still use separate
         # normalization layers at each step
        self.use_norm = use_norm
        if use_norm:
            norm_layers = [GraphNorm(self.embed_dim) 
                        for _ in range(self.n_layers - 1)]
            self.node_norm_layers = nn.ModuleList(norm_layers)
            if self.gives_edge_features:
                norm_layers = [BatchNorm(self.embed_dim) 
                            for _ in range(self.n_layers - 1)]
                self.edge_norm_layers = nn.ModuleList(norm_layers)

    def _get_layer_node_indims(self):
        return _layer_indims_helper(self.in_node_dim, self.embed_dim, 
                                    self.n_layers, self.dense)

    def _get_layer_edge_indims(self):
        return _layer_indims_helper(self.in_edge_dim, self.embed_dim,
                                    self.n_layers, self.dense)

    def _get_final_nodedim(self):
        indims = self._get_layer_node_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    def _get_final_edgedim(self):
        indims = self._get_layer_edge_indims()
        if self.dense:
            return indims[-1] + self.embed_dim
        else:
            return self.embed_dim

    @property
    def gives_edge_features(self):
        raise NotImplementedError

    def forward(self, data):
        data = self.preprocess(data)
        if self.gives_edge_features:
            output = self._forward_helper(data)
        else:
            output = self._forward_helper(data), None
        return self.postprocess(*output)

    def preprocess(self, data):
        data.x = self.node_in_proj(data.x)
        if data.edge_attr is not None:
            data.edge_attr = self.edge_in_proj(data.edge_attr)
        return data
    
    def postprocess(self, node_embeds, edge_embeds):
        node_embeds = self.node_out_proj(node_embeds)
        if edge_embeds is not None:
            edge_embeds = self.edge_out_proj(edge_embeds)
        if self.gives_edge_features:
            return node_embeds, edge_embeds
        else:
            return node_embeds

    def _forward_helper(self, data):
        node_descs = data.x
        edge_descs = data.edge_attr

        for li, layer in enumerate(self.layers):
            # apply the layer and graph normalization
            out = layer(node_descs, data.edge_index, edge_descs)
            if self.gives_edge_features:
                layer_nodes, layer_edges = out
            else:
                layer_nodes, layer_edges = out, None

            if li < len(self.layers) - 1:
                # this is not the last layer, so apply dropout and nonlinearity
                if self.use_norm:
                    layer_nodes = self.node_norm_layers[li](layer_nodes, 
                                                            data.batch)
                layer_nodes = self.nonlin(self.dropout(layer_nodes))
                if layer_edges is not None:
                    if self.use_norm:
                        layer_edges = self.edge_norm_layers[li](layer_edges)
                    layer_edges = self.nonlin(self.dropout(layer_edges))

            # get next layer's inputs based on this layer's outputs
            if self.residual and node_descs.shape[-1] == layer_nodes.shape[-1]:
                # add the input to the output
                node_descs = node_descs + layer_nodes
                if layer_edges is not None:
                    edge_descs = edge_descs + layer_edges
            elif self.dense:
                # concatenate the input to the output
                node_descs = torch.cat((node_descs, layer_nodes), dim=-1)
                if layer_edges is not None:
                    edge_descs = torch.cat((edge_descs, layer_edges), dim=-1)
            else:
                # just replace the input with the output
                node_descs = layer_nodes
                if layer_edges is not None:
                    edge_descs = layer_edges

        assert node_descs.shape[-1] == self._get_final_nodedim()

        # return the outputs
        if not self.gives_edge_features:
            # don't return the same edge features
            return node_descs
        else:
            return node_descs, edge_descs


class NoOpGraphNet(GraphNetBase):
    def __init__(self, return_edges=False, *args, **kwargs):
        self.return_edges = return_edges
        super().__init__(0, 0)

    def forward(self, data):
        if self.gives_edge_features:
            return data.x, data.edge_attr
        else:
            return data.x

    @property
    def gives_edge_features(self):
        return self.return_edges


class WeightedEdgesNetBase(GraphNetBase):
    def __init__(self, in_edge_dim=1, *args, **kwargs):
        super().__init__(in_edge_dim=1, *args, **kwargs)
        if in_edge_dim > 1:
            # project to range 0 to 1
            self.edge_in_proj = nn.Sequential(
                nn.Linear(in_edge_dim, 1),
                nn.Sigmoid()
            )

    @property
    def gives_edge_features(self):
        return False


class SimplifiedGcn(WeightedEdgesNetBase):
    def __init__(self, n_layers, in_node_dim, out_dim, *args, **kwargs):
        super().__init__(n_layers=n_layers, embed_dim=out_dim, 
                         in_node_dim=in_node_dim, out_dim=out_dim, 
                         recurrent=False, residual=False, dense=False, 
                         *args, **kwargs)
        self.net = SGConv(in_node_dim, out_dim, n_layers)

    def _forward_helper(self, data):
        act1 = self.net(data.x, data.edge_index, data.edge_attr)
        act2 = self.nonlin(act1)
        act3 = self.dropout(act2)
        return act3
    

class Gcn(WeightedEdgesNetBase):
    def __init__(self, layer_type=GCNConv, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.recurrent:
            layers = [layer_type(self.embed_dim, self.embed_dim, 
                                 **self.layer_kwargs)] * self.n_layers
        else:            
            in_dims = self._get_layer_node_indims()
            layers = [layer_type(in_dim, self.embed_dim, **self.layer_kwargs)
                      for in_dim in in_dims]
            
        self.layers = nn.ModuleList(layers)


class GraphAttnNet(GraphNetBase):
    def __init__(self, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads
        assert self.embed_dim % n_heads == 0, \
            'embed_dim is not divisible by n_heads!'
        head_dim = self.embed_dim // self.n_heads
        make_layer = lambda in_dim: \
            GATv2Conv(in_dim, head_dim, self.n_heads, 
                      edge_dim=self.in_edge_dim, **self.layer_kwargs)
        
        if self.recurrent:
            layers = [make_layer(self.embed_dim)] * self.n_layers
        else:
            in_dims = self._get_layer_node_indims()
            layers = [make_layer(in_dim) for in_dim in in_dims]

        self.layers = nn.ModuleList(layers)

    @property
    def gives_edge_features(self):
        return False


class EdgeGraphNet(GraphNetBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        make_layer = lambda ind, ied: \
            EdgeGraphNetLayer(ind, ied, out_dim=self.embed_dim, 
                              nonlin_type=self.nonlin_type,
                              dropout=self.dropout_rate, **self.layer_kwargs)
        
        if self.recurrent:
            layers = [make_layer(self.embed_dim, self.embed_dim)] * \
                self.n_layers
        else:
            in_node_dims = self._get_layer_node_indims()
            in_edge_dims = self._get_layer_edge_indims()
            in_edge_dims[0] = self.in_edge_dim
            layers = [make_layer(ind, ied) for ind, ied in 
                      zip(in_node_dims, in_edge_dims)]

        self.layers = nn.ModuleList(layers)


    @property
    def gives_edge_features(self):
        return True


# helper functions


def _layer_indims_helper(in_dim, embed_dim, n_layers, is_dense):
    indims = []
    if is_dense:
        curdim = in_dim
        for _ in range(n_layers):
            indims.append(curdim)
            curdim += embed_dim
    else:
        indims = [in_dim] 
        indims += [embed_dim] * (n_layers - 1)
    return indims


def mean_pool_sequence(sequences, padding_mask=None):
    # 2nd-to-last dim is sequence dim, last dim is feature dim
    if padding_mask is not None:
        sequences = sequences * ~padding_mask[..., None]
        seq_lens = (~padding_mask).sum(dim=-1)[..., None]
    else:
        seq_lens = sequences.shape[-2]
    # avoid generating NaNs by dividing by 0
    seq_lens[seq_lens == 0] = 1
    sums = sequences.sum(dim=-2)
    means = sums / seq_lens
    return means


def get_sinusoid_pos_embeddings(seqlen, ndims, posenc_min_rate=1/10000):
    angle_rate_exps = torch.linspace(0, 1, ndims // 2)
    angle_rates = posenc_min_rate ** angle_rate_exps
    positions = torch.arange(seqlen)
    angles_rad = positions[:, None] * angle_rates[None, :]
    sines = torch.sin(angles_rad)
    cosines = torch.cos(angles_rad)
    return torch.cat((sines, cosines), dim=-1)


def assemble_batch_routes_from_tensors(tensors):
    """
    Takes a list of batch_size x n_routes_in_tensor x max_n_stops_in_tensor
        tensors.  Assembles them into a single tensor of shape
        batch_size x total_n_routes x max_n_stops_across_tensors.
    """
    if tensors is None or len(tensors) == 0 or tensors[0] is None:
        # an "empty" input, so return None
        return None
    batch_size = tensors[0].shape[0]
    dev = tensors[0].device
    max_route_len = max([tt.shape[-1] for tt in tensors])
    n_routes = sum([tt.shape[-2] for tt in tensors])
    out = torch.zeros((batch_size, n_routes, max_route_len), device=dev)
    routes_so_far = 0
    for tt in tensors:
        end_idx = routes_so_far + tt.shape[-2]
        out[:, routes_so_far:end_idx, :tt.shape[-1]] = tt
        routes_so_far = end_idx
    return out


def select(scores, scores_as_logits=False, softmax_temp=1,
           n_selections=1, selection=None):
    """scores: an n_options or batch_size x n_options tensor."""
    if scores.ndim == 1:
        # add a batch dimension
        scores = scores[None, :]
    logits = logsoftmax(scores, softmax_temp).to(dtype=torch.float64)
    if selection is None:
        if not scores_as_logits:
            _, selection = logits.topk(n_selections)
        else:
            probs = logits.exp()
            selection = probs.multinomial(n_selections)
            # if fewer than n_selections options survive rounding down by exp()
             # then use topk to select the top n_selections
            needs_topk = (probs > 0).sum(dim=1) < n_selections
            if needs_topk.any():
                _, topk_selection = logits[needs_topk].topk(n_selections)
                selection[needs_topk] = topk_selection

        selection.squeeze_(-1)
    
    if n_selections == 1:
        # add a "selection" dimension
        selection = selection[..., None]
        if scores.ndim == 1:
            selected_logits = logits[selection]
        else:
            # there is a batch dimension
            selected_logits = logits.gather(1, selection)

    else:
        # renormalize the log-probabilities after each selection
        idxs = torch.arange(logits.shape[1]).expand(logits.shape[0], -1)
        idxs = idxs.to(scores.device)
        selection_mask = idxs[..., None] == selection[:, None]
        selection_mask = selection_mask.any(dim=-1)
        selection_neutralizer = torch.zeros_like(scores)
        selection_neutralizer[selection_mask] = TORCH_FMIN
        unselected_scores = scores + selection_neutralizer
        if scores.ndim == 1:
            selected_scores = scores[selection]
        else:
            # there is a batch dimension
            selected_scores = scores.gather(1, selection)

        # flip the selected scores so the first selected comes last
        reordered_scores = \
            torch.cat((unselected_scores, selected_scores.flip(-1)), dim=-1)
        # sum them so the last element is all scores, 2nd last is all but
         # first picked, etc.
        logsumexps = torch.logcumsumexp(reordered_scores, -1)
        # flip again so the first selected comes first
        logdenoms = logsumexps[..., -n_selections:].flip(-1)
        selected_logits = selected_scores - logdenoms

    return selection, selected_logits


def logsoftmax(energies, softmax_temp=1):
    return nn.functional.log_softmax(energies / softmax_temp, dim=-1)    


def expand_batch_dim(tnsr_to_expand, batch_size, dim=0):
    if tnsr_to_expand.shape[dim] != 1:
        # there's no unitary dimension at the start, so add one
        tnsr_to_expand = tnsr_to_expand.unsqueeze(dim)

    repeats = [1] * tnsr_to_expand.dim()
    repeats[dim] = batch_size
    expanded = tnsr_to_expand.repeat(repeats)
    return expanded


def get_valid_actions_tensor(route_reps, remaining_budgets, chosen_routes, 
                             mask=None, add_placeholder_action=False):
    """
    route_reps: a list of RouteReps representing candidate routes,
        or a batch_size list of lists of RouteReps
    remaining_budgets: batch_size tensor
    chosen_routes: a tensor of the indices of the candidate chosen routes
    mask: if provided, a n_routes or batch_size x n_routes boolean tensor 
        that is True for routes that aren't valid choices
    add_placeholder_action: if True, returned tensor will have an extra element
        at the end with value False
    """

    batch_size = remaining_budgets.shape[0]
    if type(route_reps[0]) is RouteRep:
        route_reps = [route_reps] * batch_size
    route_costs = torch.tensor([[rr.cost for rr in rrs] for rrs in route_reps], 
                                device=remaining_budgets.device)
    # route_costs = route_costs[None].tile(batch_size, 1)

    is_valid_action = route_costs <= remaining_budgets[:, None]
    if is_valid_action.shape[0] == 1 and len(chosen_routes) > 1:
        is_valid_action = is_valid_action.tile(len(chosen_routes), 1)

    for ep_idx, chosen_route_idxs in enumerate(chosen_routes):
        is_valid_action[ep_idx, chosen_route_idxs] = False
    if mask is not None:
        if mask.shape == is_valid_action.shape:
            is_valid_action[mask] = False
        elif len(mask.shape) == 1:
            is_valid_action[:, mask] = False
        else:
            raise ValueError(
                f"mask shape {mask.shape} is not compatible with"\
                f"is_valid_action shape {is_valid_action.shape}")

    if add_placeholder_action:
        # add an extra placeholder column, and set it to False by default
        tmp = is_valid_action
        new_shape = (tmp.shape[0], tmp.shape[1] + 1)
        is_valid_action = torch.zeros(new_shape, dtype=bool, device=tmp.device)
        is_valid_action[:, :-1] = tmp

    return is_valid_action


def get_mlp(n_layers, embed_dim, nonlin_type=DEFAULT_NONLIN, 
            dropout=MLP_DEFAULT_DROPOUT, in_dim=None, out_dim=None, bias=True):
    """n_layers is the number of linear layers, so the number of 'hidden 
    layers' is n_layers - 1."""
    layers = []
    for li in range(n_layers):
        if li == 0 and in_dim is not None:
            layer_in_dim = in_dim
        else:
            layer_in_dim = embed_dim
        if li == n_layers - 1 and out_dim is not None:
            layer_out_dim = out_dim
        else:
            layer_out_dim = embed_dim
        layers.append(nn.Linear(layer_in_dim, layer_out_dim, bias=bias))
        if li < n_layers - 1:
            layers.append(nn.Dropout(dropout))
            layers.append(nonlin_type())
    return nn.Sequential(*layers)


def find_best_routes(nodepair_scores, edge_lens, n_beams=10):
    """
    Beam search!

    nodepair_scores: a batch_size x n_nodes x n_nodes tensor representing the
        "quality" of connecting node i to node j.  Range is (0, 1).
    edge_lens: a batch_size x n_nodes x n_nodes tensor representing the cost to
        traverse directly from node i to node j (inf if there is no such edge)
    """
    has_batch_dim = edge_lens.ndim == 3
    if not has_batch_dim:
        # add a batch dimension
        edge_lens = edge_lens[None]
        nodepair_scores = nodepair_scores[None]
    batch_size = edge_lens.shape[0]
    num_nodes = edge_lens.shape[1]

    dev = edge_lens.device
    has_edge = edge_lens.isfinite() & (edge_lens > 0)
    
    # +1 on the last dim because having a dummy value is convenient below
    batch_routes = torch.full((batch_size, num_nodes, num_nodes, num_nodes+1), 
                              -1, dtype=int, device=dev)
    route_scores = torch.full_like(nodepair_scores, TORCH_FMIN)
    beam_idxs = torch.arange(n_beams, device=dev)
    batch_idxs = torch.arange(batch_size, device=dev)

    for ii in range(num_nodes):
        for jj in range(num_nodes):
            if ii == jj:
                continue

            is_on_route = torch.zeros((batch_size, 1, num_nodes), 
                                      device=dev, dtype=bool)
            is_on_route[..., ii] = True
            routes = torch.full((batch_size, 1, num_nodes + 1), -1, device=dev)
            routes[..., 0] = ii
            cur_nodes = routes[..., 0]
            scores = torch.zeros((batch_size, 1), device=dev)
            is_beam_valid = torch.ones((batch_size, 1), device=dev, dtype=bool)
            while True:
                # compute score of adjacent nodes
                is_valid_next = has_edge[batch_idxs[:, None], cur_nodes]
                is_valid_next &= ~is_on_route
                has_reached_goal = is_on_route[..., jj]
                is_valid_next &= ~has_reached_goal[..., None]
                is_valid_next &= is_beam_valid[..., None]
                # cur_dist = all_dists[batch_idxs[:, None], cur_nodes, jj]
                # is_closer = cur_dist[..., None] > all_dists[batch_idxs[:, None], :, jj]
                # is_valid_next &= is_closer
                if not is_valid_next.any():
                    # no more progress to be made, so break.
                    break

                # all to l
                cur_n_beams = routes.shape[1]
                exp_nps = nodepair_scores[:, None].expand(-1, cur_n_beams, -1, -1)
                exp_routes = routes[..., None].expand(-1, -1, -1, num_nodes).clone()
                exp_routes[exp_routes == -1] = 0
                toadj_scores = exp_nps.gather(-2, exp_routes)
                toadj_scores[routes == -1] = 0
                next_scores = scores[..., None] + toadj_scores.sum(dim=-2)

                # l to j
                ltoj_score = nodepair_scores[:, :, jj]
                next_scores = next_scores + ltoj_score[:, None]
                # avoid invalids
                next_scores[~is_valid_next] = TORCH_FMIN

                # pick the top n_beams scores over all beams
                # flatten the next_node and beam dimensions into one
                flat_scores = next_scores.reshape(batch_size, -1)
                kk = min(n_beams, is_valid_next.sum(dim=(-2, -1)).max())
                topn_scores, topn_idxs = \
                    flat_scores.topk(kk, dim=-1, sorted=False)
                
                topn_beams = topn_idxs // num_nodes
                topn_nextnodes = topn_idxs % num_nodes

                # make them the beams
                # first construct the beam routes

                next_routes = routes[batch_idxs[:, None], topn_beams]
                route_lens = (next_routes > -1).sum(dim=-1)
                next_routes.scatter_(-1, route_lens[..., None], 
                                     topn_nextnodes[..., None])
                routes = next_routes
                
                # then the beam is_on_route masks
                next_is_on_route = is_on_route[batch_idxs[:, None], topn_beams]
                next_is_on_route[batch_idxs[:, None], beam_idxs[:kk], 
                                 topn_nextnodes] = True

                is_on_route = next_is_on_route
                cur_nodes = topn_nextnodes
                scores = topn_scores

                # TODO somehow mask out invalid beams
                is_beam_valid = is_valid_next[batch_idxs[:, None], topn_beams, 
                                              topn_nextnodes]
                                
            best_beam = scores.argmax(dim=-1)
            route_scores[..., ii, jj] = scores[batch_idxs, best_beam]
            batch_routes[batch_idxs, ii, jj] = routes[batch_idxs, best_beam]
    
    max_route_len = (batch_routes > -1).sum(dim=-1).max()
    batch_routes = batch_routes[..., :max_route_len]
    return batch_routes, route_scores
            

def get_twostage_planner(env_rep, 
                         embed_dim, 
                         mode, 
                         n_heads=8, 
                         n_encoder_layers=2, 
                         n_route_enc_layers=1, 
                         max_budget=28200,
                         pregenerated_routes=None, 
                         binary_rollouts=False):
    route_edge_in_dim = 2
    if pregenerated_routes:
        # state_encoder = FullGraphStateEncoder(
        #     env_rep.stop_data.num_features, env_rep.demand_data.num_features,
        #     route_edge_in_dim, env_rep.demand_data.edge_attr.shape[1], 
        #     env_rep.basin_weights.shape[1], embed_dim, n_heads, 
        #     n_encoder_layers, max_budget
        # )
        # encoder = \
        #     SimpleEncoder(env_rep.stop_data.num_nodes, embed_dim, 2, 2,
        #                   max_budget)
        encoder = SimplestEncoder(len(pregenerated_routes), embed_dim, 
                                  n_encoder_layers, max_budget)



        # action_encoder = NodeSumActionEncoder(embed_dim)
        # action_encoder = \
        #     NodeSeqActionEncoder(embed_dim, n_heads, n_route_enc_layers)
        # action_encoder = \
        #     GraphActionEncoder(route_edge_in_dim, embed_dim, 
        #                         n_route_enc_layers)

        # if mode == PLCY_MODE:
        #     action_encoder = NodeSumActionEncoder(embed_dim)
        # else:
        #     # action_encoder = \
        #     #     NodeSeqActionEncoder(embed_dim, n_heads, n_route_enc_layers)
        #     action_encoder = \
        #         GraphActionEncoder(route_edge_in_dim, embed_dim, 
        #                            n_route_enc_layers)

        # encoder = Encoder(embed_dim, state_encoder, action_encoder)



        model = NoFreqPlanner(embed_dim, encoder, mode, 
            do_binary_rollout=binary_rollouts)
    else:
        encoder = EnvironmentEncoder(
            env_rep.stop_data.num_features, 
            env_rep.stop_data.edge_attr.shape[-1],
            env_rep.demand_data.num_features,
            env_rep.demand_data.edge_attr.shape[-1],
            env_rep.basin_weights.shape[1],
            embed_dim, n_encoder_layers
        )
        model = \
            RouteGenerator(embed_dim, encoder, n_route_enc_layers)

    return model
