from itertools import permutations
import time

import torch
import networkx as nx
from networkx.generators.random_graphs import fast_gnp_random_graph

from torch_utils import aggregate_dense_conns, aggregate_edge_features, aggregate_node_features, \
    floyd_warshall, get_path_edge_index, reconstruct_all_paths, reconstruct_path
from world.street_network import build_graph

torch.manual_seed(0)
import numpy as np
np.random.seed(0)

def get_random_graph_mat(nn, pp=0.5, directed=True):
    nn = nn.item()
    grph = fast_gnp_random_graph(nn, pp, directed=directed)
    assert nn == grph.number_of_nodes()
        
    has_edge_mat = torch.tensor(nx.to_numpy_array(grph)).to(dtype=bool)
    dist_mat = torch.rand((nn, nn))
    dist_mat[~has_edge_mat] = float('inf')
    dist_mat.fill_diagonal_(0)
    return dist_mat


def test_basic_floyd_warshall():
    # check on some fully-connected graphs with random edge weights
    for nn in torch.randint(11, (100,)) + 1:
        test_dist_mat = get_random_graph_mat(nn)
        check_floyd_warshall_on_edges(test_dist_mat)

    # check on matrices with some infinite values (meaning no edge)
    nn = 5
    test_dist_mat = torch.ones((nn, nn)) * float('inf')
    # a simply cycle graph
    test_dist_mat.fill_diagonal_(0)
    test_dist_mat[0, 1] = 1
    test_dist_mat[1, 2] = 1
    test_dist_mat[2, 3] = 1
    test_dist_mat[3, 4] = 1
    test_dist_mat[4, 0] = 1
    check_floyd_warshall_on_edges(test_dist_mat)


def test_batched_floyd_warshall():
    batch_size = 32
    for nn in torch.randint(11, (20,)) + 1:
        test_dist_mats = []
        for _ in range(batch_size):
            test_dist_mats.append(get_random_graph_mat(nn))
        test_dist_tensor = torch.stack(test_dist_mats, dim=0)
        check_floyd_warshall_on_edges(test_dist_tensor)


def test_reconstruct_all():
    batch_size = 32
    for nn in torch.randint(11, (20,)) + 1:
        test_dist_mats = []
        for _ in range(batch_size):
            test_dist_mats.append(get_random_graph_mat(nn))
        test_dist_tensor = torch.stack(test_dist_mats, dim=0)
        paths_objs, nexts, dists = floyd_warshall(test_dist_tensor, True)
        all_paths, path_lens = reconstruct_all_paths(nexts)
        
        for b_paths_tnsr, b_path_lens, b_path_obj in \
            zip(all_paths, path_lens, paths_objs):
            for src, dst in permutations(range(nn.item()), 2):
                path_tnsr = b_paths_tnsr[src,  dst, :b_path_lens[src, dst]]
                path = b_path_obj.get_path(src, dst)
                if path_tnsr.numel() == 0:
                    assert path is None
                else:
                    assert (path_tnsr == torch.tensor(path)).all()


def test_edge_feature_aggregation():
    # generate some graphs
    batch_sizes = torch.randint(5, (20,)) + 1
    node_counts = torch.randint(11, (20,)) + 1
    for batch_size, nn in zip(batch_sizes, node_counts):
        test_dist_mats = []
        for _ in range(batch_size):
            test_dist_mats.append(get_random_graph_mat(nn))
        test_dist_tensor = torch.stack(test_dist_mats, dim=0)
        # generate some features for the graphs
        feat_dim = torch.randint(5, (1,)) + 1
        feat_tensor = torch.randn(test_dist_tensor.shape + (feat_dim,))
        node_idxs = torch.arange(nn)
        # zero out self-connection features
        feat_tensor[:, node_idxs, node_idxs] = 0

        # compute the shortest paths through the graph
        paths_objs, nexts, _ = floyd_warshall(test_dist_tensor, True)
        batch_routes = {bi: {(ss, ee): po.get_path(ss, ee)
                        for ss in range(nn) for ee in range(nn)
                        if ss != ee} 
                        for bi, po in enumerate(paths_objs)}

        # aggregate the features over those paths
        sum_agg_feats, sum_edge_counts = \
            aggregate_edge_features(nexts, feat_tensor, "sum", True)
        mean_agg_feats, mean_edge_counts = \
            aggregate_edge_features(nexts, feat_tensor, "mean", True)
        agg_seqs, agg_counts = \
            aggregate_edge_features(nexts, feat_tensor, "concat", True)

        # compute the ground truth feature aggregations
        feat_dim = feat_tensor.shape[-1]
        gt_sum_feats = torch.zeros((batch_size, nn, nn, feat_dim))
        try:
            max_n_edges = max([len(rr) for rs in batch_routes.values() 
                                for rr in rs.values() if rr]) - 1
        except ValueError:
            max_n_edges = 0
        gt_agg_seqs = torch.zeros((batch_size, nn, nn, max_n_edges, feat_dim))
        gt_edge_counts = torch.zeros((batch_size, nn, nn), dtype=int)
        for bi, routes in batch_routes.items():
            for (ss, ee), route in routes.items():
                if route is None:
                    continue
                prev_stop = route[0]
                for edge_idx, cur_stop in enumerate(route[1:]):
                    edge_feat = feat_tensor[bi, prev_stop, cur_stop]
                    gt_sum_feats[bi, ss, ee] += edge_feat
                    gt_agg_seqs[bi, ss, ee, edge_idx] = edge_feat
                    prev_stop = cur_stop
                gt_edge_counts[bi, ss, ee] = len(route) - 1

        assert (gt_edge_counts == sum_edge_counts).all()
        assert (gt_edge_counts == mean_edge_counts).all()
        assert (gt_edge_counts == agg_counts).all()
        
        # ignore invalid entries
        gt_sum_feats = gt_sum_feats[gt_edge_counts > 0]
        sum_agg_feats = sum_agg_feats[gt_edge_counts > 0]
        gt_agg_seqs = gt_agg_seqs[gt_edge_counts > 0]
        agg_seqs = agg_seqs[gt_edge_counts > 0]
        mean_agg_feats = mean_agg_feats[gt_edge_counts > 0]
        gt_edge_counts = gt_edge_counts[gt_edge_counts > 0][:, None]
        gt_mean_feats = gt_sum_feats / gt_edge_counts
        # check sums
        assert torch.isclose(gt_sum_feats, sum_agg_feats).all()
        # check means
        assert torch.isclose(gt_mean_feats, mean_agg_feats).all()
        # check aggs
        assert torch.isclose(gt_agg_seqs, agg_seqs).all()

        # do the same for dense edge aggregation
        # and the same for dense aggregation
        seq_tnsr, seq_lens = reconstruct_all_paths(nexts)
        dense_sum_feats = aggregate_dense_conns(seq_tnsr, feat_tensor, 'sum')
        gt_dense_sums = torch.zeros((batch_size, nn, nn, feat_dim))
        gt_dense_means = torch.zeros((batch_size, nn, nn, feat_dim))
        for bi, routes in batch_routes.items():
            for (ss, ee), route in routes.items():
                if route is None:
                    continue
                edge_idx = get_path_edge_index(route, get_dense_edges=True)
                seq_feats = feat_tensor[bi, edge_idx[0], edge_idx[1]]
                gt_dense_sums[bi, ss, ee] = seq_feats.sum(dim=0)
                gt_dense_means[bi, ss, ee] = seq_feats.mean(dim=0)
        assert torch.isclose(dense_sum_feats, gt_dense_sums, atol=1e-6).all()
        
        dense_mean_feats = aggregate_dense_conns(seq_tnsr, feat_tensor, 'mean')
        assert torch.isclose(dense_mean_feats, gt_dense_means, atol=1e-6).all()


def test_node_feature_aggregation():
    # generate some graphs
    batch_sizes = torch.randint(5, (20,)) + 1
    gt_node_counts = torch.randint(11, (20,)) + 1
    for batch_size, nn in zip(batch_sizes, gt_node_counts):
        test_dist_mats = []
        for _ in range(batch_size):
            test_dist_mats.append(get_random_graph_mat(nn))
        test_dist_tensor = torch.stack(test_dist_mats, dim=0)
        # generate some features for the graphs
        feat_dim = torch.randint(5, (1,)) + 1
        feat_tensor = torch.randn((batch_size, nn, feat_dim))

        # compute the shortest paths through the graph
        paths_objs, nexts, _ = floyd_warshall(test_dist_tensor, True)
        batch_routes = {bi: {(ss, ee): po.get_path(ss, ee)
                        for ss in range(nn) for ee in range(nn)
                        if ss != ee} 
                        for bi, po in enumerate(paths_objs)}

        # aggregate the features over those paths
        sum_agg_feats, sum_node_counts = \
            aggregate_node_features(nexts, feat_tensor, "sum", True)
        mean_agg_feats, mean_node_counts = \
            aggregate_node_features(nexts, feat_tensor, "mean", True)
        agg_seqs, agg_counts = \
            aggregate_node_features(nexts, feat_tensor, "concat", True)

        # compute the ground truth feature aggregations
        feat_dim = feat_tensor.shape[-1]
        gt_sum_feats = torch.zeros((batch_size, nn, nn, feat_dim))
        try:
            max_route_len = max([len(rr) for rs in batch_routes.values() 
                                for rr in rs.values() if rr])
        except ValueError:
            max_route_len = 0
        gt_agg_seqs = torch.zeros((batch_size, nn, nn, max_route_len, feat_dim))
        gt_node_counts = torch.zeros((batch_size, nn, nn), dtype=int)
        for bi, routes in batch_routes.items():
            for (ss, ee), route in routes.items():
                if route is None:
                    continue

                for stop_idx, cur_stop in enumerate(route):
                    node_feat = feat_tensor[bi, cur_stop]
                    gt_sum_feats[bi, ss, ee] += node_feat
                    gt_agg_seqs[bi, ss, ee, stop_idx] = node_feat

                gt_node_counts[bi, ss, ee] = len(route)

        assert (gt_node_counts == sum_node_counts).all()
        assert (gt_node_counts == mean_node_counts).all()
        assert (gt_node_counts == agg_counts).all()        

        if (gt_node_counts == 0).all():
            # there's nothing to compare
            continue
         
        # ignore invalid entries
        gt_sum_feats = gt_sum_feats[gt_node_counts > 0]
        sum_agg_feats = sum_agg_feats[gt_node_counts > 0]
        gt_agg_seqs = gt_agg_seqs[gt_node_counts > 0]
        agg_seqs = agg_seqs[gt_node_counts > 0]
        mean_agg_feats = mean_agg_feats[gt_node_counts > 0]
        gt_node_counts = gt_node_counts[gt_node_counts > 0][:, None]
        gt_mean_feats = gt_sum_feats / gt_node_counts
        # check sums
        assert torch.isclose(gt_sum_feats, sum_agg_feats, atol=1e-6).all()
        # check means
        assert torch.isclose(gt_mean_feats, mean_agg_feats, atol=1e-6).all()
        # check aggs
        assert torch.isclose(gt_agg_seqs, agg_seqs, atol=1e-6).all()


def test_dense_aggregation():
    # generate some sequences
    # generate some features
    # aggregate the features
    # check the results
    pass
    

def check_floyd_warshall_on_edges(edge_tensor):
    paths_objs = floyd_warshall(edge_tensor)

    if edge_tensor.ndim == 2:
        paths_objs = [paths_objs]
        edge_tensor = edge_tensor[None]

    for edge_matrix, paths_obj in zip(edge_tensor, paths_objs):
        nx_edgemat = edge_matrix.numpy()
        nx_edgemat[nx_edgemat == float('inf')] = 0
        graph = nx.from_numpy_array(nx_edgemat, create_using=nx.DiGraph)
        candidate_times = torch.tensor(nx.floyd_warshall_numpy(graph), 
                                       dtype=torch.float32)
        assert torch.isclose(candidate_times, paths_obj.dists).all()

        # check for equality of the candidate routes
        preds, _ = nx.floyd_warshall_predecessor_and_distance(graph)
        gt_candidate_routes = {(ss, ee): nx.reconstruct_path(ss, ee, preds)
                            for ss, ss_preds in preds.items()
                            for ee in ss_preds if ss != ee} 

        nn = edge_matrix.shape[0]
        my_candidate_routes = {(ss, ee): paths_obj.get_path(ss, ee)
                               for ss in range(nn) for ee in range(nn)
                               if ss != ee}
        for terms, my_path in my_candidate_routes.items():
            if terms not in gt_candidate_routes:
                assert my_path is None
            else:
                assert gt_candidate_routes[terms] == my_path


def floyd_warshall_speed_test(network_path):
    # floyd-warshall speed test:
    # convert street graph to an edge tensor
    street_graph = build_graph(network_path)
    print("Number of nodes:", street_graph.number_of_nodes())
    edge_array = nx.to_numpy_array(street_graph, weight="drivetime", 
                                   nonedge=float('inf'))
    edge_tensor = torch.tensor(edge_array)
    # measure the time taken (cpu and gpu)
    start_time = time.perf_counter()
    floyd_warshall(edge_tensor)
    print('CPU time:', time.perf_counter() - start_time)

    edge_tensor = edge_tensor.to(torch.device("cuda"))
    start_time = time.perf_counter()
    floyd_warshall(edge_tensor)
    print('GPU time:', time.perf_counter() - start_time)
