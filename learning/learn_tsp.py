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
import torch
import numpy as np
from scipy import stats, spatial
from tqdm import tqdm

DEVICE=torch.device("cuda")

from learning.models import SingleRouteLearner, KoolNextNodeScorer, \
    KoolGraphEncoder, SimplifiedGraphConv


class TspLearner(SingleRouteLearner):
    def plan_route(self, node_descs, stochastic_choices):
        # here's where things get slow!
        solution = []
        probs = []
        step_context_node = self.route_init_placeholder

        while len(solution) < len(node_descs):
            soln_mask = torch.zeros(node_descs.shape[0], dtype=torch.bool,
                requires_grad=False, device=node_descs.device)
            soln_mask[solution] = True

            node_probs = self.node_scorer(step_context_node, node_descs, soln_mask)
            if stochastic_choices:
                next_node_idx = torch.multinomial(node_probs, 1)
            else:
                next_node_idx = torch.argmax(node_probs)

            next_node_idx = next_node_idx.item()
            solution.append(next_node_idx)
            probs.append(node_probs[next_node_idx])
            # soln_mask[next_node_idx] = True

            step_context_node = torch.cat((node_descs[solution[0]],
                                           node_descs[solution[-1]]))
        return solution, torch.stack(probs)

    def plan(self, stochastic_choices):
        routes, probs = super().plan(stochastic_choices)
        return routes[0], probs


def generate_tsp(num_nodes, dist_edge_threshold=0.1, binary_edges=False):
    # range from 0 to 1
    cities = np.random.random((num_nodes, 2))
    dist_matrix = spatial.distance_matrix(cities, cities)
    # use a gaussian-style kernel
    adjacency_matrix = np.exp(- dist_matrix**2)
    # apply the threshold
    adjacency_matrix[dist_matrix > dist_edge_threshold] = 0
    # no self-connections
    np.fill_diagonal(adjacency_matrix, 0)
    if binary_edges:
        adjacency_matrix[adjacency_matrix > 0] = 1

    return cities, adjacency_matrix


def tsp_solution_cost(cities, solution):
    # assume we start at the chosen first city
    cost = 0
    if len(solution) != len(cities):
        msg = "TSP solution had length {} but there are {} cities!"
        raise ValueError(msg.format(len(solution), len(cities)))

    ciprev = solution[0]
    for ci in solution[1:]:
        cost += np.linalg.norm(cities[ci] - cities[ciprev])
        ciprev = ci
    return cost


def reinforce_with_baseline(model, num_epochs, num_steps, batch_size, tsp_size, 
                            learning_rate, bl_update_threshold=0.1):
    torch.autograd.set_detect_anomaly(True)

    model.train()
    best_model = copy.deepcopy(model)
    best_model.eval()
    best_model.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=0.01)
    cities, _ = generate_tsp(tsp_size)
    features = torch.tensor(cities, dtype=torch.float, device=DEVICE)
    # adj_mat = torch.tensor(adj_mat, dtype=torch.float, device=DEVICE)

    for epoch in range(num_epochs):
        epoch_costs = []
        baseline_soln, _ = best_model(False, features)
        bl_cost = tsp_solution_cost(cities, baseline_soln)

        for step in tqdm(range(num_steps)):
            loss = 0
            step_cost = 0
            model.encode_graph(features)

            for _ in tqdm(range(batch_size)):
                sample_soln, probs = model.plan(True)
                sample_cost = tsp_solution_cost(cities, sample_soln)
                epoch_costs.append(sample_cost)
                loss += torch.log(probs).sum() * (sample_cost - bl_cost)
                step_cost += sample_cost

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("avg step cost:", step_cost / batch_size)

        mean_cost = np.mean(epoch_costs)
        cost_stddev = np.std(epoch_costs)
        print("Epoch {} average cost: {:4f} vs bl: {:4f}".format(
            epoch, mean_cost, bl_cost))
        if mean_cost + bl_update_threshold * cost_stddev < bl_cost:
            best_model = copy.deepcopy(model)
            best_model.eval()
            best_model.requires_grad_(False)
            print("updated best model!")
        

if __name__ == "__main__":
    # TODO resume here...build a tsp learner
    # graph_encoder = SimplifiedGraphConv(2, 128, 2)
    graph_encoder = KoolGraphEncoder(2, 128)
    node_scorer = KoolNextNodeScorer(128)
    model = TspLearner(128, graph_encoder, node_scorer)
    model.to(DEVICE)
    # reinforce_with_baseline(model, 100, 2500, 512, 20, 10**-3)
    reinforce_with_baseline(model, 1, 1, 512, 20, 10**-4)
