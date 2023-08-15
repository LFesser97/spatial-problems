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
import math
import random
import logging as log
from pathlib import Path
from itertools import product

from tqdm import tqdm
import torch
import networkx as nx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
import optuna
from omegaconf import DictConfig, OmegaConf
import hydra

# need to import CityGraphData here to allow unpickling of new datasets
from simulation.citygraph_dataset import CityGraphData, CityGraphDataset, \
    get_default_train_and_eval_split, get_dynamic_training_set, STOP_KEY, \
    DEMAND_KEY
from torch_utils import get_batch_tensor_from_routes
import learning.utils as lrnu
from learning.models import FeatureNorm, get_mlp
from learning.eval_route_generator import eval_model


BLMODE_NONE = "none"
BLMODE_GREEDY = "greedy"
BLMODE_ROLL = "rolling average"
BLMODE_NN = "neural"


class Baseline:
    def update(self, mean_cost):
        raise NotImplementedError

    def get_baseline(self, graph_data, sumwriter=None, n_eps=None):
        raise NotImplementedError
    

class FixedBaseline:
    def __init__(self, baseline=0, *args, **kwargs):
        self.baseline = baseline

    def update(self, costs):
        self.baseline = costs.mean()

    def get_baseline(self, graph_data, n_routes, sumwriter=None, n_eps=None):
        if sumwriter is not None:
            sumwriter.add_scalar("baseline", self.baseline, n_eps)
        batch_size = graph_data.num_graphs
        return self.baseline[None, None].expand(batch_size, n_routes)
    

class RollingBaseline:
    def __init__(self, alpha):
        assert 0 < alpha <= 1
        self.alpha = alpha
        self.baseline = None

    def update(self, costs):
        mean_cost = costs.mean()
        if self.baseline is None:
            self.baseline = mean_cost
        else:
            self.baseline = self.alpha * mean_cost + (1 - self.alpha) * \
                self.baseline
    
    def get_baseline(self, graph_data, sumwriter=None, n_eps=None):
        if sumwriter is not None:
            if self.baseline is None:
                sumwriter.add_scalar("baseline", 0, n_eps)
            else:
                sumwriter.add_scalar("baseline", self.baseline, n_eps)

        if self.baseline is None:
            return 0
        else:
            return self.baseline


class NNBaseline:
    def __init__(self, learning_rate=0.0005, decay=0.01):
        self.learning_rate = learning_rate
        self.decay = decay
        self.optim = None
        self.model = None
        self._curr_estimate = None
        self.loss_fn = torch.nn.MSELoss()
        self.model = torch.nn.Sequential(
            FeatureNorm(0, self.input_dim),
            get_mlp(3, self.input_dim*2, in_dim=self.input_dim, out_dim=1, 
                    dropout=0.0)
        ).to(DEVICE)
        self.optim = torch.optim.Adam(self.model.parameters(),
                                        lr=self.learning_rate, 
                                        weight_decay=self.decay)

    @property
    def input_dim(self):
        # total demand, num demand edges, mean and std edge demand,
         # mean and std edge time weighted by demand on edge, 
         # mean stop features, # routes so far and # routes left,
         # weights of the two cost components
        return 1 + 1 + 2 + 2 + 8 + 2 + 2 

    def update(self, costs):
        # do backprop
        self.optim.zero_grad()
        costs = costs.to(self._curr_estimate.dtype)
        loss = self.loss_fn(self._curr_estimate, costs)
        loss.backward()
        self.optim.step()
        self._curr_estimate = None

    def inputs_from_data(self, graph_data, cost_weights):
        dev = graph_data[STOP_KEY].x.device
        input_data = torch.zeros(graph_data.num_graphs, self.input_dim, 
                                 dtype=torch.float, device=dev)
        input_data[:, 0] = graph_data.demand.sum(dim=(1,2))

        for bi, dl in enumerate(graph_data.to_data_list()):
            dmd_graph = dl[DEMAND_KEY]
            input_data[bi, 1] = dmd_graph.num_edges
            input_data[bi, 2] = dmd_graph.edge_attr[:, 0].mean()
            if dmd_graph.num_edges > 1:
                input_data[bi, 3] = dmd_graph.edge_attr[:, 0].std()
            else:
                input_data[bi, 3] = 0

            dmd_weighted_times = dmd_graph.edge_attr[:, 0] * \
                dmd_graph.edge_attr[:, 1]
            input_data[bi, 4] = dmd_weighted_times.mean()
            input_data[bi, 5] = dmd_weighted_times.mean()
            x_dim = dl[STOP_KEY].x.shape[1]
            input_data[bi, 6:6+x_dim] = dl[STOP_KEY].x.mean(dim=0)

        for ii, cw in enumerate(cost_weights.values()):
            data_idx = -(1 + ii)
            input_data[:, data_idx] = cw

        return input_data

    def set_input_norm(self, all_graph_data, max_n_routes):
        input_data = \
            self.inputs_from_data(Batch.from_data_list(all_graph_data), {})
        self.model[0].running_mean[...] = input_data.mean(dim=0)
        self.model[0].running_var[...] = input_data.var(dim=0)
        self.model[0].running_mean[-4:-2] = max_n_routes / 2
        self.model[0].running_var[-4:-2] = (max_n_routes / 4)**2
        # assuming here that weights range from 0 to 1
        self.model[0].running_mean[-2:] = 0.5
        self.model[0].running_var[-2:] = 0.25**2
        # self.model[0].running_mean[-1] = 0.075
        # self.model[0].running_var[-1] = 0.0375**2

    def get_baseline(self, graph_data, n_routes, cost_weights, sumwriter=None,
                     n_eps=None):
        # set up the input data
        assert self._curr_estimate is None, "Baseline estimate already exists!"
        
        dev = graph_data[STOP_KEY].x.device

        input_data = self.inputs_from_data(graph_data, cost_weights)
        # add extra features and batch dimension for the numbers of routes
        input_data = input_data[:, None].repeat(1, n_routes, 1)
        routes_so_far = 1 + torch.arange(n_routes, dtype=torch.float, 
                                         device=dev)
        
        input_data[..., -2] = routes_so_far
        input_data[..., -1] = n_routes - routes_so_far
        
        baseline = self.model(input_data).squeeze(-1)
        self._curr_estimate = baseline

        sumwriter.add_scalar("baseline", baseline.mean(), n_eps)
        sumwriter.add_scalar("baseline std", baseline.std(), n_eps)

        assert baseline.isfinite().all()

        return baseline.detach()
    

def eval_model_over_nroutes(model, eval_dl, n_routes_range, min_route_len,
                            max_route_len, cost_obj, sumwriter=None, n_eps=0,
                            n_samples=None, silent=False):
    # evaluate the model over a variety of n_routes, and take the mean
    eval_costs = []
    all_mean_stats = []
    for n_routes in n_routes_range:
        mean_cost, _, stats = \
            eval_model(model, eval_dl, n_routes, min_route_len, max_route_len,
                       cost_obj, silent=silent, n_samples=n_samples)
        eval_costs.append(mean_cost)
        all_mean_stats.append(stats.mean(dim=1))

    mean_eval_cost = sum(eval_costs) / len(eval_costs)
    if sumwriter is not None:
        mean_stats = torch.stack(all_mean_stats, dim=0).mean(dim=0)
        sumwriter.add_scalar("val cost", mean_eval_cost, n_eps)        
        sumwriter.add_scalar("val mean demand time", mean_stats[0], n_eps)
        sumwriter.add_scalar("val total route time", mean_stats[1], 
                             n_eps)
        sumwriter.add_scalar("val 3+transfer and unsatisfied demand", 
                             mean_stats[5], n_eps)
        sumwriter.add_scalar("val uncovered demands", mean_stats[6], n_eps)

    return mean_eval_cost


def render_scenario(graph, routes, show_demand=False):
    # draw the routes on top of the city
    num_colours = max(2, int(np.ceil((2 * len(routes)) ** (1/3))))
    dim_points = np.linspace(0.1, 0.9, num_colours)
    # filter out grayscale colours
    colours = [cc for cc in product(dim_points, dim_points, dim_points)
                if len(np.unique(cc)) > 1]
    colours = iter(colours)

    nx_graph = nx.DiGraph()
    for ii in range(graph.num_nodes):
        nx_graph.add_node(ii)
    posdict = {ii: pos.cpu().numpy() 
               for ii, pos in enumerate(graph[STOP_KEY].pos)}
    nx.draw_networkx_nodes(nx_graph, pos=posdict, node_size=10)
    widths = torch.linspace(10, 2, len(routes))
    for width, colour, route in zip(widths, colours, routes):
        route_graph = nx.DiGraph()
        if type(route) is torch.Tensor:
            route = [ss.item() for ss in route]
        for jj in range(len(route) - 1):
            route_graph.add_edge(route[jj], route[jj+1])
        nx.draw_networkx_edges(route_graph, pos=posdict, width=width, 
                               edge_color=colour)
    
    if show_demand:
        dmd_graph = nx.DiGraph()
        for ii, jj in graph[DEMAND_KEY].edge_index.T:
            dmd_graph.add_edge(ii.item(), jj.item())
        demands = graph.demand[graph[DEMAND_KEY].edge_index[0], 
                               graph[DEMAND_KEY].edge_index[1]]
        widths = demands.cpu().numpy() * 5                               
        nx.draw_networkx_edges(dmd_graph, pos=posdict, width=widths,
                               edge_color='red', style="dotted")

    plt.show()


class DummySummaryWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def add_text(self, *args, **kwargs):
        pass


def flat_discounting(rewards, discount_rate):
    """Calculate discounted rewards.
    
    Args:
    rewards -- a list of rewards
    discount_rate -- the discount rate to use
    """
    discounted_rewards = rewards * discount_rate
    discounted_sums = discounted_rewards[:, 1:].flip([-1]).cumsum(-1).flip([-1])
    returns = rewards.clone()
    returns[:, :-1] += discounted_sums
    return returns


def train(model, min_n_routes, max_n_routes, min_route_len, max_route_len,
          n_epochs, optimizer, train_dataloader, val_dataloader, cost_obj, 
          eval_fn, baseline_mode=BLMODE_NN, discount_rate=None, sumwriter=None, 
          optuna_trial=None, bl_alpha=0.1, return_scale=1, device=None,
          checkpoint_rate=1, checkpoint_dir=None):
    """Train a model on a dataset.
    
    Args:
    eval_model -- a function that takes:
        - a model, 
        - a dataloader, 
        - a summary writer,
        - and the number of episodes so far
        as input, and returns a scalar cost.
    """
    if sumwriter is None:
        # instantiate a dummy summary writer so that the calls to it still work
        sumwriter = DummySummaryWriter()

    best_model = copy.deepcopy(model)
    best_cost = float('inf')
    try:
        graphs_per_epoch = len(train_dataloader.dataset)
    except TypeError:
        graphs_per_epoch = 9 * len(val_dataloader.dataset)
    batches_per_epoch = \
        math.ceil(graphs_per_epoch / train_dataloader.batch_size)
    pbar = tqdm(total=graphs_per_epoch * n_epochs)
    n_eps = 0

    bl_module = None
    if baseline_mode == BLMODE_NONE:
        baseline = 0
    elif baseline_mode == BLMODE_ROLL:
        bl_module = RollingBaseline(bl_alpha)
    elif baseline_mode == BLMODE_NN:
        bl_module = NNBaseline()
        bl_module.set_input_norm(train_dataloader.dataset, max_n_routes)
    elif baseline_mode == BLMODE_GREEDY:
        bl_module = FixedBaseline()
    
    if device is None:
        device = DEVICE

    log.info(f"discount rate is {discount_rate}")

    n_routes = None
    for epoch in range(n_epochs):
        avg_cost = eval_fn(model, val_dataloader, sumwriter, n_eps)
        if optuna_trial is not None:
            optuna_trial.report(avg_cost, epoch)
            if epoch > 0 and optuna_trial.should_prune():
                # don't prune just because of a bad initialization
                raise optuna.exceptions.TrialPruned()

        # Kool et al. does this with a paired t-test, we're keeping it simple
         # for the moment.
        if avg_cost < best_cost:
            # update the best model and the baseline
            best_model = copy.deepcopy(model)
            best_cost = avg_cost
            if baseline_mode == BLMODE_GREEDY:
                bl_module.update(-avg_cost * return_scale)

        model.train()
        dl_iter = iter(train_dataloader)
        for _ in range(batches_per_epoch):
            if n_routes is None:
                # use the max n routes first, so that if we're going to run out
                 # of memory, we do so right away
                n_routes = max_n_routes
            else:
                # pick a random number of routes in the evaluation range
                n_routes = random.randint(min_n_routes, max_n_routes)
            data = next(dl_iter)
            # sample cost weights
            cost_weights = cost_obj.sample_variable_weights(data.num_graphs,
                                                            device)
            assert data[STOP_KEY].x.shape[1] == 8
            if device.type != 'cpu':
                data = data.cuda()
            # run those graphs though the net
            plan_out = model(data, n_routes, min_route_len, max_route_len, 
                             cost_weights)

            # simulate proposed routes and compute rewards
            costs = []
            # compute cost with 0 routes, 1 routes, 2, up to all n routes.
            if discount_rate is not None:
                routes = get_batch_tensor_from_routes(plan_out.routes)
                for ii in range(n_routes + 1):
                    routes_upto = routes[:, :ii]
                    costs_upto = cost_obj(routes_upto, data, **cost_weights)
                    costs.append(costs_upto)
                costs = torch.stack(costs, dim=1)
                rewards = -torch.diff(costs, dim=1) * return_scale
                returns = flat_discounting(rewards, discount_rate)
                # returns = lrnu.rewards_to_returns(rewards, discount_rate)
            else:
                costs = cost_obj(plan_out.routes, data, **cost_weights)
                costs = costs[:, None].expand(-1, n_routes)
                returns = -costs * return_scale

            route_lens = [len(rr) for br in plan_out.routes for rr in br]
            sumwriter.add_scalar("avg # stops on a route", 
                                 np.mean(route_lens), n_eps)

            # costs = cost_fn(plan_out.routes, data)
            sumwriter.add_scalar("avg train cost", costs[:, -1].mean(), n_eps)
            # returns = -costs * return_scale
            
            # Log some stats with the summary writer
            avg_rtrn = returns.mean().item()
            sumwriter.add_scalar("avg return", avg_rtrn, n_eps)

            # Log some stats with the summary writer
            if bl_module is not None:
                baseline = bl_module.get_baseline(data, n_routes, cost_weights, 
                                                  sumwriter, n_eps)
                if baseline_mode != BLMODE_GREEDY:
                    bl_module.update(returns)
                returns -= baseline

                sumwriter.add_scalar("avg return vs baseline", returns.mean(),
                                      n_eps)
                sumwriter.add_scalar("std return vs baseline", returns.std(), 
                                      n_eps)

            if epoch == 0 and type(bl_module) is NNBaseline:
                # don't update the policy, we're letting the baseline catch up
                del plan_out
            else:
                objective = 0
                if plan_out.stop_logits is not None:
                    stop_signal = returns[..., None]
                    objective += (stop_signal * plan_out.stop_logits).mean()
                    sumwriter.add_scalar("stop obj", objective.item(), n_eps)
                if plan_out.route_logits is not None:
                    route_signal = returns
                    route_obj = (route_signal * plan_out.route_logits).mean()
                    sumwriter.add_scalar("route obj", route_obj.item(), n_eps)
                    objective += route_obj
                sumwriter.add_scalar("objective", objective.item(), n_eps)

                # backprop and weight update
                optimizer.zero_grad()
                objective.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            

            pbar.update(data.num_graphs)
            n_eps += data.num_graphs

        if epoch == 0:
            model.update_and_freeze_feature_norms()
        elif checkpoint_dir is not None and epoch % checkpoint_rate == 0:
            ckpt_filename = checkpoint_dir / f"epoch{epoch}.pt"
            torch.save(model.state_dict(), ckpt_filename)

    pbar.close()

    # final eval
    avg_cost = eval_fn(model, val_dataloader, sumwriter, n_eps)
    if avg_cost < best_cost:
        # update the best model and the baseline
        best_model = copy.deepcopy(model)

    return best_model, best_cost


def setup_and_train(cfg: DictConfig, trial: optuna.trial.Trial = None):
    global DEVICE
    DEVICE, run_name, sum_writer, cost_obj, model = \
        lrnu.process_standard_experiment_cfg(cfg, 'inductive_')

    optimizer_type = getattr(torch.optim, cfg.optimizer)
    optimizer = optimizer_type(model.parameters(), lr=cfg.lr, 
                               weight_decay=cfg.decay, 
                               maximize=True)
    
    if cfg.dataset.type == 'pickle':
        train_ds, val_ds = get_default_train_and_eval_split(
            **cfg.dataset.kwargs)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, 
                              shuffle=True)
    elif cfg.dataset.type == 'dynamic':
        train_ds = get_dynamic_training_set(**cfg.dataset.kwargs)
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size)
        val_ds = CityGraphDataset(cfg.dataset.val_path)
    elif cfg.dataset.type == 'mumford':
        data = CityGraphData.from_mumford_data(cfg.dataset.path, 
                                               cfg.dataset.city)
        train_ds = [data] * cfg.batch_size
        train_dl = DataLoader(train_ds, batch_size=cfg.batch_size)
        val_ds = [data]

    val_batch_size = cfg.batch_size * 4
    val_dl = DataLoader(val_ds, batch_size=val_batch_size)

    # run training
    model.to(DEVICE)
    # set a range of random cost parameters to use during validation
    val_cost_obj = copy.deepcopy(cost_obj)
    val_weights = val_cost_obj.sample_variable_weights(val_batch_size, DEVICE)
    val_cost_obj.set_weights(**val_weights)
    
    eval_n_routes = cfg.eval_n_routes
    if type(eval_n_routes) is int:
        eval_n_routes = [eval_n_routes]
        
    # define evaluation function
    eval_fn = lambda mm, dd, sw, ne: \
        eval_model_over_nroutes(mm, dd, eval_n_routes, cfg.eval.min_route_len, 
                                cfg.eval.max_route_len, val_cost_obj, sw, 
                                ne, n_samples=1, silent=True)

    if cfg.baseline_mode == BLMODE_ROLL:
        bl_alpha = cfg.bl_alpha
    else:
        bl_alpha = None

    if 'outdir' in cfg:
        output_dir = Path(cfg.outdir)
    else:
        output_dir = Path('output')

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    min_n_routes = min(eval_n_routes)
    max_n_routes = max(eval_n_routes)
    checkpoint_dir = output_dir / f'{run_name}_checkpoints'
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
    checkpoint_rate = cfg.get('checkpoint_rate', 1)
    best_model, best_cost = \
        train(model, min_n_routes, max_n_routes, cfg.eval.min_route_len, 
              cfg.eval.max_route_len, cfg.n_epochs, optimizer, train_dl, val_dl, 
              cost_obj, eval_fn, cfg.baseline_mode, sumwriter=sum_writer, 
              discount_rate=cfg.discount_rate, bl_alpha=bl_alpha, 
              return_scale=cfg.reward_scale, device=DEVICE, optuna_trial=trial,
              checkpoint_dir=checkpoint_dir, checkpoint_rate=checkpoint_rate)

    # save the new trained model
    torch.save(best_model.state_dict(), output_dir / (run_name + '.pt'))
    
    return best_cost

@hydra.main(version_base=None, config_path="../cfg", config_name="train")
def main(cfg: DictConfig):
    return setup_and_train(cfg)

if __name__ == "__main__":
    main()