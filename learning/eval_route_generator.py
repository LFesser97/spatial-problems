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
from torch_geometric.data import Batch
from omegaconf import DictConfig, OmegaConf
import hydra

from torch_geometric.loader import DataLoader
from simulation.citygraph_dataset import CityGraphData, \
    get_dataset_from_config, STOP_KEY
import learning.utils as lrnu


def sample_from_model(model, graph_data, n_routes, min_route_len, 
                      max_route_len, cost_obj, n_samples=20, 
                      sample_batch_size=None):
    model.eval()
    if isinstance(graph_data, Batch):
        # flatten and duplicate across samples
        data_list = graph_data.to_data_list()
        flat_sample_inputs = data_list * n_samples
    else:
        flat_sample_inputs = [graph_data] * n_samples 

    if sample_batch_size is None:
        if isinstance(graph_data, Batch):
            sample_batch_size = graph_data.num_graphs
        else:
            sample_batch_size = n_samples

    all_plans = []
    all_costs = []
    cost_weights = cost_obj.get_weights(graph_data[STOP_KEY].x.device)
    for ii in range(0, len(flat_sample_inputs), sample_batch_size):
        chunk = flat_sample_inputs[ii:ii+sample_batch_size]
        batch = Batch.from_data_list(chunk)
        with torch.no_grad():
            plan_out = model(batch, n_routes, min_route_len, max_route_len, 
                             cost_weights, greedy=False)
            batch_costs = cost_obj(plan_out.routes, batch)
        all_plans += plan_out.routes
        all_costs.append(batch_costs)

    all_costs = torch.cat(all_costs, dim=0).reshape(n_samples, -1)
    _, min_indices = all_costs.min(0)
    batch_size = len(flat_sample_inputs) // n_samples
    best_plans = [all_plans[mi * batch_size + ii] \
                  for ii, mi in enumerate(min_indices)]
    return best_plans


def eval_model(model, eval_dataloader, n_routes, min_route_len, max_route_len,
               cost_obj, sum_writer=None, iter_num=0, n_samples=None,
               silent=False, csv=False, sample_batch_size=None, draw=False):
    log.debug("evaluating our model on test set")
    model.eval()
    device = eval_dataloader.dataset[0][STOP_KEY].x.device
    if n_samples is None:
        cost_weights = cost_obj.get_weights(device)
        method_fn = lambda dd, nr, *args, **kwargs: \
            (model(dd, nr, min_route_len, max_route_len, cost_weights, 
                   greedy=True).routes, None)
    else:
        method_fn = lambda dd, nr, minrl, maxrl, cost_obj, *args, **kwargs: \
            (sample_from_model(model, dd, nr, minrl, maxrl, cost_obj, 
                               n_samples=n_samples, 
                               sample_batch_size=sample_batch_size), None)
    dev = next(model.parameters()).device
    mean_val, std_val, stats = \
        lrnu.test_method(method_fn, eval_dataloader, n_routes, min_route_len,
                         max_route_len, cost_obj, sum_writer, device=dev, 
                         silent=silent, iter_num=iter_num, csv=csv, draw=draw)
    return mean_val, std_val, stats


@hydra.main(version_base=None, config_path="../cfg", 
            config_name="eval_model_mumford")
def main(cfg: DictConfig):
    global DEVICE
    assert 'model' in cfg, "Must provide config for model!"
    DEVICE, _, _, cost_fn, model = \
        lrnu.process_standard_experiment_cfg(cfg, 'eval model', 
                                             weights_required=True)

    # load the data
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    draw = cfg.eval.get('draw', False)

    # evaluate the model on the dataset
    n_samples = cfg.get('n_samples', None)
    sbs = cfg.get('sample_batch_size', cfg.batch_size)
    for n_routes in cfg.eval.n_routes:
        eval_model(model, test_dl, n_routes, cfg.eval.min_route_len, 
                   cfg.eval.max_route_len, cost_fn, n_samples=n_samples, 
                   csv=cfg.eval.csv, sample_batch_size=sbs, draw=draw)
        

if __name__ == "__main__":
    main()
