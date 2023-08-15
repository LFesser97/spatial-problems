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
from itertools import combinations
import logging as log
from pathlib import Path
from collections import defaultdict, namedtuple
import pickle

import torch
import numpy as np
from tqdm import tqdm

import learning.utils as lrnu
from learning.num_scenarios_tracker import TotalInfoTracker
from simulation.timeless_sim import RouteRep, TimelessSimulator, \
    get_bagloee_reward_fn, get_satdemand_reward_fn, get_global_reward_fn, \
    RouteRep
from learning.models import Q_FUNC_MODE, PLCY_MODE, GFLOW_MODE, \
    get_twostage_planner
    
from learning.bagloee import get_newton_routes_from_config, \
    get_required_capacities, bagloee_filtering
from learning.replay_buffer import PrioritizedReplayBuffer
from torch_utils import square_pdist


DEVICE=torch.device("cuda")

REINFORCE_ARG = "reinforce"
QLEARNING_ARG = "q-learning"
GFLOWNET_ARG = "gflownet"
ALG_MODES = {
    REINFORCE_ARG: PLCY_MODE,
    QLEARNING_ARG: Q_FUNC_MODE,
    GFLOWNET_ARG: GFLOW_MODE,
    }

SATDEM_RWD_FN = "per-route satisfied demand"
BAGLOEE_RWD_FN = "bagloee"
QUADRATIC_RWD_FN = "quadratic"
GLOBAL_RWD_FN = "global"

PlanAndSimResult = namedtuple('PlanAndSimResult', 
                              ['routes', 'freqs', 'stop_data', 'route_data',
                               'freq_data'])

ActTypeResult = namedtuple('ActTypeResult', 
                           ['logits', 'actions', 'est_vals', 'returns'])


def compute_average_return(batch_rewards):
    ep_total_returns = [sum([sum(route_rtrns) for route_rtrns in ep_rtrns])
                        for ep_rtrns in batch_rewards]
    avg_stop_rtrns = np.mean(ep_total_returns)
    return avg_stop_rtrns.mean()


class Trainer:
    def __init__(self, model, simulator, budget, quality_metric, num_batches, 
                 eps_per_batch, stop_gamma, route_gamma, freq_gamma, 
                 learning_rate, weight_decay, summary_writer, use_baseline,
                 maxent_factor=0, min_freq=1/7200):
        self.model = model
        self.best_model = None
        self.sim = simulator
        self.budget = budget
        self.quality_metric = quality_metric
        self.num_batches = num_batches
        self.eps_per_batch = eps_per_batch
        self.stop_gamma = stop_gamma
        self.route_gamma = route_gamma
        self.freq_gamma = freq_gamma
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sumwriter = summary_writer
        self.use_baseline = use_baseline
        self.min_freq = min_freq
        self.maxent_factor = maxent_factor

        self.env_rep = self.sim.get_env_rep_for_nn(
            device=DEVICE, one_hot_node_feats=False)

    def log_scalar(self, *args, **kwargs):
        # just a shorthand for the summary writer
        self.sumwriter.add_scalar(*args, **kwargs)

    def setup_optimizer(self):
        params = self.model.parameters()
        optimizer = torch.optim.Adam(params, lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        return optimizer

    def filter_by_budget(self, batch_routes, batch_freqs):
        for routes, frequencies in zip(batch_routes, batch_freqs):
            remaining_budget = self.budget
            for ri, freq in enumerate(frequencies):
                used_budget = self.sim.frequency_to_capacity(freq)
                if used_budget > remaining_budget:
                    frequencies[ri] = 0.
                    # routes[ri] = []
                else:
                    remaining_budget -= used_budget

    def evaluate_model(self, avg_over_n_runs=1, **kwargs):
        with torch.no_grad():
            self.model.eval()
            routes, freqs = self.model(self.env_rep, greedy=True, **kwargs)[:2]
            self.filter_by_budget(routes, freqs)
            # remove batch
            routes = routes[0]
            freqs = freqs[0]

        # there is randomness in the sim, so average results over multiple runs
        infos = defaultdict(list)
        for _ in range(avg_over_n_runs):
            _, info = self.sim.run(routes, freqs)
            for key, value in info.items():
                infos[key].append(value)
        info = {kk: np.mean(vv) for kk, vv in infos.items()}
            
        return routes, freqs, info[self.quality_metric], info

    def update_best(self, routes, freqs, quality, ep_num):
        self.log_scalar('best quality', quality, ep_num)
        return copy.deepcopy(self.model)

    def log_greedy(self, routes, freqs, quality, info, ep_count):
        self.log_scalar('greedy quality', quality, ep_count)
        self.log_scalar('greedy # routes', len(routes), ep_count)
        lrnu.log_stops_per_route(routes, self.sumwriter, ep_count, 'greedy')

        headways = 1 / np.array(freqs)
        if len(headways) > 0:
            mean_headway = np.mean(headways)
            headway_std = np.std(headways)
        else:
            mean_headway = 0
            headway_std = 0
        self.log_scalar('greedy headway avg', mean_headway, ep_count)
        self.log_scalar('greedy headway std dev', headway_std, ep_count)
        for ik, iv in info.items():
            self.log_scalar('greedy ' + ik, iv, ep_count)

    def plan_and_log(self, log_idx, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.eps_per_batch
        plan_out = self.model.plan(self.env_rep, greedy=False, 
                                   batch_size=batch_size, **kwargs)
        batch_routes = plan_out.routes
        
        # log some statistics of the proposed systems and value estimates
        route_lens = [len(rr) for routes in batch_routes for rr in routes]
        avg_stops_per_route = np.mean(route_lens) if len(route_lens) > 0 else 0
        self.log_scalar('# stops per route', avg_stops_per_route, log_idx)
        route_counts = [len(rr) for rr in batch_routes]
        self.log_scalar('# routes', np.mean(route_counts), log_idx)
        if plan_out.route_est_vals is not None:
            self.log_scalar('route baseline avg.', 
                            plan_out.route_est_vals.mean(), log_idx)
            self.log_scalar('route baseline std.', 
                            plan_out.route_est_vals.std(), log_idx)
        return plan_out

    def plan_and_simulate(self, ep_count, **kwargs):
        plan_out = self.plan_and_log(ep_count, **kwargs)
        self.filter_by_budget(plan_out.routes, plan_out.freqs)
        
        # simulate the batch's systems and build the reward tensor
        if plan_out.stop_logits is not None:
            stop_rwd_tnsr = torch.zeros(plan_out.stop_logits.shape, 
                                        device=DEVICE)
        if plan_out.route_logits is not None:
            route_rwd_tnsr = torch.zeros(plan_out.route_logits.shape,
                                        device=DEVICE)
        if plan_out.freq_logits is not None:
            freq_rwd_tnsr = torch.zeros(plan_out.freq_logits.shape,
                                        device=DEVICE)
        
        # assemble the reward tensors
        qualities = []
        glbl_infos = defaultdict(list)
        log.info("simulating...")
        for ep_idx, (routes, freqs) in enumerate(zip(plan_out.routes, 
                                                     plan_out.freqs)):
            stop_info, glbl_info = self.sim.run(routes, freqs)
            for key, value in glbl_info.items():
                glbl_infos[key].append(value)

            stop_rewards = stop_info["rewards"]

            qualities.append(glbl_info[self.quality_metric])
            # insert values in the reward tensors
            for route_choice_idx, route_stop_rewards in \
                enumerate(stop_rewards):

                rsr_tnsr = torch.tensor(route_stop_rewards,
                                        device=DEVICE)
                if plan_out.stop_logits is not None:
                    route_idx = plan_out.routes_tensor[ep_idx, 
                                                       route_choice_idx]
                    stop_rwd_tnsr[ep_idx, route_idx, :len(rsr_tnsr)] += rsr_tnsr
                if plan_out.route_logits is not None:
                    route_rwd_tnsr[ep_idx, route_choice_idx] = rsr_tnsr.sum()
                if plan_out.freq_logits is not None:
                    freq_rwd_tnsr[ep_idx, route_choice_idx] = rsr_tnsr.sum()

        log.info("done simulating")
        self.log_scalar('quality', np.mean(qualities), ep_count)

        glbl_info = {kk: np.mean(vv) for kk, vv in glbl_infos.items()}
        for ik, iv in glbl_info.items():
            self.log_scalar(ik, iv, ep_count)

        # add entropy for maximum-entropy RL
        if plan_out.stop_logits is not None:
            stop_rwd_tnsr -= self.maxent_factor * plan_out.stop_logits.detach()
        if plan_out.route_logits is not None:
            route_rwd_tnsr -= self.maxent_factor * \
                plan_out.route_logits.detach()
        if plan_out.freq_logits is not None:
            freq_rwd_tnsr -= self.maxent_factor * plan_out.freq_logits.detach()
        if self.maxent_factor != 0:
            # stopping at max stops / routes is the only option, so entropy = 0
            stop_rwd_tnsr[:, -1] = 0
            route_rwd_tnsr[:, -1] = 0

        # # clip rewards to keep them in [-2, 2] range, stabilizing training
        # if plan_out.stop_logits is not None:
        #     stop_rwd_tnsr.clamp_min_(-2)
        # if plan_out.route_logits is not None:
        #     route_rwd_tnsr.clamp_min_(-2)
        # if plan_out.freq_logits is not None:
        #     freq_rwd_tnsr.clamp_min_(-2)

        # compute returns from rewards
        stop_result = None
        if plan_out.stop_logits is not None:
            self.log_scalar('stop reward', stop_rwd_tnsr.mean(), 
                            ep_count)
            returns = lrnu.rewards_to_returns(stop_rwd_tnsr, self.stop_gamma)
            stop_result = ActTypeResult(
                logits=plan_out.stop_logits,
                actions=plan_out.stops_tensor,
                est_vals=plan_out.stop_est_vals,
                returns=returns)

        route_result = None
        if plan_out.route_logits is not None:
            self.log_scalar('route reward', route_rwd_tnsr.mean(), 
                            ep_count)
            returns = lrnu.rewards_to_returns(route_rwd_tnsr, self.route_gamma)
            route_result = ActTypeResult(
                logits=plan_out.route_logits,
                actions=plan_out.routes_tensor,
                est_vals=plan_out.route_est_vals,
                returns=returns)

        freq_result = None
        if plan_out.freq_logits is not None:
            # route and frequency rewards are necessarily the same.
            self.log_scalar('freq reward', freq_rwd_tnsr.mean(), 
                            ep_count)
            returns = lrnu.rewards_to_returns(freq_rwd_tnsr, self.freq_gamma)
            freq_result = ActTypeResult(
                logits=plan_out.freq_logits,
                actions=plan_out.freqs_tensor,
                est_vals=plan_out.freq_est_vals,
                returns=returns)

        return PlanAndSimResult(routes=plan_out.routes, freqs=plan_out.freqs,
                                stop_data=stop_result, route_data=route_result,
                                freq_data=freq_result)

    def update_route_info(self, req_cpcties, avgd_over, routes, route_idxs, 
                          per_stop_info, ep_count):
        new_caps = get_required_capacities(self.sim, routes, per_stop_info)
        # cut off placeholder values after the end of the route
        route_idxs = route_idxs[:len(routes)]

        # decaying average w/ global count
        new_caps = torch.tensor(new_caps, device=DEVICE)
        denom = ep_count + 1
        req_cpcties[route_idxs] += new_caps / denom
        req_cpcties[route_idxs] /= 1 + 1 / denom
        avgd_over[route_idxs] += 1

    def eval_fixed_capacity_model(self, route_reps, algorithm, avg_over=1):
        with torch.no_grad():
            costs = torch.tensor([rr.cost for rr in route_reps], device=DEVICE)
            all_freqs = self.sim.capacity_to_frequency(costs)
            mask = all_freqs < self.min_freq
            if mask.all():
                log.warn("All routes are below minimum frequency!")
            # set the right argument to run greedily
            self.model.eval()
            if algorithm == GFLOWNET_ARG:
                plan_out = \
                    self.model.greedy_flow_plan(self.env_rep, route_reps,
                                                budget=self.budget,
                                                mask=mask)
                routes = plan_out.routes
            else:
                if algorithm == REINFORCE_ARG:
                    plan_out = self.model(self.env_rep, route_reps,
                                          self.budget, greedy=True, 
                                          mask=mask)
                elif algorithm == QLEARNING_ARG:
                    plan_out = self.model(self.env_rep, route_reps, 
                                          self.budget, epsilon=0, 
                                          mask=mask)
                routes = plan_out.routes[0]

            route_idxs = plan_out.routes_tensor[0, :len(routes)]
            freqs = [ff.item() for ff in all_freqs[route_idxs]]

            log.debug("eval: simulating")
            infos = defaultdict(list)
            run_returns = []
            for _ in range(avg_over):
                per_stop, info = self.sim.run(routes, freqs)
                for key, value in info.items():
                    infos[key].append(value)
                
                route_rwd_tnsr = torch.zeros(plan_out.routes_tensor.shape,
                                             device=DEVICE)
                for ri, route_stop_rwds in enumerate(per_stop["rewards"]):
                    route_reward = route_stop_rwds.sum()
                    route_rwd_tnsr[0, ri] = route_reward
                run_returns.append(route_rwd_tnsr)
            
            info = {kk: np.mean(vv) for kk, vv in infos.items()}
            returns = lrnu.rewards_to_returns(torch.cat(run_returns), 
                                              self.route_gamma)
            info["average return"] = returns.mean()

        return routes, freqs, info[self.quality_metric], info

    @property
    def num_episodes(self):
        return self.num_batches * self.eps_per_batch

    def train_shortestpath_reinforce(self, eval_per_n_batches=10, n_routes=10,
                                     n_chunks=1):
        log.info("training shortest-paths generator with REINFORCE")
        route_baseline = 0
        stop_baseline = 0
        alpha = 0.9
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay,
                                     maximize=True)

        log.debug("evaluating model")
        model_kwargs = {
            "n_routes": n_routes,
            "n_chunks": n_chunks
        }
        fixed_cpcty = self.sim.frequency_to_capacity(self.model.fixed_freq)
        eff_budget = fixed_cpcty * n_routes
        log.info(f"with {n_routes} routes, effective budget is "\
                 f"{eff_budget:0.1f}")

        routes, freqs, best_quality, info = self.evaluate_model(**model_kwargs)

        self.log_greedy(routes, freqs, best_quality, info, 0)
        best_model = self.update_best(routes, freqs, best_quality, 0)
        best_routes = routes
        best_freqs = freqs

        for ep_count in tqdm(range(0, self.num_episodes, self.eps_per_batch)):
            self.model.train()
            plan_out = self.plan_and_log(ep_count, **model_kwargs)

            stop_rwd_tnsr = torch.zeros(plan_out.stop_logits.shape, 
                                        device=DEVICE)
            route_rwd_tnsr = torch.zeros(plan_out.route_logits.shape,
                                        device=DEVICE)

            glbl_infos = defaultdict(list)
            for ep_idx, (routes, freqs) in enumerate(zip(plan_out.routes, 
                                                         plan_out.freqs)):
                _, glbl_info = self.sim.run(routes, freqs)
                for key, value in glbl_info.items():
                    glbl_infos[key].append(value)
                log.info("computing the learning signal")
                stop_rwd_tnsr[ep_idx] = glbl_info["saved time"]
                route_rwd_tnsr[ep_idx] = glbl_info["saved time"]
            
            glbl_info = {kk: np.mean(vv) for kk, vv in glbl_infos.items()}
            for ik, iv in glbl_info.items():
                self.log_scalar(ik, iv, ep_count)

            # apply maximum entropy
            stop_rwd_tnsr -= self.maxent_factor * plan_out.stop_logits.detach()
            route_rwd_tnsr -= \
                self.maxent_factor * plan_out.route_logits.detach()
            # apply baseline
            route_signal = route_rwd_tnsr - route_baseline
            stop_signal = stop_rwd_tnsr - stop_baseline

            # compute signal
            batch_signal = (stop_signal * plan_out.stop_logits).sum()
            batch_signal += (route_signal * plan_out.route_logits).sum()
            avg_obj = batch_signal.item() / self.eps_per_batch
            self.log_scalar('objective', avg_obj, ep_count)

            log.info("backprop and weight update")
            optimizer.zero_grad()
            batch_signal.backward()
            optimizer.step()

            # update the baseline
            # this works because every scenario has the same number of routes
             # if that changes, the below will break
            route_baseline = alpha * route_baseline + \
                (1 - alpha) * route_rwd_tnsr.mean()
            avg_stop_rwd = stop_rwd_tnsr[plan_out.stops_tensor >= 0].mean()
            stop_baseline = alpha * stop_baseline + (1 - alpha) * avg_stop_rwd

            if ep_count == self.num_episodes - 1 or \
               (ep_count // self.eps_per_batch > 0 and
                ep_count % (eval_per_n_batches * self.eps_per_batch) == 0):
                log.info("estimate greedy performance")
                # compute the greedy performance of the new model
                routes, freqs, quality, info = \
                    self.evaluate_model(**model_kwargs)
                self.log_greedy(routes, freqs, quality, info, ep_count)

                if quality > best_quality:
                    best_model = self.update_best(routes, freqs, quality, 
                                                  ep_count)
                    best_quality = quality
                    best_routes = routes
                    best_freqs = freqs
        
        image = self.sim.get_network_image(best_routes, best_freqs)
        self.sumwriter.add_image('City with routes', image, ep_count)
        return best_model, best_routes, best_freqs


    def train_reinforce(self, eval_per_n_batches=10):
        route_baseline = 0
        alpha = 0.9

        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        env_representation = self.sim.get_env_rep_for_nn(device=DEVICE)

        routes, freqs, best_quality, info = \
            self.evaluate_model(env_representation, budget=budget)

        self.log_greedy(routes, freqs, best_quality, info, 0)
        best_model = self.update_best(routes, freqs, best_quality, 0)
        best_routes = routes
        best_freqs = freqs
        val_loss_fn = torch.nn.MSELoss()
        # make the budget in terms of frequency
        budget = self.sim.capacity_to_frequency(self.budget)

        for ep_count in tqdm(range(0, self.num_episodes, self.eps_per_batch)):
            log.debug("starting forward pass and sim")
            self.model.train()
            self.model.encode_graph(env_representation)
            result = self.plan_and_simulate(env_representation, ep_count, 
                                            budget=budget)

            # compute the learning signal
            log.debug("computing the learning signal")
            batch_signal = 0
            
            for atr_name in ["stop_data", "route_data", "freq_data"]:
                atr = getattr(result, atr_name)
                if atr is None:
                    continue
                
                actsig = atr.returns
                if atr_name == "route_data":
                    actsig -= route_baseline
                    route_baseline = alpha * route_baseline + \
                        (1 - alpha) * actsig.mean()

                if self.use_baseline and atr.est_vals is not None:
                    actsig -= atr.est_vals.detach()
                    value_loss = val_loss_fn(atr.est_vals, atr.returns)
                    batch_signal += value_loss
                    self.log_scalar(atr_name + " value loss", value_loss, 
                                    ep_count)

                batch_signal -= (atr.logits * actsig).sum()
                
            avg_obj = batch_signal.item() / self.eps_per_batch
            self.log_scalar('objective', avg_obj, ep_count)

            # update weights
            log.debug("backprop and weight update")
            optimizer.zero_grad()
            batch_signal.backward()
            optimizer.step()

            if ep_count == self.num_episodes - 1 or \
               (ep_count // self.eps_per_batch > 0 and
                ep_count % (eval_per_n_batches * self.eps_per_batch) == 0):
                log.info("estimate greedy performance")
                # compute the greedy performance of the new model
                routes, freqs, quality, info = \
                    self.evaluate_model(env_representation, budget=budget)
                self.log_greedy(routes, freqs, quality, info, ep_count)

                if quality > best_quality:
                    best_model = self.update_best(routes, freqs, quality, 
                                                  ep_count)
                    best_quality = quality
                    best_routes = routes
                    best_freqs = freqs    

        # compute the greedy performance of the final model
        log.info("estimate greedy performance")
        routes, freqs, quality, info = \
            self.evaluate_model(env_representation, budget=budget)
        self.log_greedy(routes, freqs, quality, info, ep_count)

        if quality > best_quality:
            best_model = self.update_best(routes, freqs, quality, 
                                          ep_count)
            best_quality = quality
            best_routes = routes
            best_freqs = freqs    

        image = self.sim.get_network_image(best_routes, best_freqs)
        self.sumwriter.add_image('City with routes', image, ep_count)
        return best_model, best_routes, best_freqs

    def train_gflownet_wo_freqs(self, candidate_routes, buffer_size, 
                                tau=0, epsilon=0.05, leaf_coef=10, 
                                flow_eps=1e-6, init_capacity=272, 
                                eval_per_n_eps=20, minibatch_size=32,
                                req_cpcty_chkpt_path=None):
        log.info("training GFlowNet")
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        n_cdts = len(candidate_routes)

        phase, req_cpcties, count_offset = \
            set_up_req_capacities(req_cpcty_chkpt_path, n_cdts, init_capacity)
        req_cpcties, candidate_routes = \
            zip(*sorted(zip(req_cpcties, candidate_routes), reverse=True))
        req_cpcties = torch.stack(req_cpcties)
        mask = self.sim.capacity_to_frequency(req_cpcties) < self.min_freq
        route_reps = \
            self.sim.get_route_reps_for_nn(candidate_routes, req_cpcties,
                                           device=DEVICE)

        avgd_over = torch.ones(n_cdts, device=DEVICE, dtype=int)

        # initialize replay buffer
        replay_buffer = []
        best_routes, best_freqs, best_quality, info = \
            self.eval_fixed_capacity_model(route_reps, GFLOWNET_ARG)
        self.log_greedy(best_routes, best_freqs, best_quality, info, 0)
        best_model = self.update_best(best_routes, best_freqs, best_quality, 0)

        if tau > 0:
            tgt_model = copy.deepcopy(self.model)
        else:
            # we're just using the learning model for targets
            tgt_model = self.model


        for ep_count in tqdm(range(0, self.num_episodes, self.eps_per_batch)):
            # generate an episode
            self.model.train()
            all_freqs = self.sim.capacity_to_frequency(req_cpcties)
            old_req_cpcties = req_cpcties.clone()
            mask = all_freqs < self.min_freq
            if phase == 0:
                budget = self.budget * \
                    torch.max(torch.tensor(0.1), torch.rand(1) ** 2)
            else:
                budget = torch.tensor([float(self.budget)])
            budget = budget.to(device=DEVICE)

            log.debug("rolling out an episode")
            with torch.no_grad():
                plan_out = self.plan_and_log(
                    ep_count, route_reps=route_reps, budget=budget, 
                    mask=mask, epsilon=epsilon)

            all_rewards = []
            glbl_infos = defaultdict(list)
            for ep_idx, ep_routes in enumerate(plan_out.routes):
                ep_route_idxs = plan_out.routes_tensor[ep_idx, :len(ep_routes)]
                ep_freqs = all_freqs[ep_route_idxs]
                log.debug("simulating")
                per_stop, glbl = \
                    self.sim.run(ep_routes, ep_freqs, capacity_free=(phase==0))
                # self.update_route_info(req_cpcties, avgd_over, ep_routes, 
                #                        ep_route_idxs, per_stop, 
                #                        ep_count + count_offset)
                # accumulate global info
                for key, value in glbl.items():
                    glbl_infos[key].append(value)

                # accumulate statistics to log later
                if len(budget) == 1:
                    ep_budget = budget
                else:
                    ep_budget = budget[ep_idx]
                if len(ep_routes) == 0:
                    budget_used = 0
                else:
                    budget_used = req_cpcties[ep_route_idxs].sum().item()
                pcnt_budget_used = budget_used * 100 / ep_budget
                assert pcnt_budget_used <= 100.01
                glbl_infos["budget used (%)"].append(pcnt_budget_used.item())

                # Reward appears only at the end.
                stop_rewards = per_stop["rewards"]
                total_reward = sum([rsr.sum() for rsr in stop_rewards])
                all_rewards.append(total_reward)

                if len(replay_buffer) < buffer_size or \
                   total_reward > replay_buffer[0][0]:
                    # add the new episode to the replay buffer
                    rb_tpl = (total_reward, ep_route_idxs.cpu(), 
                              ep_budget.cpu())
                    replay_buffer.append(rb_tpl)
                if len(replay_buffer) > buffer_size:
                    # discard bottom 5% of the episodes, as in Bengio 2021
                    tmp = sorted(replay_buffer)
                    replay_buffer = tmp[max(int(0.05 * buffer_size), 1):]

            glbl_info = {kk: np.mean(vv) for kk, vv in glbl_infos.items()}
            for ik, iv in glbl_info.items():
                self.log_scalar(ik, iv, ep_count)

            # compute returns from rewards
            self.log_scalar('route reward', np.mean(all_rewards), ep_count)

            if phase == 0:
                if (avgd_over > 1).all():
                    # we've tried every route at least once, so move on
                    log.info(f"phase 0 is over after {ep_count} eps, "
                             "starting phase 1")
                    phase = 1
                else:
                    # don't do any learning in phase 0
                    continue

            # sample some minibatches from replay buffer and learn on them
            log.debug("training")
        
            # assemble a batch of episodes for training
            b_budgets = []
            b_rewards = []
            b_done = []
            b_states = []
            # learn on approximately four trajectories each batch
            eps_so_far = self.eps_per_batch * (ep_idx + 1)
            steps_per_ep = len(replay_buffer) // eps_so_far
            n_steps_in_batch = 4 * steps_per_ep

            for ii in torch.randint(len(replay_buffer), (n_steps_in_batch,)):
                reward, scenario, budget = replay_buffer[ii]
                for jj in range(len(scenario)):
                    scenario_so_far = scenario[:jj+1]
                    cost_so_far = \
                        sum([route_reps[ii].cost for ii in scenario_so_far])
                    b_budgets.append(budget - cost_so_far)
                    if len(scenario_so_far) == len(scenario):
                        b_rewards.append(reward)
                        b_done.append(True)
                    else:
                        b_rewards.append(0)
                        b_done.append(False)
                    b_states.append(scenario_so_far)

            b_budgets = torch.cat(b_budgets).to(device=DEVICE)
            b_rewards = torch.tensor(b_rewards, device=DEVICE)
            b_done = torch.tensor(b_done, device=DEVICE)

            # estimate inflow along each edge    
            # passing over the whole batch at once is too costly, so split it
             # into minibatches
            loss_for_log = 0
            leaf_loss = 0
            interior_loss = 0
            n_minibatches = int(np.ceil(n_steps_in_batch // minibatch_size))
            for mbi in range(n_minibatches):
                si = mbi * minibatch_size
                ei = si + minibatch_size
                mb_budgets = b_budgets[si:ei]
                mb_rewards = b_rewards[si:ei]
                mb_done = b_done[si:ei]
                mb_state_routes = b_states[si:ei]

                # compute the inflow to each state
                mb_parents_q_est = self.model.estimate_parent_qs(
                    self.env_rep, route_reps, mb_budgets, mb_state_routes)
                mb_in_flow = torch.zeros(len(mb_parents_q_est), device=DEVICE)
                for mi, parents_q_est in enumerate(mb_parents_q_est):
                    mb_in_flow[mi] = parents_q_est.logsumexp(dim=-1)
                
                with torch.no_grad():
                    # estimate outflow from each state
                    _, mb_q_est = tgt_model.step(
                        self.env_rep, route_reps, mb_state_routes, 
                        mb_budgets, mask=mask)
                    mb_q_est[mb_done] = -float('inf')

                mb_log_rwds = torch.log(mb_rewards)
                assert (mb_q_est[mb_rewards > 0] == -float('inf')).all()
                mb_out_flows = torch.stack((mb_q_est, mb_log_rwds), dim=1)
                mb_out_flow = mb_out_flows.logsumexp(dim=-1)
                # compute loss between inflow and outflow for each state
                losses = \
                    ((mb_in_flow + flow_eps) - (mb_out_flow + flow_eps)).pow(2)
                # penalize loss at "leaves" more than loss due to flow, since
                 # that's the "ground truth"
                losses[~mb_done] /= leaf_coef
                # scale loss by its fraction of the batch; this makes it the
                 # same as doing one pass over the batch all at once.
                mb_scale = minibatch_size / n_steps_in_batch
                loss = losses.mean() * mb_scale
                log.debug("backprop")
                loss.backward(retain_graph=mbi < n_minibatches - 1)
                # log some statistics of the loss
                loss_for_log += loss.item()
                if mb_done.any():
                    done_scale = mb_done.sum() / len(mb_done)
                    ll = losses[mb_done].mean() * done_scale
                    leaf_loss += ll.item() * mb_scale
                if not mb_done.all():
                    not_done_scale = (~mb_done).sum() / len(mb_done)
                    il = (losses[~mb_done].mean() * not_done_scale) * leaf_coef
                    interior_loss += il.item() * mb_scale

            log.debug("weight update")
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            assert np.isclose(loss_for_log, 
                leaf_loss + interior_loss / leaf_coef)
            self.log_scalar("loss", loss_for_log, ep_count)
            self.log_scalar("leaf loss", leaf_loss, ep_count)
            self.log_scalar("interior loss", interior_loss, ep_count)

            if tau > 0:
                # update target model
                for aa, bb in zip(self.model.parameters(), 
                                  tgt_model.parameters()):
                    bb.data.mul_(1-tau).add_(tau * aa)

            # periodically evaluate the model in greedy mode
            if ep_count > 0 and ep_count % eval_per_n_eps == 0:
                log.debug("estimating greedy performance")
                # compute the greedy performance of the new model
                routes, freqs, quality, info = \
                    self.eval_fixed_capacity_model(route_reps, GFLOWNET_ARG)

                self.log_greedy(routes, freqs, quality, info, ep_count)

                if quality > best_quality:
                    best_model = self.update_best(routes, freqs, quality, 
                                                  ep_count)
                    best_quality = quality
                    best_routes = routes
                    best_freqs = freqs

        # compute the greedy performance of the final model
        log.info("estimate greedy performance")
        routes, freqs, quality, info = \
            self.eval_fixed_capacity_model(route_reps, GFLOWNET_ARG)

        self.log_greedy(routes, freqs, quality, info, ep_count)
        if quality > best_quality:
            best_model = self.update_best(routes, freqs, quality, ep_count)
            best_quality = quality
            best_routes = routes
            best_freqs = freqs    

        # return best values
        image = self.sim.get_network_image(best_routes, best_freqs)
        self.sumwriter.add_image('City with routes', image, ep_count)
        return best_model, best_routes, best_freqs


    def train_q_network_wo_freqs(self, candidate_routes, alpha, beta,
                                 minibatch_size, buffer_size, tau=0,
                                 init_capacity=272, eval_per_n_eps=1,
                                 binary_rollouts=False,
                                 req_cpcty_chkpt_path=None):
        log.info("training Q network")
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        n_cdts = len(candidate_routes)

        phase, req_cpcties, count_offset = \
            set_up_req_capacities(req_cpcty_chkpt_path, n_cdts, init_capacity)
        req_cpcties, candidate_routes = \
            zip(*sorted(zip(req_cpcties, candidate_routes), reverse=True))
        req_cpcties = torch.stack(req_cpcties)

        avgd_over = torch.ones(n_cdts, device=DEVICE, dtype=int)
        # minimum allowed frequency is every two hours
        epsilon = 1.0
        final_epsilon = 0.1
        epsilon_step = (epsilon - final_epsilon) / self.num_batches
        final_beta = 1.0
        beta_step = (final_beta - beta) / self.num_batches
        route_reps = self.sim.get_route_reps_for_nn(candidate_routes, 
                                                    req_cpcties, device=DEVICE)

        # initialize replay buffer
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        # replay_buffer = ReplayBuffer(buffer_size)
        alg = QLEARNING_ARG
        best_routes, best_freqs, best_quality, info = \
            self.eval_fixed_capacity_model(route_reps, alg)
        self.log_greedy(best_routes, best_freqs, best_quality, info, 0)
        best_model = self.update_best(best_routes, best_freqs, best_quality, 0)

        if tau > 0:
            tgt_model = copy.deepcopy(self.model)
        else:
            # we're just using the learning model for targets
            tgt_model = self.model
        
        # the following code assumes batch size is 1, which we enforce 
         # elsewhere
        for batch_count in tqdm(range(self.num_batches)):
            # generate an episode
            all_freqs = self.sim.capacity_to_frequency(req_cpcties)
            old_req_cpcties = req_cpcties.clone()
            mask = all_freqs < self.min_freq
            if phase == 0:
                budget = self.budget * \
                    torch.max(torch.tensor(0.1), torch.rand(1) ** 2)
            else:
                budget = torch.tensor([float(self.budget)])
            budget = budget.to(device=DEVICE)
            log.debug("rolling out an episode")
            self.model.train()
            with torch.no_grad():
                plan_out = self.plan_and_log(
                    batch_count, route_reps=route_reps, batch_size=1, 
                    budget=budget, route_costs=req_cpcties, 
                    mask=mask, epsilon=epsilon)

            # process the generated epsiode
            ep_routes = plan_out.routes[0]
            ep_route_idxs = plan_out.routes_tensor[0]
            ep_freqs = all_freqs[ep_route_idxs]
            log.debug("simulating")
            per_stop, glbl = \
                self.sim.run(ep_routes, ep_freqs, capacity_free=(phase==0))
            if phase == 0:
                self.update_route_info(req_cpcties, avgd_over, ep_routes, 
                                       ep_route_idxs, per_stop, 
                                       batch_count + count_offset)

            # accumulate statistics to log later
            if len(ep_routes) == 0:
                budget_used = 0
            else:
                budget_used = old_req_cpcties[ep_route_idxs].sum().item()
            pcnt_budget_used = budget_used * 100 / budget
            assert pcnt_budget_used <= 100
            glbl["budget used (%)"] = pcnt_budget_used

            # assemble rewards
            route_rwd_tnsr = torch.zeros(plan_out.route_est_vals.shape,
                                         device=DEVICE)
            global_reward = glbl[self.quality_metric] * 25 / 3519400
            route_rwd_tnsr[0, -1] = global_reward
            # stop_rewards = per_stop["rewards"]
            # for ii, route_stop_rewards in enumerate(stop_rewards):
            #     route_reward = route_stop_rewards.sum()
            #     route_rwd_tnsr[0, -1] += route_reward

            for ik, iv in glbl.items():
                self.log_scalar(ik, iv, batch_count)

            # compute returns from rewards
            self.log_scalar('route reward', route_rwd_tnsr.mean().item(), 
                            batch_count)
            self.log_scalar('total route reward', route_rwd_tnsr.sum().item(), 
                            batch_count)

            if phase == 0:
                if (avgd_over > 1).all():
                    # we've tried every route, so move on to phase 1
                    log.info(f"phase 0 is over after {batch_count} eps, "
                             "starting phase 1")
                    phase = 1
                else:
                    # don't do any learning in phase 0
                    continue

            # add all generated steps and their returns to the replay buffer
            log.debug("adding new episode to replay buffer")
            if binary_rollouts:
                ii = 0
                routes_so_far = np.array([], dtype=int)
                remaining_budget = budget.item()

                for ci in range(n_cdts):
                    if ii < len(ep_route_idxs) and ci == ep_route_idxs[ii]:
                        # route is included
                        ii += 1
                        action = 1
                        routes_so_far = ep_route_idxs[:ii].cpu().numpy()
                        remaining_budget -= old_req_cpcties[ci].item()
                    else:
                        action = 0
                    obs_t = (routes_so_far, route_reps, remaining_budget, ci)
                    reward = route_rwd_tnsr[0, ci].item()
                    done = ci == len(route_reps) - 1
                    replay_buffer.add(obs_t, action, reward, None, done)

            else:
                for ii, ri in enumerate(ep_route_idxs):
                    routes_so_far = ep_route_idxs[:ii]
                    remaining_budget = \
                        budget - old_req_cpcties[routes_so_far].sum()
                    obs_t = (np.array(routes_so_far.cpu()), 
                             route_reps, np.array(remaining_budget.cpu()))
                    action = ri.item()
                    reward = route_rwd_tnsr[0, ii].item()
                    done = ii == len(ep_route_idxs) - 1
                    replay_buffer.add(obs_t, action, reward, None, done)

            n_minibatches = max(route_rwd_tnsr.shape[1] * 4 // minibatch_size, 1)
            if n_minibatches * minibatch_size * 16 > len(replay_buffer):
                continue

            # sample some minibatches from replay buffer and learn on them
            log.debug("training on minibatches")

            # learn on 1 minibatch for each 4 steps in the new trajectory
            minibatch_losses = []
            for _ in range(n_minibatches):
                # sample and format the minibatch
                log.debug("assembling minibatch")
                # assemble numpy arrays for the minibatch
                obs_mb, act_mb, reward_mb, _, done_mask, weights, idxes = \
                    replay_buffer.sample(minibatch_size, beta)
                # obs_mb, act_mb, reward_mb, _, done_mask = \
                #     replay_buffer.sample(minibatch_size)
                mb_state_routes = []
                mb_costs = np.zeros((minibatch_size, n_cdts))
                mb_budgets = np.zeros(minibatch_size)
                mb_cdt_action_idxs = np.zeros(minibatch_size)
                act_mb = torch.tensor(act_mb, device=DEVICE)
                mb_next_state_routes = []
                mb_routereps = []
                for ii, obs in enumerate(obs_mb):
                    routes_so_far, route_rep_obs, rmng_bdgt = obs[:3]

                    routes_tnsr = torch.tensor(routes_so_far, device=DEVICE)
                    mb_state_routes.append(routes_tnsr)
                    if binary_rollouts:
                        cdt_idx = obs[3]
                        mb_cdt_action_idxs[ii] = cdt_idx

                    if not binary_rollouts or act_mb[ii] == 1:
                        next_state_routes = \
                            torch.cat((routes_tnsr, act_mb[ii][None]))
                    else:
                        next_state_routes = routes_tnsr
                    mb_next_state_routes.append(next_state_routes)

                    mb_costs[ii] = [rr.cost for rr in route_rep_obs]
                    mb_budgets[ii] = rmng_bdgt
                    mb_routereps.append(route_rep_obs)

                # compute frequencies from costs
                mb_freqs = self.sim.capacity_to_frequency(mb_costs)
                mb_masks = torch.tensor(mb_freqs < self.min_freq, 
                                        device=DEVICE)
                # convert the numpy arrays to pytorch tensors                
                mb_costs = torch.tensor(mb_costs, device=DEVICE, 
                                        dtype=torch.float32)
                mb_rewards = torch.tensor(reward_mb, device=DEVICE, 
                                          dtype=torch.float32)
                mb_budgets = torch.tensor(mb_budgets, device=DEVICE,
                                          dtype=torch.float32)
                done_mask = torch.tensor(done_mask, dtype=bool, device=DEVICE)
                weights = torch.tensor(weights, device=DEVICE)
                mb_idxs = torch.arange(minibatch_size, device=DEVICE)
                mb_next_budgets = mb_budgets - mb_costs[mb_idxs, act_mb]

                if binary_rollouts:
                    mb_cdt_action_idxs = torch.tensor(mb_cdt_action_idxs,
                                                      device=DEVICE, dtype=int)
                    mb_next_cdt_action_idxs = mb_cdt_action_idxs + 1
                    mb_next_cdt_action_idxs[done_mask] = 0
                else:
                    mb_cdt_action_idxs = None
                    mb_next_cdt_action_idxs = None

                # run the model on the minibatch
                log.debug("forward pass on minibatch")
                self.model.train()

                _, q_ests = \
                    self.model.step(
                        self.env_rep, mb_routereps, mb_state_routes, mb_budgets, 
                        mask=mb_masks, next_actions=act_mb, 
                        action_cdt_idx=mb_cdt_action_idxs)
                if torch.std(q_ests) == 0:
                    log.warning(f"std dev of q est: {torch.std(q_ests)}")

                with torch.no_grad():
                    # double DQN: choose next actions with the online policy...
                    next_acts, _ = \
                        self.model.step(
                            self.env_rep, mb_routereps, mb_next_state_routes, 
                            mb_next_budgets, mask=mb_masks, 
                            action_cdt_idx=mb_next_cdt_action_idxs)
                    # avoid giving invalid indexes to the next step
                    next_acts[done_mask] = 0
                    assert next_acts.max() < len(candidate_routes)
                    # ...but estimate their values with the target policy
                    _, next_q_ests = tgt_model.step(
                        self.env_rep, mb_routereps, mb_next_state_routes, 
                        mb_next_budgets, next_actions=next_acts, 
                        mask=mb_masks, action_cdt_idx=mb_next_cdt_action_idxs)

                # we know that the value after the terminal state is 0.
                next_q_ests[done_mask] = 0
                # compute the loss
                log.debug("computing errors")
                objective = mb_rewards + self.route_gamma * next_q_ests
                td_errors = q_ests - objective
                squared_td_errors = td_errors**2
                loss = (weights * squared_td_errors).mean()
                # loss = squared_td_errors.mean()
                minibatch_losses.append(squared_td_errors.mean().item())
                
                priorities = np.array(td_errors.abs().detach().cpu())

                # update weights
                log.debug("backprop and weight update")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                log.debug(f"epsilon: {epsilon} beta: {beta}")
                if tau > 0:
                    # update target model
                    for aa, bb in zip(self.model.parameters(), 
                                      tgt_model.parameters()):
                        bb.data.mul_(1-tau).add_(tau * aa)

                # update the priorities of the sampled minibatch
                # add an epsilon since replay buffer doesn't accept priority 0
                replay_buffer.update_priorities(idxes, priorities + 1e-6)

            # anneal beta and epsilon after each batch
            if epsilon > final_epsilon:
                epsilon -= epsilon_step
                epsilon = max(final_epsilon, epsilon)
            if beta < final_beta:
                beta += beta_step
                beta = min(final_beta, beta)
            
            # log the loss

            if len(minibatch_losses) > 0:
                avg_loss = np.mean(minibatch_losses)
            else: 
                avg_loss = 0
            self.log_scalar("unweighted loss", avg_loss, batch_count)

            # periodically evaluate the model in greedy mode
            if batch_count > 0 and batch_count % eval_per_n_eps == 0:
                log.debug("estimating greedy performance")
                # compute the greedy performance of the new model
                routes, freqs, quality, info = \
                    self.eval_fixed_capacity_model(route_reps, alg)

                self.log_greedy(routes, freqs, quality, info, batch_count)

                if quality > best_quality:
                    image = self.sim.get_network_image(routes, freqs, 
                        show_demand=True)
                    self.sumwriter.add_image('City with routes', image, 
                                             batch_count)
                    best_model = self.update_best(routes, freqs, quality, 
                                                  batch_count)
                    best_quality = quality
                    best_routes = routes
                    best_freqs = freqs

            # update the route representations with the new capacities.
            route_reps = \
                RouteRep.get_updated_costs_collection(route_reps, req_cpcties)


        # compute the greedy performance of the final model
        log.info("estimate greedy performance")
        routes, freqs, quality, info = \
            self.eval_fixed_capacity_model(route_reps, alg)

        self.log_greedy(routes, freqs, quality, info, batch_count)
        if quality > best_quality:
            best_model = self.update_best(routes, freqs, quality, batch_count)
            best_quality = quality
            best_routes = routes
            best_freqs = freqs

        # return best values
        image = self.sim.get_network_image(best_routes, best_freqs, 
                                           show_demand=True)
        self.sumwriter.add_image('City with routes', image, batch_count)
        return best_model, best_routes, best_freqs

    def train_wo_learning_freqs(self, candidate_routes, init_capacity=272,
                                eval_per_n_batches=10, binary_rollouts=False,
                                req_cpcty_chkpt_path=None):
        baseline = 0
        alpha = 0.9

        scen_tracker = TotalInfoTracker(len(candidate_routes))

        log.info("training without learning frequencies")
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay, 
                                     maximize=True)
        n_cdts = len(candidate_routes)

        phase, req_cpcties, count_offset = \
            set_up_req_capacities(req_cpcty_chkpt_path, n_cdts, init_capacity)
        req_cpcties, candidate_routes = \
            zip(*sorted(zip(req_cpcties, candidate_routes), reverse=True))
        req_cpcties = torch.stack(req_cpcties).to(dtype=torch.float32)

        avgd_over = torch.ones(n_cdts, device=DEVICE, dtype=int)

        route_reps = self.sim.get_route_reps_for_nn(candidate_routes,
                                                    req_cpcties, 
                                                    device=DEVICE)
        routes, freqs, best_quality, info = \
            self.eval_fixed_capacity_model(route_reps, REINFORCE_ARG)

        self.log_greedy(routes, freqs, best_quality, info, 0)

        for ep_count in tqdm(range(0, self.num_episodes, self.eps_per_batch)):
            all_freqs = self.sim.capacity_to_frequency(req_cpcties)
            mask = all_freqs < self.min_freq 
            # plan, providing the req cpcties to the planner
            if phase == 0:
                budget = self.budget * \
                    torch.max(torch.tensor(0.1), 
                              torch.rand(self.eps_per_batch) ** 2)
            else:
                budget = torch.tensor([float(self.budget)])
            budget = budget.to(device=DEVICE)
            self.model.train()
            with torch.no_grad():
                plan_out = self.plan_and_log(ep_count, budget=budget,
                                             route_reps=route_reps, mask=mask)

            scen_tracker.add_scenarios(plan_out.routes_tensor)

            # determine frequencies from req capacities
            batch_freqs = []
            for routes, route_idxs in zip(plan_out.routes, 
                                          plan_out.routes_tensor):
                freqs = all_freqs[route_idxs[:len(routes)]]
                batch_freqs.append([ff.item() for ff in freqs])

            # simulate and update values based on results
            route_rwd_tnsr = torch.zeros(plan_out.route_logits.shape,
                                         device=DEVICE)
            all_rewards = []
            glbl_infos = defaultdict(list)
            old_req_cpcties = req_cpcties.clone()

            iterable = enumerate(zip(plan_out.routes_tensor, plan_out.routes, 
                                     batch_freqs))
            for ep_idx, (route_idxs, routes, freqs) in iterable:
                if phase == 0:
                    # simulate capacity-free
                    per_stop, glbl = \
                        self.sim.run(routes, freqs, capacity_free=True)
                    self.update_route_info(req_cpcties, avgd_over, routes, 
                                           route_idxs, per_stop, 
                                           ep_count + ep_idx + count_offset)

                # simulate capacity-enabled to get rewards
                elif phase == 1:
                    per_stop, glbl = self.sim.run(routes, freqs)

                # accumulate statistics to log later
                for key, value in glbl.items():
                    glbl_infos[key].append(value)
                route_idxs = route_idxs[:len(routes)]
                if len(route_idxs) == 0:
                    budget_used = 0
                else:
                    budget_used = old_req_cpcties[route_idxs].sum()
                if budget.size()[0] == 1:
                    ep_budget = budget[0]
                else:
                    ep_budget = budget[ep_idx]
                pcnt_budget_used = budget_used * 100 / ep_budget
                glbl_infos["budget used (%)"].append(pcnt_budget_used.item())

                stop_rewards = per_stop["rewards"]

                for ii, route_stop_rewards in enumerate(stop_rewards):
                    route_reward = route_stop_rewards.sum()
                    if binary_rollouts:
                        # tensor is 1 x n_candidates
                        # ci = plan_out.routes_tensor[ep_idx, ii]
                        # route_rwd_tnsr[ep_idx, ci] = route_reward
                        route_rwd_tnsr[ep_idx, -1] += route_reward
                    else:
                        # tensor is 1 x n_selected_routes
                        route_rwd_tnsr[ep_idx, ii] = route_reward
                    all_rewards.append(route_reward)

                # if phase == 1:
                #     # update req capacities based on capacity-enabled sim
                #     self.update_route_info(req_cpcties, avgd_over, routes, 
                #                            route_idxs, per_stop, 
                #                            ep_count + ep_idx + count_offset)

            if self.maxent_factor > 0:
                ent_logits = plan_out.route_logits.detach()
                ent_logits[ent_logits == -float("inf")] = 0
                route_rwd_tnsr -= self.maxent_factor * ent_logits

            # log average statistics of the batch
            glbl_info = {kk: np.mean(vv) for kk, vv in glbl_infos.items()}
            for ik, iv in glbl_info.items():
                self.log_scalar(ik, iv, ep_count)

            raw_returns = lrnu.rewards_to_returns(route_rwd_tnsr, 
                                                  self.route_gamma)
            returns = raw_returns - baseline
            baseline = alpha * baseline + (1 - alpha) * raw_returns.mean()

            # do backprop one step at a time, so we only have to store one
             # step's computation graphs at a time
            budget_before_act = budget.repeat(self.eps_per_batch)
            batched_routereps = [route_reps] * self.eps_per_batch
            act_tensor = plan_out.routes_tensor
            act_tensor[act_tensor == len(candidate_routes)] = -1
            # don't apply any gradient to invalid actions
            returns[act_tensor == -1] = 0
            batch_mask = mask.expand(self.eps_per_batch, -1)
            for step_idx in range(act_tensor.shape[1]):
                curr_state = act_tensor[:, :step_idx]
                curr_acts = act_tensor[:, step_idx]
                _, scores = self.model.step(
                    self.env_rep, batched_routereps, curr_state, 
                    budget_before_act, mask=batch_mask, next_actions=curr_acts)
                loss = (scores * returns[:, step_idx]).sum()
                loss.backward()

                budget_before_act -= old_req_cpcties[curr_acts]

            self.log_scalar('route reward', np.mean(all_rewards), ep_count)
            self.log_scalar('n scenarios seen', 
                            scen_tracker.n_scenarios_so_far, ep_count)

            # if self.use_baseline:
            #     # TODO remove this or make it use the proper baseline code above
            #     val_est_loss = ((plan_out.route_est_vals - returns)**2).mean()
            #     self.log_scalar("value loss", val_est_loss.item(), ep_count)
            #     loss = val_est_loss
            #     baseline_returns = returns - plan_out.route_est_vals.detach()
            #     if (ep_count // self.eps_per_batch) % 2:
            #         # only update the policy every other batch, to let the 
            #          # baseline catch up.
            #         loss += (plan_out.route_logits * baseline_returns).sum()
            # else:
            #     loss = (plan_out.route_logits * returns).sum()

            if phase == 1:
                log.debug("weight update")
                optimizer.step()
                optimizer.zero_grad()

            self.log_scalar('total exploration', scen_tracker.total_info(), 
                            ep_count)

            # periodically evaluate the model in greedy mode
            if ep_count == self.num_episodes - 1 or \
               (ep_count // self.eps_per_batch > 0 and 
                ep_count % (eval_per_n_batches * self.eps_per_batch) == 0):
                log.debug("estimate greedy performance")
                # compute the greedy performance of the new model
                routes, freqs, quality, info = \
                    self.eval_fixed_capacity_model(route_reps, REINFORCE_ARG)
                self.log_greedy(routes, freqs, quality, info, ep_count)

                if quality > best_quality:
                    best_model = self.update_best(routes, freqs, quality, 
                                                  ep_count)
                    best_quality = quality
                    best_routes = routes
                    best_freqs = freqs

            if phase == 0 and (avgd_over > 1).all():
                # we've tried every route at least once, so move on
                log.info(f"phase 0 is over after {ep_count} eps, "
                         "starting phase 1")
                phase = 1

            # update route representations for the next iteration
            route_reps = \
                RouteRep.get_updated_costs_collection(route_reps, req_cpcties)

        # compute the greedy performance of the final model
        log.info("estimate greedy performance")
        routes, freqs, quality, info = \
            self.eval_fixed_capacity_model(route_reps, REINFORCE_ARG)
        baseline = info["average return"]
        self.log_greedy(routes, freqs, quality, info, ep_count)
        if quality > best_quality:
            best_model = self.update_best(routes, freqs, quality, 
                                            ep_count)
            best_quality = quality
            best_routes = routes
            best_freqs = freqs    

        # return best values
        image = self.sim.get_network_image(best_routes, best_freqs, 
                                           show_demand=True)
        self.sumwriter.add_image('City with routes', image, ep_count)
        return best_model, best_routes, best_freqs

    def train_seeded_reinforce(self, n_routes, sim_penalty_factor=1, 
                               eval_per_n_batches=10):
        torch.autograd.set_detect_anomaly(True)
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)
        env_representation = self.sim.get_env_rep_for_nn(device=DEVICE)

        # sample evaluation seeds from a 0-mean 1-std normal.
        eval_seeds = torch.normal(0., 1., (1, n_routes, self.model.embed_dim),
                                  device=DEVICE)
        dists = torch.pdist(eval_seeds.squeeze())
        log.info(
            f"average seed distance mean: {dists.mean()} std: {dists.std()}")

        # 1 km is about 4x the average inter-stop distance in Laval
        routes, freqs, best_quality, info = \
            self.evaluate_model(env_representation, eval_seeds)

        self.log_greedy(routes, freqs, best_quality, info, 0)
        best_model = self.update_best(routes, freqs, best_quality, 0)
        best_routes = routes
        best_freqs = freqs
        val_loss_fn = torch.nn.MSELoss()

        threshold = self.sim.basin_radius_m * 2
        unclipped_stop_sub_costs = self.sim.crow_dists / threshold

        stop_sub_costs = torch.clamp(unclipped_stop_sub_costs, max=1)
        for ep_count in tqdm(range(0, self.num_episodes, self.eps_per_batch)):
            self.model.train()
            seeds = torch.normal(0., 1., (self.eps_per_batch, n_routes, 
                                          self.model.embed_dim), 
                                 device=DEVICE)
            result = self.plan_and_simulate(env_representation, ep_count, 
                                            seeds=seeds)

            # compute the learning signal
            log.info("computing the learning signal")
            batch_signal = 0
            # iterate over stop, route, and frequency data
            for atr in result:
                if atr is None or type(atr) is not ActTypeResult:
                    # we're only processing ActTypeResults here
                    continue
                action_signal = atr.returns
                if self.use_baseline and atr.est_vals is not None:
                    action_signal -= atr.est_vals.detach()
                    batch_signal += val_loss_fn(atr.est_vals, atr.returns)
                batch_signal -= (atr.logits * action_signal).sum()

            # compute similarity signal
            log.info("computing similarity penalties")
            penalties = torch.zeros(result.stop_data.logits.shape, 
                                    device=DEVICE)
            avg_pnlties = []
            for ep_idx in range(self.eps_per_batch):
                ep_routes = result.routes[ep_idx]
                ep_seeds = seeds[ep_idx]
                pnlty = self.compute_similarity_penalties(ep_routes, ep_seeds, 
                                                          stop_sub_costs)
                penalties[ep_idx, :pnlty.shape[0], :pnlty.shape[1]] = pnlty
                avg_pnlties.append(pnlty.mean().item())

            penalties *= sim_penalty_factor
            batch_signal += (result.stop_data.logits * penalties).sum()
                
            self.log_scalar('similarity penalty', np.mean(avg_pnlties), 
                            ep_count)
            avg_obj = batch_signal.item() / self.eps_per_batch
            self.log_scalar('objective', avg_obj, ep_count)

            # update weights
            log.info("backprop and weight update")
            optimizer.zero_grad()
            batch_signal.backward()
            optimizer.step()

            if ep_count == self.num_episodes - 1 or \
               (ep_count // self.eps_per_batch > 0 and
                ep_count % (eval_per_n_batches * self.eps_per_batch) == 0):
                log.info("estimate greedy performance")
                # compute the greedy performance of the new model
                routes, freqs, quality, info = \
                    self.evaluate_model(env_representation, eval_seeds)
                self.log_greedy(routes, freqs, quality, info, ep_count)

                if quality > best_quality:
                    best_model = self.update_best(routes, freqs, quality, 
                                                  ep_count)
                    best_quality = quality
                    best_routes = routes
                    best_freqs = freqs    
        
        image = self.sim.get_network_image(best_routes, best_freqs)
        self.sumwriter.add_image('City with routes', image, ep_count)
        return best_model, best_routes, best_freqs


    def compute_similarity_penalties(self, routes, seeds, stop_sub_costs):
        if len(routes) == 0:
            # return an empty 2D tensor
            return torch.empty((0, 0))

        max_route_len = max([len(rr) for rr in routes])
        routes_stop_penalties = torch.zeros((len(routes), max_route_len))
    
        # compute seed distances
        seed_dists = square_pdist(seeds)
        seed_exp_dists = 1 - torch.exp(-seed_dists)
        for (i1, r1), (i2, r2) in combinations(enumerate(routes), 2):
            _, sp1, sp2 = soft_levenshtein_ratio(r1, r2, stop_sub_costs)
            # if seeds are the same, there should be no penalty
            routes_stop_penalties[i1, :len(sp1)] += sp1 * seed_exp_dists[i1, i2]
            routes_stop_penalties[i2, :len(sp2)] += sp2 * seed_exp_dists[i1, i2]

        return routes_stop_penalties.to(DEVICE)


def soft_levenshtein_ratio(route1, route2, sub_costs=None):
    interim_dist_matrix = torch.zeros((len(route1) + 1, len(route2) + 1))
    min_choices = -torch.ones((len(route1) + 1, len(route2) + 1))
    # 0 corresponds to choosing "up"
    min_choices[1:, 0] = 0
    # 1 corresponds to choosing "left"
    min_choices[0, 1:] = 1

    # compute the interim distances
    for ii in range(len(route1)):
        interim_dist_matrix[ii+1, 0] = ii + 1
    for jj in range(len(route2)):
        interim_dist_matrix[0, jj+1] = jj + 1
    for ii, s1 in enumerate(route1):
        for jj, s2 in enumerate(route2):
            if sub_costs is not None:
                cost = sub_costs[s1, s2]
            elif s1 == s2:
                cost = 0
            else:
                cost = 1
            candidates = (interim_dist_matrix[ii, jj+1] + 1,
                          interim_dist_matrix[ii+1, jj] + 1,
                          interim_dist_matrix[ii, jj] + cost)
            min_idx = np.argmin(candidates)
            min_choices[ii + 1, jj + 1] = min_idx
            interim_dist_matrix[ii + 1, jj + 1] = candidates[min_idx]

    lev_dist = interim_dist_matrix[len(route1), len(route2)]

    stop_costs_1 = torch.zeros(len(route1))
    stop_costs_2 = torch.zeros(len(route2))
    # walk backwards to determine which stop contributes which cost
    idx = (len(route1), len(route2))
    while idx != (0, 0):
        min_choice = min_choices[idx]
        cur_dist = interim_dist_matrix[idx]
        if min_choice == 0:
            # min was same col, previous row
            prev_idx = (idx[0] - 1, idx[1])
        elif min_choice == 1:
            # min was same row, previous col
            prev_idx = (idx[0], idx[1] - 1)
        elif min_choice == 2:
            # min was previous row, previous col
            prev_idx = (idx[0] - 1, idx[1] - 1)
        else:
            raise ValueError("Invalid min choice index found!")
        prev_dist = interim_dist_matrix[prev_idx]
        move_cost = cur_dist - prev_dist
        if min_choice == 0:
            stop_costs_1[prev_idx[0]] = move_cost
        elif min_choice == 1:
            stop_costs_2[prev_idx[1]] = move_cost
        elif min_choice == 2:
            stop_costs_1[prev_idx[0]] = move_cost
            stop_costs_2[prev_idx[1]] = move_cost
        idx = prev_idx

    max_possible_dist = max(len(route1), len(route2))
    lev_ratio = 1 - lev_dist / max_possible_dist
    per_stop_ratios_1 = (1 - stop_costs_1) / max_possible_dist
    per_stop_ratios_2 = (1 - stop_costs_2) / max_possible_dist

    return lev_ratio, per_stop_ratios_1, per_stop_ratios_2


def set_up_req_capacities(checkpoint_path, n_candidates, init_capacity=272):
    if checkpoint_path is not None and checkpoint_path.exists():
        with open(checkpoint_path, 'rb') as ff:
            req_cpcty_checkpoint = pickle.load(ff)

        req_cpcties, count_offset = req_cpcty_checkpoint
        log.info(f"total req. capacity: {req_cpcties.sum()}")
        req_cpcties = torch.tensor(req_cpcties, device=DEVICE)
        phase = 1
    else:
        req_cpcties = torch.ones(n_candidates, device=DEVICE) * init_capacity
        count_offset = 0
        phase = 0

    return phase, req_cpcties, count_offset

def log_config(summary_writer, args, config_dict, computed_params):
    lrnu.log_model_args(args, summary_writer)
    lrnu.log_learning_args(args, summary_writer)
    summary_writer.add_text('stop discount rate', str(args.sdr), 0)
    summary_writer.add_text('route discount rate', str(args.rdr), 0)
    summary_writer.add_text('frequency discount rate', str(args.fdr), 0)
    summary_writer.add_text('# batches', str(args.num_batches), 0)
    summary_writer.add_text('Max. Ent. factor', str(args.maxent), 0)
    summary_writer.add_text('training alg', str(args.alg), 0)
    summary_writer.add_text('reward function', str(args.reward), 0)
    summary_writer.add_text('use baseline', str(args.bl), 0)
    summary_writer.add_text('number of routes to generate', str(args.ng), 0)
    summary_writer.add_text('use newton routes', str(args.newton), 0)
    summary_writer.add_text('similarity penalty scale', str(args.sp), 0)
    summary_writer.add_text('budget', str(args.budget), 0)
    summary_writer.add_text('quality metric', str(args.qm), 0)
    summary_writer.add_text('binary rollouts', str(args.binary), 0)

    for key, value in computed_params.items():
        summary_writer.add_text(key, str(value), 0)

    def log_config_recurse_helper(cfg_container, prefix=None):
        if type(cfg_container) is dict:
            iterator = cfg_container.items()
        elif type(cfg_container) is list:
            iterator = enumerate(cfg_container)
        else:
            raise ValueError("This should only be called with a dict or list!")

        for key, val in iterator:
            label = str(key)
            if prefix:
                # preappend the prefix and a separator
                label = prefix + '/' + label
            
            if type(val) in [list, dict]:
                # recurse on this sub-collection
                log_config_recurse_helper(val, label)
            else:
                # add this parameter
                summary_writer.add_text(label, str(val), 0)

    log_config_recurse_helper(config_dict)


def get_args():
    parser = argparse.ArgumentParser()
    cfg_grp = parser.add_mutually_exclusive_group()
    cfg_grp.add_argument('--sim', help="The simulator configuration file.")
    cfg_grp.add_argument('--newton',
        help="The newton-route generation configuration file.")
    parser.add_argument('--binary', action="store_true",
        help="If provided, do binary policy rollouts: consider each route "\
            "one at a time in a predefined sequence.  Otherwise, consider "\
            "all routes at each step.")
    parser.add_argument('--sdr', type=float, default=0.0, 
        help="The exponential discount rate (gamma) for stop rewards.")
    parser.add_argument('--rdr', type=float, 
        help="The exponential discount rate (gamma) for route rewards.")
    parser.add_argument('--fdr', type=float, default=0.1, 
        help="The exponential discount rate (gamma) for frequency rewards.")
    parser.add_argument('--num_batches', '--nb', type=int, default=128, 
        help="The number of batches to train on.")
    parser.add_argument('--bl', action="store_true",
        help="If true, equip model with an NN value estimator.")
    parser.add_argument('--alg', choices=[REINFORCE_ARG, QLEARNING_ARG, GFLOWNET_ARG], 
        default=REINFORCE_ARG, help="The training algorithm to use.")
    parser.add_argument('--reward', choices=[SATDEM_RWD_FN, BAGLOEE_RWD_FN, 
                                             QUADRATIC_RWD_FN, GLOBAL_RWD_FN], 
        default=SATDEM_RWD_FN, help="The training algorithm to use.")
    parser.add_argument('--maxent', type=float, default=0,
        help="Constant by which entropy term is scaled.  0 (disabled) by default.")
    parser.add_argument("--budget", "-b", type=int, default=28200, 
                        help="budget in seats for the run")
    parser.add_argument("--qm", default="saved time", 
                        help="quality metric to use during training")
    parser.add_argument('--sp', type=float, default=1,
        help="Scale for the similarity penalty when generating routes")
    parser.add_argument('--gtfs',
        help="include choosing routes specified in the given gtfs directory.")
    parser.add_argument('--encweights', help="a set of pretrained weights to"\
        "use for the encoder.")
    parser.add_argument('--ng', type=int, 
        help="Number of routes to generate, if generating routes")

    parser = lrnu.get_model_args(parser)
    parser = lrnu.get_standard_learning_args(parser)
    
    args = parser.parse_args()
    # check validity of provided arguments
    assert args.sim is not None or args.newton is not None, \
        "Either a simulator config file or a newton-route config file must " \
        "be provided!"

    assert (args.ng is None) == (args.sim is None), \
        "You must specify the number of routes to generate!"

    if args.alg in [QLEARNING_ARG, GFLOWNET_ARG]:
        # the following reward functions aren't allowed
        assert args.reward not in \
            [BAGLOEE_RWD_FN, QUADRATIC_RWD_FN, GLOBAL_RWD_FN]
    if args.alg == QLEARNING_ARG:
        assert args.batch_size == 1
    if args.sim is not None:
        # insist on global reward if we're generating new routes, since 
         # binary-stop-inclusion needs rewards for unchosen stops
        assert args.reward == GLOBAL_RWD_FN
        assert args.ng is not None
    
    if args.rdr is None:
        if args.alg == REINFORCE_ARG:
            args.rdr = 0.0
        elif args.alg == QLEARNING_ARG:
            args.rdr = 1.0

    return args


def run(args):
    # TODO rework this to use hydra config
    global DEVICE
    DEVICE, run_name, summary_writer = \
        lrnu.process_standard_learning_args(args)

    # set the numpy random seed since we use numpy randomness here.
    if not args.noseed:
        np.random.seed(args.seed)

    model_args = {"embed_dim": args.embed, 
                  "n_heads": args.nheads,
                  "mode": ALG_MODES[args.alg],
                  "binary_rollouts": args.binary,
                  "max_budget": args.budget,
                  }

    if args.sim:
        log.info(f'generating {args.ng} routes')
        sim = TimelessSimulator(args.sim,
                                # stops_path="/localdata/ahollid/laval/gtfs/stops.txt"
                                )
        kept_nodes, _ = bagloee_filtering(sim)
        sim.filter_nodes(kept_nodes)

    else:
        # using newton pregenerated routes
        sim, pregen_routes = \
            get_newton_routes_from_config(args.newton, device=DEVICE)

        log.info(f"{len(pregen_routes)} routes generated")                 
        if args.gtfs:
            # parse the routes from the gtfs
            _, gtfs_routes, _ = sim.translate_gtfs(args.gtfs, 
                                                   kept_nodes_only=True)
            pregen_routes += gtfs_routes
            log.info(f"adding {len(gtfs_routes)} routes")

        route_lens = [len(rr) for rr in pregen_routes]
        log.info(f"Route len mean: {np.mean(route_lens)} min: " \
                 f"{min(route_lens)} max: {max(route_lens)}")
        model_args['pregenerated_routes'] = pregen_routes

    # node_features, adj_mat, demand_mat = sim.get_env_rep_for_nn(device=DEVICE)
    env_rep = sim.get_env_rep_for_nn(device=DEVICE)
    model_args["env_rep"] = env_rep

    computed_params = {}

    # this is sort of ad-hoc.
    # / total demand * 0.5 makes max possible return 2 if all demand is 
    # satisfied at 0 cost, which will never happen; realistically we might then
    # get to 1 (although for some large balancing constant, we could get close
    # to 2). * 79 since the real system has 79 routes, so the max possible 
    #  *per-route* reward is 2, hopefully bringing the typical scale close to 
    # -1 to 1.
    # # per_stop_scale = 79 / (0.5 * sim.total_demand)
    # per_stop_scale = 0.00450213
    # per_stop_scale = 1
    bagloee_max_savedtime = 3519400
    per_stop_scale = 1 / 50
    if args.reward == BAGLOEE_RWD_FN:
        global_scale = 25 / bagloee_max_savedtime
    else:
        global_scale = 50 / sim.total_demand

    log.info(f"sim per-stop reward scale is {per_stop_scale}")
    computed_params["per-stop reward scale"] = per_stop_scale
    log.info(f"sim global reward scale is {global_scale}")
    computed_params["global reward scale"] = global_scale
    # inverse of saved time achieved by Bagloee
    if args.reward == SATDEM_RWD_FN:
        reward_fn = get_satdemand_reward_fn(per_stop_scale)
    elif args.reward == BAGLOEE_RWD_FN:
        reward_fn = get_bagloee_reward_fn(per_stop_scale, global_scale,
                                          "saved time")
    # elif args.reward == QUADRATIC_RWD_FN:
    #     reward_fn = get_quadratic_reward_fn(per_stop_scale, global_scale, args.qm)
    elif args.reward == GLOBAL_RWD_FN:
        reward_fn = get_global_reward_fn(global_scale, "saved time")

    sim.per_stop_reward_fn = reward_fn

    model_args["n_encoder_layers"] = args.nel

    model = get_twostage_planner(**model_args)

    if args.fromsaved is not None:
        model.load_state_dict(torch.load(args.fromsaved))
    if args.encweights is not None:
        model.set_pretrained_state_encoder(args.encweights)

    model.to(DEVICE)

    # add the configuration to the summary writer
    log_config(summary_writer, args, sim.cfg, computed_params)
    trainer = Trainer(model, sim, args.budget, args.qm, args.num_batches, 
                      args.batch_size, args.sdr, args.rdr, args.fdr, 
                      args.learning_rate, args.decay, summary_writer, args.bl, 
                      args.maxent)
    
    if args.newton:
        config_path = Path(args.newton)
        pkl_filename = config_path.stem + 'required_capacities.pkl'
        cpcties_path = config_path.parent / pkl_filename

        if args.alg == REINFORCE_ARG:
            # best_model, best_routes, best_freqs = trainer.train_reinforce()
            best_model, best_routes, best_freqs = \
                trainer.train_wo_learning_freqs(pregen_routes, 
                    binary_rollouts=args.binary,
                    req_cpcty_chkpt_path=cpcties_path)
        elif args.alg == QLEARNING_ARG:
            alpha = 0.6
            beta = 0.5
            # ~95% of weights are replaced after...
                # 2400 minibatches = 4 coverages of 5 episodes of 120 steps
            tau = 0.0023
            # # this would give 99%
            # tau = 0.002
            # eps per run * approx. steps per ep = steps per run
            # divide by 10 to get 1/10th the total number of steps
            num_eps = args.num_batches
            approx_steps_per_ep = 20
            n_transitions = num_eps * approx_steps_per_ep
            # store 1/10th of all transitions
            buffer_size = n_transitions // 10
            # 32 * 25 = 800
            minibatch_size = 32
            best_model, best_routes, best_freqs = \
                trainer.train_q_network_wo_freqs(
                    pregen_routes, alpha, beta, minibatch_size, buffer_size, 
                    tau, binary_rollouts=args.binary, 
                    req_cpcty_chkpt_path=cpcties_path)
        elif args.alg == GFLOWNET_ARG:
            # same as in their molecular synthesis code...
            buffer_size = 1000
            # 99% replaced after 43 iterations
            tau = 0.1
            leaf_coef = 50
            best_model, best_routes, best_freqs = \
                trainer.train_gflownet_wo_freqs(
                    pregen_routes, buffer_size, tau=tau, leaf_coef=leaf_coef,
                    minibatch_size=16, req_cpcty_chkpt_path=cpcties_path)

    else:
        if args.alg == QLEARNING_ARG:
            raise ValueError("Q learning not supported for route learning")
        best_model, best_routes, best_freqs = \
            trainer.train_shortestpath_reinforce(n_routes=args.ng)

    output_dir = Path("outputs")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    torch.save(best_model.state_dict(), output_dir / (run_name + '.pt'))
    metadata = {"best routes": best_routes,
                "best freqs": best_freqs,
                "args": args, 
                "sim config": sim.cfg, 
                "computed params": computed_params,
                "run type": "my method"}
    output_dir = Path("outputs")
    if not output_dir.exists():
        output_dir.mkdir()
    with open(output_dir / (run_name + '_meta.pkl'), "wb") as ff:
        pickle.dump(metadata, ff)

    summary_writer.close()


if __name__ == "__main__":
    args = get_args()
    run(args)
