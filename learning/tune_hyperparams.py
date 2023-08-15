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
import configparser

import torch
from torch import nn
import optuna
from optuna.storages import RetryFailedTrialCallback
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
from optuna.samplers import RandomSampler
from torch_geometric.loader import DataLoader
from omegaconf import OmegaConf

from learning.inductive_route_learning import setup_and_train, BLMODE_GREEDY, \
    BLMODE_NONE, BLMODE_ROLL, BLMODE_NN
from simulation.citygraph_dataset import CityGraphData
from learning.eval_route_generator import eval_model


# Model ranges
NSLFL_RANGE = (1, 3)
SLF_HIDDEN_RANGE = (2, 32)
NNPSL_RANGE = (1, 3)
ALPHA_RANGE = (0.0001, 0.1)
ENV_NET_TYPES = ["edge graph", "graph attn", "sgc", "none"]
ROUTE_NET_TYPES = ["edge graph", "graph attn", "none"]
DROPOUT_RANGE = (0.0, 0.5)
NONLINS = ["Tanh", "ReLU", "LeakyReLU", "GELU", "ELU"]

# Learning ranges
SPACE_SCALE_RANGE = (0.1, 1.0)
DMD_SCALE_RANGE = (0.1, 1.0)
LR_RANGE = (1e-7, 1e-2)
DECAY_RANGE = (1e-5, 1e-1)
NROUTES_RANGE = (4, 16)
BL_MODES = [BLMODE_NONE, BLMODE_GREEDY, BLMODE_ROLL, BLMODE_NN]


def suggest_cfg(trial, batch_size, n_epochs, dataset_path, eval_n_routes):
    embed_power = trial.suggest_int('embed_power', 3, 7)
    embed_dim = 2 ** embed_power
    common_model_cfg = {
        'dropout': trial.suggest_float("dropout", *DROPOUT_RANGE),
        # 'nonlin_type': trial.suggest_categorical("nonlin_type", NONLINS),
        'nonlin_type': 'ReLU',
        'embed_dim': embed_dim
    }

    use_norm = trial.suggest_categorical("use_norm", [True, False])

    # env net config
    # env_net_type = trial.suggest_categorical("env_net_type", ENV_NET_TYPES)
    env_net_type = 'graph attn'
    env_gn_cfg = {
        'net_type': env_net_type,
    }
    if env_net_type == 'none':
        env_gn_cfg['kwargs'] = {'return_edges': True}
    if env_net_type != 'none':
        env_kwargs = {
            'n_layers': 1,
            'in_edge_dim': 2,
            'in_node_dim': 8,
            'use_norm': use_norm,
        }
        if env_net_type == 'graph attn':
            head_power = trial.suggest_int("n_heads_power", 1, 3)
            n_heads = 2 ** head_power
            env_kwargs['n_heads'] = n_heads
        elif env_net_type == 'edge graph':
            env_layer_kwargs = {
                'n_edge_layers': trial.suggest_int("env_n_edge_layers", 1, 3),
                'hidden_dim': embed_dim * 2,
            }
            env_kwargs['layer_kwargs'] = env_layer_kwargs
        env_gn_cfg['kwargs'] = env_kwargs

    # route net config
    # route_net_type = trial.suggest_categorical("route_net_type", 
    #                                            ROUTE_NET_TYPES)
    route_net_type = 'graph attn'
    route_gn_cfg = {
        'net_type': route_net_type,
    }
    if route_net_type != 'none':
        route_kwargs = {
            'n_layers': 3,
            'in_edge_dim': 1,
            'use_norm': use_norm,
        }
        if env_net_type == 'none':
            route_kwargs['in_node_dim'] = 8

        if route_net_type == 'graph attn':
            if env_net_type != 'graph attn':
                head_power = trial.suggest_int("n_heads_power", 1, 3)
                n_heads = 2 ** head_power
            route_kwargs['n_heads'] = n_heads

        elif route_net_type == 'edge graph':
            route_gn_kwargs = {
                'n_edge_layers': trial.suggest_int("route_n_edge_layers", 1, 3),
                'hidden_dim': embed_dim * 2,
            }
            # if env_net_type == 'none':
            #     route_gn_kwargs['in_node_dim'] = 8
            route_kwargs['layer_kwargs'] = route_gn_kwargs

        # dense = trial.suggest_categorical("route_dense", [True, False])
        dense = True
        route_kwargs['dense'] = dense
        if not dense:
            residual = trial.suggest_categorical("route_residual", 
                                                 [True, False])
            route_kwargs['residual'] = residual
            recurrent = trial.suggest_categorical("route_recurrent", 
                                                  [True, False])
            route_kwargs['recurrent'] = recurrent

        route_gn_cfg['kwargs'] = route_kwargs

    # generator module config
    n_scorelenfn_layers = trial.suggest_int("n_scorelenfn_layers", *NSLFL_RANGE)
    if n_scorelenfn_layers > 1:
        scorelenfn_hidden_dim = trial.suggest_int("scorelenfn_hidden_dim", 
                                                  *SLF_HIDDEN_RANGE, log=True)
    else:
        # no hidden layers, so hidden dim is meaningless
        scorelenfn_hidden_dim = 1

    if env_net_type != "none":
        use_extra_dmd_feats = trial.suggest_categorical("use_extra_dmd_feats", 
                                                        [True, False])
    else:
        use_extra_dmd_feats = False
    use_extra_route_feats = trial.suggest_categorical("use_extra_route_feats", 
                                                      [True, False])
    # mask_used_paths = trial.suggest_categorical("mask_used_paths",
    #                                             [True, False])
    mask_used_paths = False
    feat_alpha = trial.suggest_float("feat_alpha", *ALPHA_RANGE, log=True),
    generator_cfg = {
        'feat_alpha': feat_alpha,
        'n_nodepair_layers': \
            trial.suggest_int("n_nodepair_layers", *NNPSL_RANGE),
        'n_scorelenfn_layers': n_scorelenfn_layers,
        'scorelenfn_hidden_dim': scorelenfn_hidden_dim,
        'use_extra_dmd_feats': use_extra_dmd_feats,
        'use_extra_route_feats': use_extra_route_feats,
        'mask_used_paths': mask_used_paths,
    }
    model_cfg = {'common': common_model_cfg,
                 'env_gn': env_gn_cfg,
                 'route_gn': route_gn_cfg,
                 'route_generator': generator_cfg}
    
    reward_scale = 1.0
    lr = trial.suggest_float("lr", *LR_RANGE, log=True)
    decay = trial.suggest_float("decay", *DECAY_RANGE, log=True)
    optimizer_name = 'Adam'
    space_scale = trial.suggest_float("space_scale", *SPACE_SCALE_RANGE)
    dmd_scale = trial.suggest_float("demand_scale", *DMD_SCALE_RANGE)
    # bl_mode = trial.suggest_categorical("bl_mode", BL_MODES)
    bl_mode = BLMODE_NN
    dr = trial.suggest_float("discount_rate", 0.0, 1.0)

    # batch_power = trial.suggest_int("batch_power", 3, 8)
    # batch_size = 2 ** batch_power

    cfg = {
        'model': model_cfg,
        'experiment': {
            'symmetric_routes': True,
            'mean_stop_time_s': 60,
        },
        'batch_size': batch_size,
        'n_epochs': n_epochs,
        'reward_scale': reward_scale,
        'lr': lr,
        'decay': decay,
        'optimizer': optimizer_name,
        'space_scale': space_scale,
        'demand_scale': dmd_scale,
        'baseline_mode': bl_mode,
        'discount_rate': dr,
    }
    # set up the dataset
    dataset_cfg = {
        'type': 'static',
        'kwargs': {
            'path': dataset_path,
            'space_scale': space_scale,
            'demand_scale': dmd_scale,
        }
    }
    cfg['dataset'] = dataset_cfg

    if bl_mode == BLMODE_ROLL:
        bl_alpha = trial.suggest_float("bl_alpha", *ALPHA_RANGE, log=True)
        cfg['bl_alpha'] = bl_alpha
    elif bl_mode == BLMODE_NN:
        cfg['bl_alpha'] = feat_alpha

    cfg['eval_n_routes'] = eval_n_routes

    return OmegaConf.create(cfg)


class Objective:
    def __init__(self, batch_size, n_epochs, dataset_path, eval_n_routes,
                 prune=False):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.dataset_path = dataset_path
        self.eval_n_routes = eval_n_routes
        self.prune = prune
        self.cfg = None

    def __call__(self, trial):
        self.cfg = suggest_cfg(trial, self.batch_size, self.n_epochs, 
                               self.dataset_path, self.eval_n_routes)
        if self.prune:
            train_trial = trial
        else:
            train_trial = None
        final_cost = setup_and_train(self.cfg, train_trial)

        trial.report(final_cost, self.n_epochs)

        return final_cost
    
    def callback(self, study, trial):
        if study.best_trial == trial:
            print("Best trial yaml cfg:")
            print(OmegaConf.to_yaml(self.cfg))


def build_mysql_storage(db_name, mysql_config_path='/home/ahollid/.my.cnf'):
    # load the mysql config
    config = configparser.ConfigParser()
    config.read(mysql_config_path)
    mysql_config = config['client']
    user = mysql_config['user']
    password = mysql_config['password'].strip("'").strip('"')
    host = mysql_config['host']
    dbstr = f"mysql+pymysql://{user}:{password}@{host}/{user}_{db_name}"

    storage = optuna.storages.RDBStorage(
        dbstr, heartbeat_interval=60, grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=3),
    )
    return storage


def tune_hyperparams():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="path to the dataset")
    parser.add_argument("study_name", help="the name of the study")
    parser.add_argument("--bs", default=32, type=int, help="batch size")
    parser.add_argument("--ne", "--n_epochs", type=int, default=15)
    parser.add_argument("-t", "--ntrials", type=int, default=100)
    parser.add_argument('--cpu', action='store_true',
        help="If true, run on the CPU.")
    parser.add_argument("--seed", type=int, default=0, 
        help="random seed to use")
    parser.add_argument("--noseed", action='store_true',
        help="if provided, no random seed is manually set")
    parser.add_argument("--prune", action='store_true',
        help="if provided, allow pruning of trials partway through.")
    parser.add_argument("--random", action='store_true',
        help="if provided, use optuna's random sampler instead of TPE.")
    parser.add_argument("--mysql", action='store_true',
        help="if provided, attempt to use a MySQL database with PyMySQL "\
             "instead of SQLite.  Use this for distributed runs.")
    parser.add_argument("--er", "--eval_routes", type=int, nargs='+', 
        default=[1, 5, 10, 15, 20], help='numbers of routes to evaluate over')

    args = parser.parse_args()

    # handle the random seed args
    if not args.noseed:
        torch.manual_seed(args.seed)

    # handle the device arg
    global DEVICE
    if args.cpu:
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda")

    if args.mysql:
        storage = build_mysql_storage(args.study_name)
    else:
        storage = f"sqlite:///{args.study_name}.db"

    sampler = None
    if args.random:
        sampler = RandomSampler()
        
    study = optuna.create_study(
        study_name=args.study_name,
        sampler=sampler,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=optuna.pruners.SuccessiveHalvingPruner())
    
    # load the datasets

    maxtrials_callback = MaxTrialsCallback(args.ntrials, 
        states=(TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL))
    obj = Objective(args.bs, args.ne, args.dataset, args.er, args.prune)

    # catch runtime errors in case GPU memory runs out
    study.optimize(obj, callbacks=[maxtrials_callback, obj.callback], 
                   catch=(RuntimeError,))

    pruned_trials = [tt for tt in study.trials 
                     if tt.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [tt for tt in study.trials 
                       if tt.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    tune_hyperparams()
