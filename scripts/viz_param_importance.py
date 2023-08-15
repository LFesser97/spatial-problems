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

import optuna
from matplotlib import pyplot as plt

from learning.tune_hyperparams import build_mysql_storage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("study_name", 
                        help="the name of the study to visualize")
    parser.add_argument("--mysql", action='store_true',
                        help="The database is MySQL, not SQLite.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-s", "--save", 
        help="Save the plot to this file instead of displaying it.")
    args = parser.parse_args()

    # load the study
    if args.mysql:
        storage = build_mysql_storage(args.study_name)
    else:
        storage = f"sqlite:///{args.study_name}.db"
    study = optuna.load_study(None, storage=storage)

    # get the importances
    fanova = optuna.importance.FanovaImportanceEvaluator(seed=args.seed)
    importances = optuna.importance.get_param_importances(study, 
                                                          evaluator=fanova)

    # plot the importances
    plt.bar(range(len(importances)), list(importances.values()))
    plt.xticks(range(len(importances)), list(importances.keys()), rotation=45,
               ha="right")
    if args.save:
        plt.savefig(args.save, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
