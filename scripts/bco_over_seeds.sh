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

seeds=(
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
)

for ((ii=0; ii<${#seeds[@]}; ii++))
do
    seed=${seeds[$ii]}
    bash scripts/mumford_eval.sh learning/bee_colony.py bco_mumford \
        hydra/job_logging=disabled experiment/cost_function=mine \
        experiment.seed=$seed +run_name=balanced_seed_$seed \
        >> bco_seeds_balanced.csv
    bash scripts/mumford_eval.sh learning/bee_colony.py bco_mumford \
        hydra/job_logging=disabled experiment/cost_function=mine_pp \
        experiment.seed=$seed +run_name=pp_seed_$seed \
        >> bco_seeds_pp.csv
    bash scripts/mumford_eval.sh learning/bee_colony.py bco_mumford \
        hydra/job_logging=disabled experiment/cost_function=mine_op \
        experiment.seed=$seed +run_name=op_seed_$seed \
        >> bco_seeds_op.csv
done
