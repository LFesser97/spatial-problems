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

alphas=(
    0.0
    0.1
    0.2
    0.3
    0.4
    0.5
    0.6
    0.7
    0.8
    0.9
    1.0
)
betas=(
    1.0
    0.9
    0.8
    0.7
    0.6
    0.5
    0.4
    0.3
    0.2
    0.1
    0.0
)

for ((ii=0; ii<${#alphas[@]}; ii++))
do
    alpha=${alphas[$ii]}
    beta=${betas[$ii]}
    bash scripts/mumford_eval.sh learning/bee_colony.py bco_mumford \
        hydra/job_logging=disabled \
        experiment.cost_function.kwargs.demand_time_weight=$alpha \
        experiment.cost_function.kwargs.route_time_weight=$beta \
        experiment.seed=0 +run_name=alpha_$alpha \
        >> bco_pareto.csv
done