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

#!/bin/bash

cities=(
    # Mandl
    # Mandl
    # Mandl
    # Mandl
    Mumford0
    Mumford1
    Mumford2
    Mumford3
)

n_routes=(
    # 4
    # 6
    # 7
    # 8
    12
    15
    56
    60
)

n_iters=(
    # 72000000
    # 33000000
    # 24000000
    # 19000000
    5000000
    3000000
    524000
    365000
)

min_rls=(
    # 2
    # 2
    # 2
    # 2
    2
    10
    10
    12
)

max_rls=(
    # 8
    # 8
    # 8
    # 8
    15
    30
    22
    25
)

for (( ii=0; ii<${#cities[@]}; ii++))
do
    city=${cities[$ii]}
    nr=${n_routes[$ii]}
    ni=${n_iters[$ii]}
    min_rl=${min_rls[$ii]}
    max_rl=${max_rls[$ii]}
    python learning/hyperheuristics.py --config-name hyperheuristic_mumford \
        "$@" +eval.dataset.city=$city +min_route_len=$min_rl \
        +max_route_len=$max_rl +eval.n_routes="[$nr]" +n_iterations=$ni \
        hydra/job_logging=disabled
done
