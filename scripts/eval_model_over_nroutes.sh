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

# args are weight file, dataset, output csv
weights=$1
dataset=$2
output_csv=$3
extra_args=("${@:4}")

# for ii in 1 2 4 6 8
for ii in 1 5 10 15 20
# for ii in 5 10 15 20 35 50
do
	python learning/eval_route_generator.py --config-name=eval_model \
	    csv=True +model.weights=$weights +dataset=$dataset +n_routes=$ii \
        batch_size=64 hydra/job_logging=disabled >> $output_csv
done
