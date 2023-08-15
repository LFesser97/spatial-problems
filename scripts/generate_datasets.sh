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

dataset_names=(
    "mixed"
    "mixed_test"
)
dataset_sizes=(
    32768
    2048
)
node_counts=(
    20
    50
    100
)

n_datasets=${#dataset_names[@]}
for n_nodes in "${node_counts[@]}"
do
    echo $n_nodes
    dataset_dir="datasets/${n_nodes}_nodes"
    for (( ii=0; ii<$n_datasets; ii++ ))
    do
        # ds_type=${dataset_types[$ii]}
        ds_name=${dataset_names[$ii]}
        ds_size=${dataset_sizes[$ii]}
        dataset="$dataset_dir/$ds_name"
        python learning/citygraph_dataset.py $dataset mixed -n $ds_size \
            --min $n_nodes --max $n_nodes --delete
    done
done
