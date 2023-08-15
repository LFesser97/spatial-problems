#!/bin/bash

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

script=$1
config=$2
extra_args=("${@:3}")

cities=(
    # Mandl
    mandl
    # Mandl
    # Mandl
    mumford0
    mumford1
    mumford2
    mumford3
)

for city in "${cities[@]}"
do
    python $script --config-name=$config +eval=$city ${extra_args[@]}
done
