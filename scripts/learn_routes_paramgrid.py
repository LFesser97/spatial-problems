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

import yaml
import logging as log
import tempfile

from learning import learn_routes
from config_utils import expand_config_to_param_grid


def run_param_grid():
    args = learn_routes.get_args()
    with open(args.sim_config, "r") as ff:
        base_sim_cfg = yaml.load(ff, Loader=yaml.Loader)
    sim_cfgs = expand_config_to_param_grid(base_sim_cfg)
    log.info(f"Grid expanded to {len(sim_cfgs)} configs")

    base_name = args.run_name

    # run each separate configuration
    for ii, sim_cfg in enumerate(sim_cfgs):
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as ff:
            # write the config to a temp file
            yamlstr = yaml.dump(sim_cfg, Dumper=yaml.Dumper)
            ff.write(yamlstr)
            # set the temp file's name in args
            args.sim_config = ff.name

        # adjust the set route name if one was given
        if base_name:
            args.run_name = base_name + "_" + str(ii)

        learn_routes.run(args)


if __name__ == "__main__":
    run_param_grid()