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

import sys
import pickle
import matplotlib.pyplot as plt

from simulation.timeless_sim import TimelessSimulator


def render_planned_routes(pickle_path, sim_cfg_path, outpath):
    with open(pickle_path, "rb") as ff:
        meta_dict = pickle.load(ff)
    routes = meta_dict["best routes"]
    freqs = meta_dict["best freqs"]

    sim = TimelessSimulator(sim_cfg_path, True)
    sim.render_plan_on_html_map(routes, freqs, outfile=outpath)

render_planned_routes(sys.argv[1], sys.argv[2], sys.argv[3])