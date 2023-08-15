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

# from pathlib import Path
import logging as log
import yaml
import matplotlib.pyplot as plt

import config_utils
from simulation.timeless_sim import TimelessSimulator


def get_node_id_routes(sim, gtfs_dir):
    node_id_routes = {}
    for route_ids, route, freq in zip(*sim.translate_gtfs(gtfs_dir)):
        # translate node indexes to node ids
        route = [sim.get_node_id_by_index(node_idx) for node_idx in route]
            
        # add to the dict
        if type(route_ids) is str:
            key = route_ids
        elif len(route_ids) == 1:
            key = route_ids[0]
        else:
            key = tuple(route_ids)
        
        node_id_routes[key] = (route, freq)
    
    return node_id_routes


def get_routes_as_yaml_dict(node_id_routes, route_ids, default_vehicle=None):
    yaml_dict = {}
    for route_id in route_ids:
        if route_id not in node_id_routes:
            log.warning(f"route {route_id} not found in gtfs routes!")
        else:
            route, freq = node_id_routes[route_id]
            stops_list = [{"node_id": ni} for ni in route]
            period_str = config_utils.float_time_to_str(1 / freq)
            route_dict = {}
            route_dict["trip_period"] = period_str
            if default_vehicle:
                route_dict["vehicle"] = default_vehicle.copy()
            route_dict["stops"] = stops_list
            yaml_dict[str(route_id)] = route_dict

    return yaml_dict


def render_selected_routes(sim, gtfs_dir):
    sim.render_gtfs(gtfs_dir, map_stops_to_nodes=False)
    # route_ids, routes, _ = sim.translate_gtfs(gtfs_dir)
    # route_ids, routes = zip(*[(route_id, route) 
    #                           for route_id, route in zip(route_ids, routes) 
    #                           if route_id in selected_route_ids])
    # sim.render_plan(routes, route_names=route_ids, show_node_labels=False, 
    #                 show_legend=True, map_nodes=False)
    # TODO render the full routes as described in the GTFS on top, as well as
    # the "snapped" routes?  Or even just the GTFS routes?
    plt.show()


def main():
    laval_cfg = "/localdata/ahollid/transit_learning/simulation/laval_cfg.yaml"
    sim = TimelessSimulator(laval_cfg, True)
    gtfs_dir = "/localdata/ahollid/laval/gtfs_nov2013"
    route_ids = ["AOUT1352E", "AOUT1352O",
              "AOUT1355N", "AOUT1355S",
              "AOUT13151N", "AOUT13151S",
              "AOUT13902N", "AOUT13902S",
              "AOUT13904N", "AOUT13904S", 
              "AOUT13925N", "AOUT13925S"]
    # start by writing the yaml
    node_id_routes = get_node_id_routes(sim, gtfs_dir)
    default_vehicle = {"id": "0",
                       "mode": "road",
                       "description": "regular bus",
                       "seats": 30,
                       "standing_room": 60,
                       "length_m": 18.0,
                       "avg_power_kW": 26.7,
                       }
    yaml_dict = get_routes_as_yaml_dict(node_id_routes, route_ids, 
                                        default_vehicle)
    outdir = "generated.yaml"
    with open(outdir, "w") as ff:
        yaml_str = yaml.dump(yaml_dict)
        ff.write(yaml_str)

    # then render the routes to know where to add the montreal external nodes
    render_selected_routes(sim, gtfs_dir, route_ids)



if __name__ == "__main__":
    main()
