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

import time
import datetime
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations, cycle
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
from scipy import spatial
import networkx as nx
import pandas as pd
import pyproj
from scipy.spatial.distance import cdist

import config_utils
from world import PtSystem
from world import street_network as network
from simulation.timeless_sim import TimelessSimulator, \
    get_gtfs_route_stops_and_freqs


LAVAL_CSD = 65
MTL_CSD = 66
# Therese-De Blainville
TDB_CSD = 73
# Deux-Montagnes
DEUXMONTAGNES_CSD = 72
# Les Moulins
LESMOULINS_CSD = 64


def parse_node_elem(node_elem):
    id = node_elem.get("id")
    xx = node_elem.get("x")
    yy = node_elem.get("y")
    return (id, xx, yy)


def parse_link_elem(link_elem):
    id = link_elem.get("id")
    from_id = link_elem.get("from")
    to_id = link_elem.get("to")
    speed = link_elem.get("freespeed")
    return (id, from_id, to_id, speed)


def plot_links(ax, node_dict, link_set, colour):
    print("Rendering", len(link_set), "lines...")
    lines = []
    for (_, from_id, to_id, _) in link_set:
        orig = node_dict[from_id]
        dest = node_dict[to_id]
        lines.append([orig, dest])
    line_collection = LineCollection(lines, colors=colour, linewidths=2)
    ax.add_collection(line_collection)
    ax.autoscale()
    plt.xticks([])
    plt.yticks([])


def compare_network(car_network_file, pt_network_file):
    car_net = config_utils.parse_xml(car_network_file)
    pt_net = config_utils.parse_xml(pt_network_file)

    car_nodes = {}
    for node_elem in car_net.iter("{*}node"):
        id, xx, yy = parse_node_elem(node_elem)
        car_nodes[id] = (xx, yy)

    count = 0
    pt_nodes = {}
    for node_elem in pt_net.iter("{*}node"):
        id, xx, yy = parse_node_elem(node_elem)
        pt_nodes[id] = (xx, yy)
        if not id in car_nodes:
            count += 1
            # print(node, "not present in car network!")
    print(count, "network nodes are not present in the car network!")

    car_links = set()
    transit_links = set()
    for link_elem in car_net.iter("{*}link"):
        link = parse_link_elem(link_elem)
        car_links.add(link)
        modes = link_elem.get("modes").split(",")
        if "bus" in modes or "pt" in modes:
            transit_links.add(link)

    count = 0
    inf_count = 0
    connects_to_missing_count = 0
    partials_count = 0
    # check for subsequent stops on a route have the same arr / dep time.
    pt_links = set()
    for link_elem in pt_net.iter("{*}link"):
        link = parse_link_elem(link_elem)
        if link not in car_links:
            pt_links.add(link)
            count += 1
            if link [3] == "Infinity":
                print("link", link[1], "has speed", link[3],
                      ", and is not in car network!")
                inf_count += 1

            if link[1] in car_nodes and link[2] in car_nodes:
                print("missing link", link[1],
                     "is connected to non-missing nodes!")
                connects_to_missing_count += 1
            if link[2] in car_nodes or link[2] in car_nodes:
                partials_count += 1

    print(count, "network links are not present in the car network!",
          inf_count, "of those have infinite speed.")
    print(connects_to_missing_count,
          "missing links connect only to non-missing nodes.")
    print(partials_count,
          "missing links connect to one non-missing node.")

    # plot each car link in one colour
    # plt.figure()
    _, ax = plt.subplots()
    plot_links(ax, car_nodes, car_links, "blue")
    plot_links(ax, pt_nodes, pt_links, "yellow")
    inf_links = [ll for ll in pt_links if ll[3] == "Infinity"]
    plot_links(ax, pt_nodes, inf_links, "red")
    ax.set_title("All links")

    _, ax = plt.subplots()
    plot_links(ax, car_nodes, transit_links, "blue")
    ax.set_title("Transit links")
    plt.show()


def find_sametime_deps(ts_file):
    ts_xml = config_utils.parse_xml(ts_file)
    for line_elem in ts_xml.iter("{*}transitLine"):
        line_id = line_elem.get('id')
        for route_elem in line_elem.iter("{*}transitRoute"):
            route_id = route_elem.get('id')
            prev_dep_time_s = 0
            for stop_elem in route_elem.iter("{*}stop"):
                arr_time_s, dep_time_s = None, None
                if "departureOffset" in stop_elem:
                    dep_time_s = stop_elem.get("departureOffset")
                    dep_time_s = config_utils.str_time_to_float(dep_time_s)
                if "arrivalOffset" in stop_elem:
                    arr_time_s = stop_elem.get("arrivalOffset")
                    arr_time_s = config_utils.str_time_to_float(arr_time_s)

                if arr_time_s is not None and arr_time_s - prev_dep_time_s <= 0:
                    print(line_id, route_id, stop_elem.get("refId"))

                prev_dep_time_s = dep_time_s

    # check that no vehicle has simultaneous deps.
    vehicles_deptimes = defaultdict(set)
    for dep_elem in ts_xml.iter("{*}departure"):
        vehicle = dep_elem.get("vehicleRefId")
        dep_time_s = config_utils.str_time_to_float(dep_elem.get("departureTime"))
        if dep_time_s in vehicles_deptimes[vehicle]:
            print("vehicle", vehicle, "departs more than once!",
                  dep_elem.get("id"))
        vehicles_deptimes[vehicle].add(dep_time_s)


def count_pt_accessible_nodes(network_path):
    # count nodes that have at least one pt or bus mode link connected
    links_data = network.get_links(network_path)
    for mode in ['bus', 'pt']:
        graph = nx.DiGraph()
        for link_data in links_data.values():
            if mode in link_data['modes']:
                graph.add_edge(link_data['source'], link_data['dest'])
        print('There are', len(graph), 'nodes accessible by', mode)

    # find out what all the link types in this network are
    non_bus_types = ['living_street', 'pedestrian', 'track', 'escape', 
        'raceway', 'footway', 'bridleway', 'steps', 'path', 'corridor', 
        'sidewalk', 'crossing', 'cycleway']
    highway_types = set()
    valid_nodes = set()
    
    stats = defaultdict(lambda: {'attr and mode': 0,
                                 'attr only': 0,
                                 'mode only': 0,
                                 'neither': 0})
    for link_data in links_data.values():
        try:
            attrs = link_data['attributes']
            hwy_val = attrs['osm:way:highway']
            if hwy_val in non_bus_types:
                if ('bus' in link_data['modes'] or 'pt' in link_data['modes']):
                    print('WOAH, a', hwy_val, 'link is', link_data['modes'], 
                          'accessible!')
            else:
                highway_types.add(hwy_val)
                valid_nodes.add(link_data['source'])
                valid_nodes.add(link_data['dest'])
            # if hwy_val in non_bus_types and ('bus' in link_data['modes'] or
            #     'pt' in link_data['modes']):
            #     print('WOAH, a', hwy_val, 'link is', link_data['modes'], 
            #         'accessible!')
            if hwy_val in ['unclassified', 'residential']:
                if 'osm:relation:route' in attrs and \
                    'bus' in attrs['osm:relation:route'].split(','):
                    if 'bus' in link_data['modes']:
                        stats[hwy_val]['attr and mode'] += 1
                    else:
                        stats[hwy_val]['attr only'] += 1
                else:
                    if 'bus' in link_data['modes']:
                        stats[hwy_val]['mode only'] += 1
                    # else:
                    #     stats[hwy_val]['neither'] += 1

        except KeyError:
            pass
    print('highway types are:')
    print(highway_types)
    print(len(valid_nodes), 'valid highway nodes')

    for key, val in stats.items():
        total = sum(val.values())
        msg = key + ' has '
        for outcome, count in val.items():
            msg += '{} ({}) {}, '.format(count, count / total, outcome)
        print(msg)

    # count links that ought to be accessible to buses based on type
    graph = nx.DiGraph()
    for link_data in links_data.values():
        try:
            hwy_val = link_data['attributes']['osm:way:highway']
        except KeyError:
            hwy_val = None
        if hwy_val not in non_bus_types:
            graph.add_edge(link_data['source'], link_data['dest'])
    print('There are', len(graph), 'theoretically bussable nodes')


def show_highway_tag(network_path, highway_tag):
    # start with all edges
    graph = network.build_graph(network_path)
    poss = {nn: (dd["data"]["xpos"], dd["data"]["ypos"])
            for nn, dd in graph.nodes(data=True)}
    nx.draw(graph, pos=poss, node_size=2, node_color='black', 
            edge_color='black')

    # on top of that, draw all edges of the desired highway type
    hwyedges = [(aa, bb, ii) for aa, bb, ii in graph.edges(keys=True)
                if graph.edges[aa, bb, ii].get('osm:way:highway') == 
                    highway_tag]
    hwygraph = graph.edge_subgraph(hwyedges)
    nx.draw_networkx_edges(hwygraph, pos=poss, width=4, edge_color='blue')

    # then draw just those that have the 'bus' mode.  Give these labels
    # so we can make changes if need be.
    busmodeedges = [(aa, bb, ii) for aa, bb, ii in hwygraph.edges(keys=True)
                    if 'bus' in hwygraph.edges[aa, bb, ii]['modes']]
    busmodegraph = hwygraph.edge_subgraph(busmodeedges)
    nx.draw(busmodegraph, pos=poss, node_size=0, width=4, edge_color='orange', 
            with_labels=True)

    hasbusrel = lambda dd: 'osm:relation:route' in dd and \
                            'bus' in dd['osm:relation:route'].split(',')
    busreledges = [(aa, bb, ii) for aa, bb, ii in hwygraph.edges(keys=True)
                   if hasbusrel(hwygraph.edges[aa, bb, ii])]
    busrelgraph = hwygraph.edge_subgraph(busreledges)
    nx.draw(busrelgraph, pos=poss, node_size=0, width=2, edge_color="pink",
            with_labels=True)
    plt.show()


def time_routing(ts_path, tv_path):
    ptsys = PtSystem.from_xml(ts_path, tv_path)
    print("done parsing ptsystem")
    # add nodes
    stop_locs = []
    stop_ids = []
    for stop in ptsys.get_stop_facilities():
        stop_ids.append(stop.id)
        stop_locs.append(stop.position)

    # add walking links
    walk_speed = 1.35 * 1.3
    walk_dists = spatial.distance_matrix(stop_locs, stop_locs)
    walk_costs = walk_dists / walk_speed + 10
    walk_costs[walk_dists >= 100] = 0
    routing_graph = nx.from_numpy_array(walk_costs,
                                        create_using=nx.MultiDiGraph())
    print(time.perf_counter(),
          "created {} walking links".format(len(routing_graph.edges)))

    # add route links
    print('Laval has', len(ptsys.get_routes()), 'transit routes.')
    print('Laval has', len(ptsys.get_stop_facilities()), 'stops.')
    route_lens = [len(rr.stops) for rr in ptsys.get_routes()]
    print('# stops on routes - min: {} max: {} mean: {} median: {}' \
        'total: {}'.format(
        min(route_lens), max(route_lens), np.mean(route_lens), 
        np.median(route_lens), sum(route_lens)
    ))

    inter_stop_dists = []
    for route in ptsys.get_routes():
        # compute inter-stop times
        leg_costs = []
        last_time = 0
        last_stop = route.stops[0]
        for stop in route.stops[1:]:
            leg_costs.append(stop.arrival_offset_s - last_time)
            last_time = stop.arrival_offset_s
            dist = np.linalg.norm(stop.facility.position - 
                last_stop.facility.position)
            inter_stop_dists.append(dist)
            last_stop = stop

        # add links for each pair of stops
        for (ii, si), (jj, sj) in combinations(enumerate(route.stops), 2):
            cost = sum(leg_costs[ii:jj])
            si_idx = stop_ids.index(si.facility.id)
            sj_idx = stop_ids.index(sj.facility.id)
            routing_graph.add_edge(si_idx, sj_idx, weight=cost+1)

    print('inter-stop dists - min: {} max: {} mean: {} median: {}' \
        'total: {}'.format(
        min(inter_stop_dists), max(inter_stop_dists), np.mean(inter_stop_dists), 
        np.median(inter_stop_dists), sum(inter_stop_dists)
    ))
    # time it!
    print(time.perf_counter(),
         "Routing network is ready with {} nodes and {} edges!".format(
         len(routing_graph.nodes), len(routing_graph.edges)
    ))
    print('greatest out-degree:', 
        max(routing_graph.out_degree, key=lambda x: x[1]))
    starttime = time.perf_counter()
    paths = nx.shortest_path(routing_graph)
    endtime = time.perf_counter()
    print(time.perf_counter(), "Total routing time:", endtime - starttime)


def draw_network_and_demands(network_path, demand_csv):
    ptgraph = network.build_graph(network_path)
    poss = {nn: (dd["data"]["xpos"], dd["data"]["ypos"])
            for nn, dd in ptgraph.nodes(data=True)}
    
    nx.draw(ptgraph, pos=poss, node_size=2, node_color='black', 
            edge_color='black')

    demands_df = pd.read_csv(demand_csv)
    demand_graph = nx.DiGraph()
    demand_poss = {}
    for _, row in demands_df.iterrows():
        orig = (row['t_orix'], row['t_oriy'])
        demand_poss[orig] = orig
        dest = (row['t_desx'], row['t_desy'])
        demand_poss[dest] = dest
        demand_graph.add_edge(orig, dest)
    nx.draw_networkx_nodes(demand_graph, demand_poss, node_color="blue", 
                           node_size=1)
    # nx.draw_networkx_edges(demand_graph, demand_poss, edge_color="blue")

    plt.show()


def parse_census_code(census_code):
    census_code = int(census_code)
    province_code = census_code // 100000
    div_code = (census_code // 1000) % 100
    subdiv_code = census_code % 1000
    return province_code, div_code, subdiv_code


def count_demand_internal_vs_external(demand_csv):
    demands_df = pd.read_csv(demand_csv)
    internal_count = 0
    from_ext_count = 0
    to_ext_count = 0
    from_codes = defaultdict(float)
    to_codes = defaultdict(float)
    for _, row in demands_df.iterrows():
        modes = (row["t_mode1"], row["t_mode2"], row["t_mode3"],
                 row["t_mode4"], row["t_mode5"], row["t_mode6"],
                 row["t_mode7"], row["t_mode8"], row["t_mode9"])
        if 1 in modes or 2 in modes:
            # ignore car trips
            continue
        ori_div_code = parse_census_code(row['t_oricsd'])[1]
        from_codes[ori_div_code] += row['t_expf']
        des_div_code = parse_census_code(row['t_descsd'])[1]
        to_codes[des_div_code] += row['t_expf']
        if ori_div_code != LAVAL_CSD and des_div_code == LAVAL_CSD:
            from_ext_count += row['t_expf']
        elif ori_div_code == LAVAL_CSD and des_div_code != LAVAL_CSD:
            to_ext_count += row['t_expf']
        elif ori_div_code == LAVAL_CSD and des_div_code == LAVAL_CSD:
            internal_count += row['t_expf']
    total = internal_count + from_ext_count + to_ext_count
    print("total demand:", total)
    print("internal demand:", internal_count, "(", internal_count / total, ")")
    print("from ext demand:", from_ext_count, "(", from_ext_count / total, ")")
    print("to ext demand:", to_ext_count, "(", to_ext_count / total, ")")

    for dd in (from_codes, to_codes):
        plt.figure(1)
        plt.bar(range(len(dd)), list(dd.values()), align='center')
        plt.xticks(range(len(dd)), list(dd.keys()))
        plt.figure(2)
        plt.pie(list(dd.values()), labels=list(dd.keys()))
        plt.show()


def draw_network(network_path):
    ptgraph = network.build_graph(network_path)
    poss = {nn: (dd["data"]["xpos"], dd["data"]["ypos"])
            for nn, dd in ptgraph.nodes(data=True)}
    
    nx.draw(ptgraph, pos=poss, node_size=2, node_color='black', 
            edge_color='black')
    # busgraph = network.build_graph(network_path, ["bus"])
    # poss = {nn: (dd["data"]["xpos"], dd["data"]["ypos"])
    #         for nn, dd in busgraph.nodes(data=True)}
    # nx.draw(busgraph, pos=poss, node_size=2, node_color='blue', 
    #         edge_color='blue', with_labels=True)
    
    # plt.show()

    # ptcomponents = nx.strongly_connected_components(ptgraph)
    # ptcomponents = list(sorted(ptcomponents, key=len))[:-1]
    # print("num. components:", len(list(ptcomponents)))
    # poss = {nn: (dd["data"]["xpos"], dd["data"]["ypos"])
    #         for nn, dd in ptgraph.nodes(data=True)}
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colours = cycle(prop_cycle.by_key()['color'])
    # for cc in ptcomponents:
    #     colour = next(colours)
    #     subgraph = ptgraph.subgraph(cc).copy()
    #     nx.draw(subgraph, pos=poss, node_color=colour, edge_color=colour,
    #             node_size=20, with_labels=True)
    # plt.show()

    # adjmat = nx.to_numpy_array(graph)
    sim = TimelessSimulator("simulation/laval_cfg.yaml")
    poss = {nn: (sim.stop_positions[nn])
            for nn, _ in sim.street_graph.nodes(data=True)}
    # nx.draw(sim.street_graph, pos=poss, node_size=2, node_color='red', 
    #         edge_color='red')

    simcomps = list(nx.strongly_connected_components(sim.street_graph))
    simcomps = sorted(simcomps, key=len, reverse=True)

    print("sim's num components:", len(simcomps))

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colours = cycle(prop_cycle.by_key()['color'])
    with_labels = False
    for cc in simcomps:
        colour = next(colours)
        subgraph = sim.street_graph.subgraph(cc).copy()
        print(len(cc))
        nx.draw(subgraph, pos=poss, node_color=colour, edge_color=colour,
                node_size=20, with_labels=with_labels)
        with_labels = True
    plt.show()
    env_rep = sim.get_env_rep_for_nn()


def check_gtfs_routes(gtfs_dir, start_time=None, end_time=None, 
                      service_id="AOUT13SEM"):
    gtfs_dir = Path(gtfs_dir)

    # we need routes, trips, stops, and stop_times
    trips_df = pd.read_csv(gtfs_dir / "trips.txt")
    trips_df = trips_df[trips_df["service_id"] == service_id]
    stop_times_df = pd.read_csv(gtfs_dir / "stop_times.txt")
    deptimes = pd.to_datetime(stop_times_df["departure_time"], errors='coerce')
    arrtimes = pd.to_datetime(stop_times_df["arrival_time"], errors='coerce')
    times_are_valid = ~(deptimes.isnull() | arrtimes.isnull())
    stop_times_df = stop_times_df[times_are_valid]

    get_gtfs_route_stops_and_freqs(trips_df, stop_times_df, start_time, end_time)


def filter_demand_by_time(demands_df, start_time=None, end_time=None):
    if not start_time and not end_time:
        # no filtering to be done.
        return demands_df

    if start_time:
        start_time_s = config_utils.str_time_to_float(start_time)
    else:
        start_time_s = 0
    if end_time:
        end_time_s = config_utils.str_time_to_float(end_time)
    else:
        end_time_s = 24 * 3600

    out_rows = []
    for _, row in demands_df.iterrows():
        trip_time = config_utils.od_fmt_time_to_float(row["t_time"])
        if trip_time >= start_time_s and trip_time <= end_time_s:
            out_rows.append(row)
    
    return pd.DataFrame(out_rows)


def filter_demand_by_zone(demands_df, zone_ids=[LAVAL_CSD]):
    """By default, only keep demand that is entirely within Laval."""
    if not zone_ids:
        return demands_df

    out_rows = []
    for _, row in demands_df.iterrows():
        orig = parse_census_code(row['t_oricsd'])[1]
        dest = parse_census_code(row['t_descsd'])[1]
        if orig in zone_ids and dest in zone_ids:
            out_rows.append(row)
    
    return pd.DataFrame(out_rows)


def get_interzone_demand_stats(demand_csv, start_time=None, end_time=None):
    demands_df = pd.read_csv(demand_csv)
    demands_df = filter_demand_by_time(demands_df, start_time, end_time)
    demands_df = filter_demand_by_zone(demands_df)

    # assemble all locations
    locs = set()
    for _, row in demands_df.iterrows():
        orig = (row['t_orix'], row['t_oriy'])
        dest = (row['t_desx'], row['t_desy'])
        locs.add(orig)
        locs.add(dest)

    # convert to a dict of indices
    locs_to_idx = {loc: ii for ii, loc in enumerate(locs)}
    demand_counts = np.zeros((len(locs), len(locs)))

    # assemble statistics
    for _, row in demands_df.iterrows():
        orig = (row['t_orix'], row['t_oriy'])
        orig_idx = locs_to_idx[orig]
        dest = (row['t_desx'], row['t_desy'])
        dest_idx = locs_to_idx[dest]
        demand_counts[orig_idx, dest_idx] += row["t_expf"]

    # print mean, standard deviation, etc.
    print(f"There are {len(locs)} demand locations. Of all pairs of these, "\
          f"{(demand_counts > 0).sum()} have non-zero demand.")
    print("interzone demand stats")
    print(f"mean: {demand_counts.mean()}")
    print(f"std: {demand_counts.std()}")
    print(f"min: {demand_counts.min()}")
    print(f"max: {demand_counts.max()}")

    # plot a histogram
    plt.hist(demand_counts.flatten(), bins=100)
    plt.title("Interzone demand counts")
    plt.yscale('log')
    plt.show()


def count_laval_busriders(demand_csv, start_time=None, end_time=None):
    demands_df = pd.read_csv(demand_csv)
    demands_df = filter_demand_by_time(demands_df, start_time, end_time)

    bus_trips = 0
    parknride_trips = 0
    bus_row_count = 0
    stl_mode_code = 6
    car_codes = [1, 2]
    for _, row in demands_df.iterrows():
        modes = (row["t_mode1"], row["t_mode2"], row["t_mode3"],
                 row["t_mode4"], row["t_mode5"], row["t_mode6"],
                 row["t_mode7"], row["t_mode8"], row["t_mode9"])
        if stl_mode_code in modes:
            bus_trips += row['t_expf']
            bus_row_count += 1
            if any(cc in modes for cc in car_codes):
                parknride_trips += row['t_expf']
        
    print(f"OD data indicates {bus_trips} trips were made on Laval buses, \
        over {bus_row_count} rows")
    print(f"of those, {parknride_trips} were park-and-rides.")


def transit_distance_statistics(demand_csv, gtfs_path, 
                                start_time=None, end_time=None, 
                                mapframe="EPSG:3348"):
    stops_path = Path(gtfs_path) / "stops.txt"
    stops_df = pd.read_csv(stops_path)
    stop_locs = []
    transformer = pyproj.Transformer.from_crs("EPSG:4326", mapframe)
    for _, row in stops_df.iterrows():
        loc = (row["stop_lat"], row["stop_lon"])
        stop_locs.append(transformer.transform(*loc))

    demands_df = pd.read_csv(demand_csv)
    demands_df = filter_demand_by_time(demands_df, start_time, end_time)

    start_locs = []    
    end_locs = []
    trip_dists = []
    stl_mode_code = 6
    total_trips = 0
    mode_counts = defaultdict(float)
    for _, row in demands_df.iterrows():
        total_trips += row["t_expf"]
        modes = (row["t_mode1"], row["t_mode2"], row["t_mode3"],
                 row["t_mode4"], row["t_mode5"], row["t_mode6"],
                 row["t_mode7"], row["t_mode8"], row["t_mode9"])
        if stl_mode_code in modes:
            start_loc = (row["t_orix"], row["t_oriy"])
            end_loc = (row["t_desx"], row["t_desy"])
            for mode_code in set(modes):
                # this check avoids nans
                if mode_code == mode_code:
                    mode_counts[mode_code] += row["t_expf"]
            for _ in np.arange(row["t_expf"]):
                start_locs.append(start_loc)
                end_locs.append(end_loc)

    print("total trips in period:", total_trips)

    # show counts of different modes
    x_inds = np.arange(len(mode_counts))
    labels, heights = zip(*mode_counts.items())
    plt.bar(x_inds, heights)
    plt.xticks(x_inds, labels)
    plt.show()
    
    # show histogram of distances between ori/dest and first/last stop 
    start_locs = np.array(start_locs)
    end_locs = np.array(end_locs)
    start_dists = cdist(stop_locs, start_locs).min(axis=0)
    end_dists = cdist(stop_locs, end_locs).min(axis=0)
    start_pctls = np.percentile(start_dists, [5, 75, 85])
    print("start dist percentiles:", start_pctls)
    end_pctls = np.percentile(end_dists, [50, 75, 85])
    print("end dist percentiles:", end_pctls)
    plt.hist(start_dists, label="start distances", bins=20, histtype='step')
    plt.hist(end_dists, label="end distances", bins=20, histtype='step')
    plt.legend()
    plt.show()

    # plot walking distances against trip distances
    trip_dists = np.linalg.norm(start_locs - end_locs, axis=1)
    plt.scatter(trip_dists, start_dists, label="starts")
    plt.scatter(trip_dists, end_dists, label="ends")
    plt.legend()
    plt.show()


def draw_gtfs_routes_on_network(network_path, gtfs_path, mapframe="EPSG:3348"):
    # draw network
    ptgraph = network.build_graph(network_path)
    poss = {nn: (dd["data"]["xpos"], dd["data"]["ypos"])
            for nn, dd in ptgraph.nodes(data=True)}
    
    nx.draw(ptgraph, pos=poss, node_size=2, node_color='black', 
            edge_color='black', arrows=False)

    # draw routes with real stops
    stops_path = Path(gtfs_path) / "stops.txt"
    stops_df = pd.read_csv(stops_path)
    stop_locs = {}
    transformer = pyproj.Transformer.from_crs("EPSG:4326", mapframe)
    for _, row in stops_df.iterrows():
        loc = (row["stop_lat"], row["stop_lon"])
        stop_locs[row["stop_id"]] = transformer.transform(*loc)
    
    trips_df = pd.read_csv(gtfs_path / "trips.txt")
    stop_times_df = pd.read_csv(gtfs_path / "stop_times.txt")
    # filter out stops with invalid times (< 00:00:00 or >= 24:00:00)
    deptimes = pd.to_datetime(stop_times_df["departure_time"], errors='coerce')
    arrtimes = pd.to_datetime(stop_times_df["arrival_time"], errors='coerce')
    times_are_valid = ~(deptimes.isnull() | arrtimes.isnull())
    stop_times_df = stop_times_df[times_are_valid]

    start_time = datetime.time(hour=7)
    end_time = datetime.time(hour=9)
    stops_on_routes, _ = \
        get_gtfs_route_stops_and_freqs(trips_df, stop_times_df, start_time, 
                                  end_time, "AOUT13SEM")

    # translate routes from lists of stop_ids to lists of stop locations
    labelled_routes = set()
    routes_df = pd.read_csv(gtfs_path / "routes.txt")
    for _, row in routes_df.iterrows():
        route_id = row["route_id"]
        if route_id not in stops_on_routes:
            continue
        route_graph = nx.DiGraph()
        prev_id = None
        for stop_id in stops_on_routes[route_id]:
            route_graph.add_node(stop_id)
            if prev_id is not None:
                route_graph.add_edge(prev_id, stop_id)
            prev_id = stop_id
        colour = "#" + row["route_color"]
        edge_col = nx.draw_networkx_edges(route_graph, pos=stop_locs, 
                                          edge_color=colour,
                                          width=3, node_size=0)
        # name = row["route_short_name"]
        # if name not in labelled_routes:
        edge_col[0].set_label(row["route_short_name"])
            # labelled_routes.add(name)

    plt.legend()
    plt.show()

if __name__ == "__main__":
    # carnet = "/home/andrew/matsim/my_envs/laval/generate/final_car_network.xml"
    # ptnet = "/home/andrew/matsim/my_envs/laval/network.xml"
    # compare_network(carnet, ptnet)
    # ts = "/home/andrew/matsim/my_envs/laval/generate/final_ts.xml"
    # find_sametime_deps(ts)
    # sim_cfg_path = "/localdata/ahollid/transit_learning/simulation/laval_cfg.yaml"
    network_path = Path("/usr/local/data/ahollid/laval/network.xml")
    demand_path = Path("/usr/local/data/ahollid/laval/od_2013.csv")
    get_interzone_demand_stats(demand_path, "07:00:00", "09:00:00")

    draw_network_and_demands(network_path, demand_path)

    gtfs_path = Path("/localdata/ahollid/laval/gtfs_nov2013")
    draw_gtfs_routes_on_network(network_path, gtfs_path)

    transit_distance_statistics(demand_path, gtfs_path, 
                                "07:00:00", "09:00:00")
    # count_laval_busriders(demand_path, "07:00:00", "09:00:00")
    check_gtfs_routes(gtfs_path, datetime.time(7), datetime.time(9))
    # check_gtfs_routes(gtfs_path, datetime.time(0), datetime.time(23,59,59))

    count_demand_internal_vs_external(demand_path)
    # draw_network(network_path)
    # show_highway_tag(network_path, "service")
    # network_path = "/usr/local/data/ahollid/laval/network.xml"
    # count_pt_accessible_nodes(network_path)

    # ts = "/home/andrew/matsim/my_envs/laval/ts.xml"
    # tv = "/home/andrew/matsim/my_envs/laval/tv.xml"
    # time_routing(ts, tv)
