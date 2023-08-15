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

import subprocess
import shlex
import numpy as np
from pathlib import Path
import os
from collections import defaultdict
from itertools import product
from dataclasses import dataclass, field
from typing import List, Tuple
from copy import deepcopy
from scipy.spatial import KDTree
import pandas as pd
import logging

from tqdm import tqdm

from world.transit import *


DEFAULT_BASIN_M = 500
DEFAULT_CHANGE_TIME_S = 60
DEFAULT_DELAY_TOLERANCE_S = 1800
TRANSFER_RADIUS = 100


@dataclass
class PtJourneyLeg:
    line_id: str
    route_id: str
    board_stop_id: str
    disembark_stop_id: str
    board_time_s: float = None
    disembark_time_s: float = None
    vehicle_id: str = None


@dataclass
class PassengerTrip:
    id: str
    origin: Tuple[float, float]
    destination: Tuple[float, float]
    start_time_s: float
    planned_journey: List[PtJourneyLeg] = field(default_factory=list)
    real_journey: List[PtJourneyLeg] = field(default_factory=list)
    leg_idx: int = 0
    end_time_s: float = None

    def board(self, time_s, vehicle_id):
        real_leg = deepcopy(planned_journey[leg_idx])
        real_leg.board_time_s = time_s
        real_leg.vehicle_id = vehicle_id
        real_journey.append(real_leg)

    def disembark(self, time_s):
        real_journey[leg_idx].disembark_time_s = time_s
        self.leg_idx += 1

    def is_finished(self):
        return self.leg_idx == len(self.planned_journey)


class AbstractSim:
    def __init__(self, env_dir, config_file='config.xml'):
        if type(self) is AbstractSim:
            raise ValueError(
                "This class is not meant to be instantiated, only subclassed!")

        self.environment_dir = Path(env_dir)
        self.config_path = self.environment_dir / config_file
        # initialize from config files
        self._config_tree = config_utils.parse_xml(self.config_path)

        if (self.environment_dir / 'setup.sh').exists():
            # run the setup script
            print('running sim environment setup script...')
            orig_wd = os.getcwd()
            os.chdir(self.environment_dir)
            cmd = 'bash setup.sh'
            subprocess.run(shlex.split(cmd),
                           capture_output=True, check=True)
            os.chdir(orig_wd)

        logging.info('reading transit from xml')
        pt_system = PtSystem.from_xml(self._get_pt_schedule_path(),
                                      self._get_pt_vehicles_path())
        logging.info('done reading transit')
        # we set it directly rather than calling set_transit(), because sub-
        # classes may override that method to do things that are dependent on
        # the object already being initialized.
        self._transit = pt_system

    def run(self):
        raise NotImplementedError("Subclasses must implement this method!")

    def get_pt_passenger_traces(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method!")

    def get_pt_vehicle_traces(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method!")

    def get_pt_trips(self, **kwargs):
        raise NotImplementedError("Subclasses must implement this method!")

    def get_start_time_s(self):
        time_str = self._get_param('qsim', 'startTime')
        return config_utils.str_time_to_float(time_str)

    def get_end_time_s(self):
        time_str = self._get_param('qsim', 'endTime')
        return config_utils.str_time_to_float(time_str)

    def get_walk_time(self, origin, destination):
        beeline_dist = np.linalg.norm(np.array(origin) - np.array(destination))
        return self.get_dist_walk_time(beeline_dist)

    def get_dist_walk_time(self, distance_m):
        """distance_m can be a scalar or numpy array."""
        beeline_factor = float(self._get_param('planscalcroute',
                                               'beelineDistanceFactor',
                                               'teleportedModeParameters',
                                               default=1.3))
        walk_speed_mps = float(self._get_param('planscalcroute',
                                               'teleportedModeSpeed',
                                               'teleportedModeParameters',
                                               default=4.167))
        return distance_m * beeline_factor / walk_speed_mps

    def get_transit(self):
        return self._transit

    def set_transit(self, pt_system):
        self._transit = pt_system

    # private methods

    def _get_network_path(self):
        return self._get_param_as_path('network', 'inputNetworkFile')

    def _get_pt_schedule_path(self):
        return self._get_param_as_path('transit', 'transitScheduleFile')

    def _get_pt_vehicles_path(self):
        return self._get_param_as_path('transit', 'vehiclesFile')

    def _get_param_as_path(self, *args, **kwargs):
        """Utility function for retrieving paths from config.xml.

        Takes the same arguments as _get_param()"""
        path = Path(self._get_param(*args, **kwargs))
        if path.is_absolute():
            return path
        else:
            return self.environment_dir / path

    def _get_param(self, module_name, param_name, paramset_name=None,
                   default=None):
        target_module_elem = None
        for module_elem in self._config_tree.iter('{*}module'):
            if module_elem.get('name') == module_name:
                target_module_elem = module_elem
                break

        if target_module_elem is None:
            return default
            # err = 'Module {} not found in sim env config!'.format(module_name
            # raise ValueError(err)

        target_paramset_elem = None
        if paramset_name:
            for paramset_elem in target_module_elem.iter('{*}parameterset'):
                if paramset_elem.get('type') == paramset_name:
                    target_paramset_elem = paramset_elem
                    break
        else:
            target_paramset_elem = module_elem
        for param in target_paramset_elem.iter('{*}param'):
            if param.get('name') == param_name:
                return param.get('value')

        # if we get here, we failed to find the requested parameter
        return default
        # or alternatively...raise an error?
        # err = 'Param {pp} not found in module {mm}'
        # if paramset_name:
        #     err += ', paramset {ps}'
        # err += '!'
        # err = err.format(pp=param_name, mm=module_name, ps=paramset_name)
        # raise ValueError('')


class AbstractTransitOnlySim(AbstractSim):
    def __init__(self, od_data, basin_radius_m=DEFAULT_BASIN_M,
                 delay_tolerance_s=DEFAULT_DELAY_TOLERANCE_S,
                 change_time_s=DEFAULT_CHANGE_TIME_S, max_transfers=None,
                 **kwargs):
        super().__init__(**kwargs)
        # TODO justify the values of these parameters!
        # only people living this close to a stop will consider taking transit
        self.basin_radius_m = basin_radius_m
        # people aren't willing to delay starting a trip to accomodate transit
        # by more than this amount of time.
        self.delay_tolerance_s = delay_tolerance_s
        # the average time it takes to change buses at one stop
        self.change_time_s = change_time_s
        # increment the maximum number of transfers.
        if max_transfers is None:
            self.max_transfers = float('inf')
        else:
            self.max_transfers = max_transfers
        logging.info('reading trips from OD...')
        self.trips_from_od_data(od_data)
        logging.info('Finished reading trips from OD.')
        self.assign_trips()

    def set_transit(self, transit):
        super().set_transit(transit)
        """Subclasses must implement this.  The end product should be,
        for each scheduled stop of each bus, a collection of passengers who
        attempt to start their journeys by boarding that bus at that stop, and
        the bus stopping there.
        """
        self.assign_trips()

    def trips_from_od_data(self, od_data):
        raise NotImplementedError("Subclasses must implement this!")

    def get_accessible_stops_by_trip(self, orig_or_dest):
        """
        orig_or_dest: a string, either "orig" or "dest" indicating whether to
        compute distances from trip origins or destinations.

        Returns:
        {trip ID: {set of all stops within range of trip's end point}}"""
        leafsize = max(len(self.trips) // 40, 1)
        if orig_or_dest == 'orig':
            trip_endpoints = [tt.origin for tt in self.trips.values()]
        elif orig_or_dest == 'dest':
            trip_endpoints = [tt.destination for tt in self.trips.values()]
        kdtree = KDTree(trip_endpoints, leafsize)
        all_stops = self._transit.get_stop_facilities()
        stop_locs = [ss.position for ss in all_stops]
        result = kdtree.query_ball_point(stop_locs, self.basin_radius_m)
        flipped_result = defaultdict(set)
        trips_list = list(self.trips.values())
        for stop_idx, trips_near_stop in enumerate(result):
            # now unflip the results
            for trip_idx in trips_near_stop:
                trip_id = trips_list[trip_idx].id
                flipped_result[trip_id].add(all_stops[stop_idx].id)
        return flipped_result

    def assign_trips(self):
        """
        Plans the shortest transit journey from the origin to the destination,
        starting from the given time.

        Implements the pareto-Connection Scan Algorithm of Dibbelt et al.
        (2013), modified to allow multiple start and end points, and to allow
        thresholding the allowed number of transfers.
        """
        # this is where the magic happens!

        logging.info('pcsa preproc starting')
        stop_profiles = self._pcsa_preprocessing()
        logging.info('pcsa preproc done')
        self._assign_trips_from_profiles(stop_profiles)
        logging.info('pcsa trip assignment done')

    def _pcsa_preprocessing(self):
        # Step 1: initialize collections needed to search for routes in the
        # network:
        # the connections table...
        connections = []
        for route in self._transit.get_routes():
            for dep in route.departures:
                for si in range(len(route.stops) - 1):
                    dep_stop = route.stops[si]
                    dep_time = dep.start_time_s + dep_stop.departure_offset_s
                    arr_stop = route.stops[si+1]
                    arr_time = dep.start_time_s + arr_stop.arrival_offset_s

                    conn = {'dep stop': dep_stop.facility,
                            'dep time': dep_time,
                            'arr stop': arr_stop.facility,
                            'arr time': arr_time,
                            'route': route,
                            'departure': dep
                            }
                    connections.append(conn)
        connections = sorted(connections, key=lambda cc: cc['dep time'],
                             reverse=True)
        logging.info('there are ' + str(len(connections)) + ' connections')

        # ...and the foot-travel times between walkable stop pairs
        foottimes = defaultdict(list)
        # keep walk times between the stop and itself as 0
        # in_range_mat = self._transit.stop_dist_matrix <= self.basin_radius_m
        in_range_mat = self._transit.stop_dist_matrix <= TRANSFER_RADIUS
        walktimes_mat = self.get_dist_walk_time(self._transit.stop_dist_matrix)
        all_stop_facs = self._transit.get_stop_facilities()
        for ii, stop in enumerate(all_stop_facs):
            in_range_js = np.where(in_range_mat[ii])[0]
            in_range_times = walktimes_mat[ii, in_range_js]
            in_range_ids = [all_stop_facs[jj].id for jj in in_range_js]
            ii_foottimes = list(zip(in_range_ids, in_range_times))
            # sort the foottimes so we consider the stop itself first
            foottimes[stop.id] = sorted(ii_foottimes, key=lambda x: x[1])

        # step 2: build the profiles for each stop
        stop_profiles = defaultdict(list)
        # profile entry = (t_dep, t*, c_this, p_next, next_profile_index,
        #                  num_transfers)
        for dest_stop in tqdm(all_stop_facs):
            # TODO incorporate transfer time!!!
            for cc in connections:
                if cc['dep stop'] == dest_stop:
                    # ignore connections leaving the destination
                    continue
                # consider all stops in walking distance of arr, including arr
                # itself
                new_entry = None
                for csid, walk_time in foottimes[cc['arr stop'].id]:
                    if csid != dest_stop.id:
                        cand_stop = self._transit.get_stop_facility(csid)
                        arr_prof = stop_profiles[(csid, dest_stop.id)]

                        # find the best journey starting from candidate stop
                        valid_journey_found = False
                        for ei in reversed(range(len(arr_prof))):
                            arr_t_dep, arr_t_dest, next_cc = arr_prof[ei][:3]
                            num_transfers = arr_prof[ei][-1]
                            if next_cc['route'] == cc['route']:
                                # this is continuing on the same route
                                change_time_s = 0
                            else:
                                # this is a transfer!
                                num_transfers += 1
                                if num_transfers > self.max_transfers:
                                    continue
                                if csid == cc['arr stop'].id:
                                    # this is a transfer at the same stop
                                    change_time_s = self.change_time_s
                                else:
                                    # this is a transfer between stops
                                    change_time_s = walk_time

                            if arr_t_dep >= cc['arr time'] + change_time_s:
                                valid_journey_found = True
                                break
                    else:
                        # the destination is in walking distance of the arrival
                        # stop (or it *is* the arrival stop).
                        valid_journey_found = True
                        arr_t_dest = cc['arr time'] + walk_time
                        num_transfers = 0
                        ei = None

                    if valid_journey_found and (new_entry is None or
                                                arr_t_dest < new_entry[1]):
                        # we found a new best one!
                        new_entry = (cc['dep time'], arr_t_dest, cc, csid, ei,
                                     num_transfers)

                dep_prof = stop_profiles[(cc['dep stop'].id, dest_stop.id)]
                if new_entry and (len(dep_prof) == 0 or
                                  dep_prof[-1][1] > new_entry[1] or
                                  dep_prof[-1][0] < new_entry[0]):
                    # latest does not dominate new (or the profile is empty),
                    # so add new
                    dep_prof.append(new_entry)
        return stop_profiles

    def _assign_trips_from_profiles(self, stop_profiles):
        # Ceder 2011 (Transit network design methodology...) uses an
        # "exponential perception of distance" to allocate demand from
        # "transit centers" to nearby stops (eqn 5). Could this be a good model
        # to use to randomize ridership at a stop?  I should see if there is
        # some research on whether that is a realistic model of rider
        # behaviour.

        # Step 2: filter out trips that don't start and end in range of a stop
        orig_stop_ids_by_trip = self.get_accessible_stops_by_trip('orig')
        orig_trip_ids = set(orig_stop_ids_by_trip.keys())
        logging.info('valid origins found')

        dest_stop_ids_by_trip = self.get_accessible_stops_by_trip('dest')
        dest_trip_ids = set(dest_stop_ids_by_trip.keys())
        logging.info('valid destinations found')
        # the set of all trips with origs and dests in range of transit!
        tracked_trip_ids = orig_trip_ids.intersection(dest_trip_ids)

        # apply some randomness to the trips?
        # maybe randomly ignore it, depending on fraction of people we expect
            # to use transit?

        # Step 3: for each trip, compute the best route
        logging.info('{}/{} trips within range'.format(
            len(tracked_trip_ids), len(self.trips)))
        logging.info('routing passengers!')
        for trip_id in tqdm(tracked_trip_ids):
            trip = self.trips[trip_id]
            trip.real_journey = []
            trip.planned_journey = []

            orig_stops = [self._transit.get_stop_facility(sid)
                          for sid in orig_stop_ids_by_trip[trip_id]]
            dest_stops = [self._transit.get_stop_facility(sid)
                          for sid in dest_stop_ids_by_trip[trip_id]]
            orig_walk_times = {origs.id: self.get_walk_time(trip.origin,
                                                         origs.position)
                               for origs in orig_stops}
            dest_walk_times = {ds.id: self.get_walk_time(trip.destination,
                                                         ds.position)
                               for ds in dest_stops}

            # find the best (orig, dest) stop pair of all those in range of
            # the trip's origin and destination
            best_time = trip.start_time_s + \
                self.get_walk_time(trip.origin, trip.destination)
            best_od_stops = None
            first_entry_idx = None
            for orig_stop, dest_stop in product(orig_stops, dest_stops):
                # get index of first profile entry
                od_first_entry_idx = None
                at_orig_stop_time = trip.start_time_s + \
                    orig_walk_times[orig_stop.id]
                profile = stop_profiles[orig_stop.id, dest_stop.id]
                for ei in reversed(range(len(profile))):
                    entry = profile[ei]
                    if entry[0] - trip.start_time_s > self.delay_tolerance_s:
                        # starts too late!
                        break
                    if entry[0] >= at_orig_stop_time:
                        od_first_entry_idx = ei
                        break

                if od_first_entry_idx is None:
                    # no valid journeys starting from this orig stop
                    continue

                if entry[1] + dest_walk_times[dest_stop.id] < best_time:
                    # best we've found so far!
                    best_time = entry[1] + dest_walk_times[dest_stop.id]
                    best_od_stops = (orig_stop, dest_stop)
                    first_entry_idx = od_first_entry_idx

            if best_od_stops is None:
                # no valid journey could be found for this trip
                continue

            # construct the journey for the trip
            orig_stop, dest_stop = best_od_stops
            next_stop_id = orig_stop.id
            route_conns = []
            best_journey = []
            next_entry_idx = first_entry_idx
            def build_leg(route_conns):
                rs = route_conns[0]
                re = route_conns[-1]
                return PtJourneyLeg(rs['route'].line_id, rs['route'].route_id,
                                    rs['dep stop'].id, re['arr stop'].id,
                                    rs['dep time'], re['arr time'],
                                    rs['departure'].vehicle.id)

            while next_stop_id is not dest_stop.id:
                # get and unpack the entry for the next step
                profile = stop_profiles[next_stop_id, dest_stop.id]
                entry = profile[next_entry_idx]
                t_dep, t_dest, cc, next_stop_id, next_entry_idx = entry[:5]
                if (len(route_conns) > 0 and
                    route_conns[-1]['route'] != cc['route']):
                    # there has been a transfer, or we've reached the end!
                    trip.planned_journey.append(build_leg(route_conns))
                    route_conns = []
                route_conns.append(cc)

            # add the final route
            if len(route_conns) > 0:
                trip.planned_journey.append(build_leg(route_conns))

        # filter out trips for which best journey isn't good enough
        self.tracked_trips = [self.trips[tt] for tt in tracked_trip_ids
                              if len(self.trips[tt].planned_journey) > 0]

    def run(self):
        # TODO initialize dict of waiting passengers by route/stop
        for stop_event in self.stop_events:
            pass
            # remove disembarking passengers
            # for each disembarker:
                # determine time of arrival at their next dest, add it to their real journey
                # if their next destination is the end of their journey:
                    # log their completed journey
                # elif next destination is a transfer:
                    # add them to the transferer list for that route/stop at their arrival time

            # add stop_event's new passengers to waiting passengers by route/stop
            # n_allowed = remaining room on bus
            # if remaining room on bus < number of waiting passengers:
                # sort passengers by arrival time
            # load the first n_allowed passengers


class ConstantDemandSim(AbstractTransitOnlySim):

    def trips_from_od_data(self, od_data):
        # TODO implement this
        pass


class DisaggregatedDemandSim(AbstractTransitOnlySim):
    """Doesn't model vehicle movement at all, just assignment of passengers
       to routes."""

    def trips_from_od_data(self, od_csv_file):
        self.trips = OrderedDict()
        col_types = {'ID': str, 't_orix': float, 't_oriy': float,
                     't_desx': float, 't_desy': float, 't_time': float,
                     't_expf': float}
        oddf = pd.read_csv(self.environment_dir / od_csv_file, dtype=col_types)
        logging.info('Finished reading od file')
        for _, row in oddf.iterrows():
            if any(pd.isnull(row[col]) for col in col_types):
                continue
            t_time = row['t_time']
            start_hours = int(t_time) // 100
            # this allows fractional minutes to represent seconds
            start_minutes = t_time % 100
            if start_hours >= 24:
                start_hours -= 24
            start_time_s = start_hours * 3600 + start_minutes * 60
            # TODO randomize the start seconds?  Or even more?
            # TODO Randomize origin and destination?
            trip = PassengerTrip(row['ID'], (row['t_orix'], row['t_oriy']),
                                 (row['t_desx'], row['t_desy']),
                                 start_time_s)
            if 't_expf' in row and row['t_expf'] > 1:
                for ii in range(round(row['t_expf'])):
                    # TODO we should probably handle expansion factors better.
                    # Right now we ignore the fractional component.
                    # t_expf=5.32 is treated as t_expf=5, t_expf=1.55 is
                    # t_expf=2.
                    copy_trip = deepcopy(trip)
                    copy_trip.id = '{}.{}'.format(trip.id, ii)
                    self.trips[copy_trip.id] = copy_trip
            else:
                self.trips[trip.id] = trip
        logging.info('all trips added')

    def run(self):
        pass
        # set up the log

        # bus states:
        # idle
        # stopped
        # deadhead
        # travelling on route

        # for each timestep:

            # this part will be ignored if we're not spawning buses at will
            # for each route:
                # if a new departure should take place:
                    # spawn a new bus at the beginning of the route

            # for each bus:
                # if the bus is stopped:
                    # if boarding is done *or* it is time to start next route traversal:
                    # if the bus should resume:
                        # add currently-waiting passengers for this bus to the bus, up to capacity
                        # log boarding events
                        # set to not-stopped
                        # log the bus's departure

                # if the bus is not stopped:
                    # update the position of the bus

                # if the bus should now stop at its next stop:
                    # set its state to stopped
                    # log the bus's arrival
                    # remove passengers who are getting off here
                    # log disembarking events
                    # if state is "deadhead", set state to "idle"

                # if the bus is at the end of its current route:
                    # set it on its deadhead to the start of the next route
                    # log the start of the deadhead


            # for each stop:
                # add a new waiting passenger for each new trip that starts at this time
                # log the new waiting passengers
                # remove any passengers who have waited too long and give up
                # log any give-up events
