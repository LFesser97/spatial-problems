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

from lxml import etree
import os
from pathlib import Path
import gzip
import io
import sys
from collections import defaultdict
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class PtStop:
    """A simple struct for representing a vehicle's stop at a stop facility."""
    # id of the stop facility
    facility_id: str
    arrival_time: float
    departure_time: float = None
    delay: float = None
    boarders: List[str] = field(default_factory=list)
    disembarkers: List[str] = field(default_factory=list)
    boarder_wait_times: List[float] = field(default_factory=list)


@dataclass
class PtTrip:
    line_id: str
    route_id: str
    departure_id: str
    vehicle_id: str
    driver_id: str
    start_time: float
    end_time: float = None
    stops: List[PtStop] = field(default_factory=list)

    @classmethod
    def from_driver_starts_event(cls, event):
        if type(event) is etree._ElementTree:
            event = event.attrib
        return cls(event['transitLineId'], event['transitRouteId'],
                   event['departureId'], event['vehicleId'],
                   event['driverId'], float(event['time']))

    def __post_init__(self):
        if self.line_id == 'Wenden' and self.route_id == 'Wenden' and \
           self.departure_id == 'Wenden':
            self.line_id = None
            self.route_id = None
            self.departure_id = None

    def is_deadhead(self):
        return self.line_id is None and self.route_id is None and \
            self.departure_id is None

    def get_trip_OD_matrix(self):
        OD_matrix = np.zeros((len(self.stops), len(self.stops)), dtype=int)
        # iterate over the stops to build the matrix
        for ii, istop in enumerate(self.stops):
            for boarder in istop.boarders:
                for jj, jstop in enumerate(self.stops[ii + 1:], start=ii + 1):
                    if boarder in jstop.disembarkers:
                        OD_matrix[ii, jj] += 1
                        break

        return OD_matrix


class PtVehicleTrace:
    def __init__(self, event=None):
        """Optionally takes a first event (dict or lxml Element is fine)"""
        self.vehicle_id = None
        # a trip is a traversal of a route or a journey in-between routes
        self.trips = []

        # internal-use-only members
        self._passengers = []
        self._in_progress_trip = None
        self._in_progress_stop = None
        if event is not None:
            self.add_event(event)

    def add_event(self, event, passenger_wait_time=None):
        def finalize_trip():
            self._in_progress_trip.end_time = float(event['time'])
            self.trips.append(self._in_progress_trip)
            self._in_progress_trip = None


        if type(event) is etree._ElementTree:
            event = event.attrib

        # assign the vehicle ID if we don't have one yet
        for vehicle_key in ['vehicle', 'vehicleId']:
            if vehicle_key in event:
                vehicle_id = event[vehicle_key]
                if self.vehicle_id != vehicle_id:
                    if self.vehicle_id is not None:
                        warn = 'Vehicle ID was {} but now is {}?'
                        raise Warning(warn.format(self.vehicle_id, vehicle_id))
                    self.vehicle_id = vehicle_id

        # what kind of event could it be?
        if event['type'] == 'TransitDriverStarts':
            # create new trip
            if self._in_progress_trip is not None:
                warn = """Vehicle {} is starting a new trip, but wasn\'t \
                    finished the last one!"""
                raise Warning(warn.format(self.vehicle_id))
            self._in_progress_trip = PtTrip.from_driver_starts_event(event)

        elif event['type'] == 'VehicleArrivesAtFacility':
            # initialize a new in-progress stop
            if self._in_progress_stop is not None:
                warn = """Vehicle {} arrived at stop {}, but wasn\'t \
                    finished the last one!"""
                raise Warning(warn.format(self.vehicle_id, event['facility']))
            self._in_progress_stop = PtStop(event['facility'],
                                            float(event['time']))
            if event['delay'] != 'Infinity':
                self._in_progress_stop.delay = float(event['delay'])

        elif event['type'] == 'PersonEntersVehicle':
            # add person at the current stop
            if event['person'] != self._in_progress_trip.driver_id:
                self._passengers.append(event['person'])
                self._in_progress_stop.boarders.append(event['person'])
                self._in_progress_stop.boarder_wait_times.append(passenger_wait_time)

        elif event['type'] == 'PersonLeavesVehicle':
            if event['person'] == self._in_progress_trip.driver_id:
                # the driver left, so finalize the current trip
                # finalize_trip()
                self._in_progress_trip.end_time = float(event['time'])
                self.trips.append(self._in_progress_trip)
                self._in_progress_trip = None
            else:
            # remove person at the current stop
                self._passengers.remove(event['person'])
                self._in_progress_stop.disembarkers.append(event['person'])

        elif event['type'] == 'VehicleDepartsAtFacility':
            # finalize the in-progress stop
            self._in_progress_stop.departure_time = float(event['time'])
            if event['delay'] != "Infinity":
                self._in_progress_stop.delay = float(event['delay'])
            # save the just-finished stop
            self._in_progress_trip.stops.append(self._in_progress_stop)
            self._in_progress_stop = None

        elif event['type'] == 'vehicle aborts':
            # this event occurs if a transit vehicle is in motion when the
            # simulation ends.
            # finalize_trip()
            self._in_progress_trip.end_time = float(event['time'])
            self.trips.append(self._in_progress_trip)
            self._in_progress_trip = None

        elif event['type'] in ['vehicle enters traffic',
                               'vehicle leaves traffic', 'left link',
                               'entered link']:
            # ignore these, we don't need them
            pass

        else:
            raise Warning('Unknown event type {} given!'.format(event['type']))


def get_pt_passenger_traces(events_tree):
    # build a collection of everyone who took transit (not counting drivers)
    events_elem = events_tree.getroot()
    passenger_ids = set()
    for event_elem in events_elem.iter('event'):
        if event_elem.get('type') == 'waitingForPt':
            passenger_ids.add(event_elem.get('agent'))

    # build a dict of passenger IDs to traces of their movements
    passenger_traces = defaultdict(list)
    for event_elem in events_elem.iter('event'):
        event_dict = event_elem.attrib
        for passenger_key in ['person', 'agent']:
            if passenger_key in event_dict:
                id = event_dict[passenger_key]
                if id in passenger_ids:
                    passenger_traces[id].append(event_dict)

    return passenger_traces


def get_pt_vehicle_traces(events_tree):
    """Returns a dictionary mapping vehicle IDs to PtVehicleTrace objects."""
    events_elem = events_tree.getroot()
    vehicle_traces = {}
    pt_waits = {}
    person_ids = set()
    for event_elem in events_elem.iter('event'):
        event_dict = event_elem.attrib
        # record when people start waiting for transit
        if event_dict['type'] =='waitingForPt':
            pt_waits[event_dict['agent']] = float(event_dict['time'])

        # track each agent ID
        for person_key in ['agent', 'person']:
            if person_key in event_dict:
                person_ids.add(event_dict[person_key])
                break

        # add vehicle events
        vehicle_id = None
        for vehicle_key in ['vehicle', 'vehicleId']:
            if vehicle_key in event_dict:
                vehicle_id = event_dict[vehicle_key]
                break
        if vehicle_id is None or vehicle_id in person_ids:
            # this is a car, not a PT vehicle!  Skip the event.
            continue

        wait_time = None
        if event_dict['type'] == 'PersonEntersVehicle' and \
            event_dict['person'] in pt_waits:
            # add the time this person waited
            wait_time = float(event_dict['time']) - \
                pt_waits[event_dict['person']]
            # they're done waiting, so remove them from waiting dict
            del pt_waits[event_dict['person']]

        # add the event to a vehicle trace
        if vehicle_id not in vehicle_traces:
            vehicle_traces[vehicle_id] = PtVehicleTrace(event_dict)
        else:
            vehicle_traces[vehicle_id].add_event(event_dict, wait_time)
    return vehicle_traces


def load_events(output_dir='.', iter=-1):
    """By default, looks at the last iteration.  set 'iter' to change this."""
    # find the latest iteration directory
    search_dir = Path(output_dir) / 'ITERS'
    latest_path = sorted(search_dir.iterdir(), key=os.path.getmtime)[iter]
    # extract the events xml from the gzip
    events_gz = next(latest_path.glob("*.events.xml.gz"))
    with gzip.open(str(events_gz), 'rb') as ff:
        xml_str = ff.read()
    # print(io.StringIO(xml_str))
    events_tree = etree.parse(io.BytesIO(xml_str))
    return events_tree


# the below is testing code


def main():
    if len(sys.argv) > 1:
        events_tree = load_events(sys.argv[1])
    else:
        events_tree = load_events()
    print('analyzing events')
    vehicle_traces = get_pt_vehicle_traces(events_tree)
    # np.random.seed(5)
    randkey = np.random.choice(list(vehicle_traces.keys()))
    vt = vehicle_traces[randkey]
    for trip in vt.trips:
        if not trip.is_deadhead():
            odm = trip.get_trip_OD_matrix()
            if odm.sum() > 0:
                print(trip.start_time)
                print(odm)
    # passenger_traces = get_pt_passenger_traces(events_tree)
    # randkey = np.random.choice(list(passenger_traces.keys()))
    # trace = passenger_traces[randkey]



if __name__ == "__main__":
    main()
