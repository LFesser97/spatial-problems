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

import xml.etree.ElementTree as xmlparser


def parse_vehicles(vehicles_xmlfile):
    """Translates a MATsim vehicles xml to a python data structure."""
    xmltree = xmlparser.parse(vehicles_xmlfile)
    # TODO fil this in


def schedule_to_xml(departures_list, schedule_xmlfile):
    """Inserts the timetabled schedule in a matsim xml transit file."""
    xmltree = xmlparser.parse(schedule_xmlfile)
    # TODO fill this in


def learn_schedule(vehicles, num_episodes, ep_length):
    # initialize the learnable model
    # for ep in range(num_episodes):
        # for tt in range(ep_length):
            # update vehicle simulator to get positions, free-status of all vehicles
            # record any departures that have occured at this timestep
            # for vehicle in vehicles:
                # if vehicle is free:
                    # construct input - what does it consist of???
                    # record the input
                    # put input through model, get its choice of route
                    # assign vehicle to chosen route
        # output chosen departures for this day to the mobility simulator
        # run mobility simulator
        # get passenger movements output from mobility simulator
        # compute return for each action
        # compute and log overall utility of the system
        # backprop using the returns for each action, updating the model
    # return the learned model
