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
import shutil
import shlex
from pathlib import Path

from . import matsim_events as event_module
from .simulator import AbstractSim


"""Note that when searching for xml elements with find() or iter(), we prefix
the search string with "{*}" to indicate that we are agnostic as to the
namespace, as some of these config files use namespaces."""


class MatsimSimulator(AbstractSim):
    def __init__(self, show_output=False,
                 matsim_path='/home/andrew/matsim/my_envs/mandl/',
                 java_exec='/usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java',
                 **kwargs):
        """
        Note that as of version 0.10.1, matsim is not compatible with java
        versions newer than 8.  That's why we use java8 in java_exec here.
        """
        super().__init__(**kwargs)

        self.matsim_path = Path(matsim_path)
        self.show_output = show_output
        # initialize transit from the configuration files
        self._java_exec = java_exec

    def run(self, memory_MB=512):
        cmd = '{java} -Xmx{mem}m -cp {matsim} org.matsim.run.Controler {cfg}'
        cmd = cmd.format(java=self._java_exec, mem=memory_MB,
                         matsim=self.matsim_path, cfg=self.config_path)
        # TODO clear output directory...how to handle this?  Right now we
        # just delete it, but we should probably save every run in a new dir.
        # Oh well, add that functionality when we find we need it.
        output_dir = self._get_param_as_path('controler', 'outputDirectory')
        if output_dir.exists():
            shutil.rmtree(output_dir)
        subprocess.run(shlex.split(cmd), capture_output=not self.show_output,
                       check=True)

    def get_pt_passenger_traces(self, iter=-1):
        output_dir = self._get_param_as_path('controler', 'outputDirectory')
        events_tree = event_module.load_events(output_dir, iter)
        return event_module.get_pt_passenger_traces(events_tree)

    def get_pt_vehicle_traces(self, iter=-1):
        output_dir = self._get_param_as_path('controler', 'outputDirectory')
        events_tree = event_module.load_events(output_dir, iter)
        return event_module.get_pt_vehicle_traces(events_tree)

    def get_pt_trips(self, iter=-1):
        pt_traces = self.get_pt_vehicle_traces(iter)
        trips = []
        for trace in pt_traces.values():
            trips += trace.trips
        return trips

    def set_transit(self, pt_system):
        super().set_transit(pt_system)
        self._transit.write_as_xml(self._get_pt_schedule_path(),
                                   self._get_pt_vehicles_path())
