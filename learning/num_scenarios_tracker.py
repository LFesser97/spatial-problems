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

import torch
import numpy as np
import matplotlib.pyplot as plt

class NumScenariosTracker:
    def __init__(self) -> None:
        self.unique_scenarios = set()
        self.step_new_scenarios = []

    def add_scenarios(self, scenarios):
        """scenarios is a list of sets."""
        old_len = len(self.unique_scenarios)
        for scenario in scenarios:
            scenario = _scenario_to_set(scenario)
            self.unique_scenarios.add(scenario)
        n_new = len(self.unique_scenarios) - old_len
        self.step_new_scenarios.append(n_new)

    def add_scenario(self, scenario):
        """scenario is a set."""
        self.add_scenarios([scenario])

    @property
    def n_scenarios_so_far(self):
        return len(self.unique_scenarios)

    def plot(self):
        plt.plot(self.step_new_scenarios)
        plt.xlabel('Step')
        plt.ylabel('New Scenarios')

    def plot_cum(self):
        n_scenarios_so_far = np.cumsum(self.step_new_scenarios)
        plt.plot(n_scenarios_so_far)
        plt.xlabel('Step')
        plt.ylabel('Scenarios So Far')


class TotalInfoTracker(NumScenariosTracker):
    def __init__(self, n_routes, device=None) -> None:
        super().__init__()
        # add 1 to handle values of -1
        self.scenario_tensor = torch.zeros((0, n_routes + 1), device=device,
                                           dtype=bool)

    @property
    def _n_candidates(self):
        return self.scenario_tensor.shape[1]

    @property
    def _dev(self):
        return self.scenario_tensor.device

    def add_scenarios(self, scenarios):
        scenario_tensors = []
        for scenario in scenarios:
            tensor_scenario = torch.zeros(self._n_candidates, device=self._dev,
                                          dtype=bool)
            tensor_scenario[scenario] = 1
            scenario = _scenario_to_set(scenario)
            if scenario not in self.unique_scenarios:
                scenario_tensors.append(tensor_scenario)
        new_tensor = torch.stack(scenario_tensors)
        self.scenario_tensor = torch.cat((self.scenario_tensor, new_tensor))

        return super().add_scenarios(scenarios)

    def total_info(self):
        """scenario_tensor is a tensor of shape (n_scenarios, n_candidates)"""
        # get number of routes in which each pair of scenarios differ
        if self.scenario_tensor.shape[0] == 0:
            # no scenarios, so no information
            return 0

        n_routes_tried = self.scenario_tensor.any(dim=0).sum().item()
        if self.scenario_tensor.shape[0] == 1:
            # information is just the number of routes that have been seen
            return n_routes_tried

        xors = self.scenario_tensor[None] ^ self.scenario_tensor[:, None]
        diffs = xors.sum(dim=-1)
        # get minimum diff of each scenario from any other, which is how much
         # info it uniquely contributes
        diffs.fill_diagonal_(self._n_candidates)
        mins = diffs.min(dim=-1).values
        # sum each scenario's contribution to get total info
        total_exclusive_info = mins.sum().item()
        
        return total_exclusive_info + n_routes_tried


def _scenario_to_set(scenario):
    """Convert a scenario to a frozen set"""

    if type(scenario) in (set, frozenset):
        # already formatted correctly
        return frozenset(scenario)

    if len(scenario) == 0:
        # it's empty
        return frozenset()

    # if it's a tensor, convert it to a list or list of lists
    if type(scenario) is torch.Tensor:
        scenario = scenario.tolist()
    elif type(scenario[0]) is torch.Tensor:
        scenario = [rr.tolist() for rr in scenario]

    if type(scenario[0]) is list:
        # Convert the list routes to tuples
        scenario = [tuple(rr) for rr in scenario]

    return frozenset(scenario)

