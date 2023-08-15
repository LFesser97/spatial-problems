import argparse
from tqdm import tqdm
from lxml import etree
from pathlib import Path
import torch
import torch.nn as tnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from collections import defaultdict, OrderedDict, namedtuple

from simulation import MatsimSimulator
from world import Departure, PtVehicle, PtSystem

# the number of simulator seconds covered by each MDP step
TS_SIZE = 60


class TorchModel(tnn.Module):
    def __init__(self, pt_system, input_size=1):
        super().__init__()
        self.trunk = tnn.Sequential(
            tnn.Linear(1, 32),
            tnn.ReLU(),
            )
        vehicle_type_ids = [vt.id for vt in pt_system.get_vehicle_types()]
        self.action_space = [None] + vehicle_type_ids
        routes = pt_system.get_routes()
        self.route_ids = [rr.get_unique_id() for rr in pt_system.get_routes()]
        # we use a ModuleList here instead of a ModuleDict with route ids as
        # keys because ModuleDict keys have to be strings, but route ids are
        # specifically tuples and not strings to enforce uniqueness.
        self.route_heads = tnn.ModuleList(
            [tnn.Linear(32, len(self.action_space)) for rr in routes])

    def forward(self, input_time):
        normalized_time = (input_time - (3600 * 12)) / (3600 * 12)
        act = self.trunk(torch.tensor([normalized_time]))
        head_outputs = [head(act) for head in self.route_heads]
        return dict(zip(self.route_ids, head_outputs))


@dataclass
class _TripRecord:
    """Helper struct for building a schedule from a model."""
    output: torch.Tensor
    action_idx: int
    dep_id: str = None

    @property
    def chosen_output(self):
        return self.output[self.action_idx]


def learn_schedule(sim, model, num_episodes, epsilon):
    loss_func = tnn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    # for logging purposes
    progress_log = []

    for ep in tqdm(range(num_episodes)):
        # run the forward pass to select actions and build a schedule
        ep_records = build_and_set_schedule(sim, model, epsilon)
        sim.run()

        # get the reward for each departure and compute losses
        trips = {(trip.line_id, trip.route_id, trip.departure_id): trip
                 for trip in sim.get_pt_trips()}
        chosen_outputs = []
        rewards = []
        for time_records in ep_records:
            for (line_id, route_id), record in time_records.items():
                # find the trip associated with this action and compute return
                key = (line_id, route_id, record.dep_id)
                if key in trips:
                    reward = person_dist_trip_reward(sim.get_transit(),
                                                     trips[key])
                else:
                    # must mean no action was performed
                    reward = 0
                chosen_outputs.append(record.chosen_output)
                rewards.append(reward)

        # run the backwards pass and update the weights
        loss = loss_func(torch.stack(chosen_outputs), torch.tensor(rewards))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # compute and log overall utility of the system
        ep_return = sum(rewards)
        ep_loss = loss.item()
        print('return:', ep_return, 'loss:', ep_loss)
        progress_log.append({'episode': ep,
                             'return': ep_return,
                             'avg loss': ep_loss})

    return pd.DataFrame(progress_log).set_index('episode')


def build_and_set_schedule(sim, model, epsilon=0):
    input_times = np.arange(sim.get_start_time_s(), sim.get_end_time_s(),
                            TS_SIZE)
    records = []
    for time in input_times:
        output = model(time)
        time_records = {}
        for key, route_output in output.items():
            if epsilon and epsilon > np.random.random():
                # choose a random action
                action_idx = np.random.randint(len(model.action_space))
            else:
                # let the network choose the action
                action_idx = torch.argmax(route_output)
            time_records[key] = _TripRecord(route_output, action_idx)
        records.append(time_records)

    # clear the existing departures
    current_pt_system = sim.get_transit()
    routes = current_pt_system.get_routes()
    for route in routes:
        route.departures = []

    # create a departure for each action and store its dep id
    # new_vehicles = []
    for time, time_records in zip(input_times, records):
        for ruid, route_record in time_records.items():
            route = current_pt_system.get_route(*ruid)
            veh_type_id = model.action_space[route_record.action_idx]
            if veh_type_id is not None:
                route_record.dep_id = str(len(route.departures))
                # define a new vehicle for each departure
                vehicle_type = current_pt_system.get_vehicle_type(veh_type_id)
                vehicle_id = '{}_{}_{}'.format(route.line_id, route.route_id,
                                               route_record.dep_id)
                vehicle = PtVehicle(vehicle_id, vehicle_type)
                # new_vehicles.append(vehicle)
                dep = Departure(route_record.dep_id, time, vehicle)
                route.departures.append(dep)

    # create a new PtSystem object from the routes and vehicle types, and set
    # it in the simulator.
    sim.set_transit(PtSystem(routes, current_pt_system.get_vehicle_types()))
    # sim.pt_vehicles = new_vehicles
    return records


def person_dist_trip_reward(pt_system, pt_trip, pd_weight=1,
                            energy_weight=0.5):
    route = pt_system.get_route(pt_trip.line_id, pt_trip.route_id)
    # convert from meters to kilometers to keep values in check
    dist_matrix_km = route.get_stop_crowdist_matrix() / 1000
    stops_od_matrix = pt_trip.get_trip_OD_matrix()
    # TODO should we penalize the time between stops somehow?  The sim takes
    # care of that implicitly now.

    if dist_matrix_km.shape != stops_od_matrix.shape:
        # not all route stops were made on this trip.  Can happen if a trip
        # gets cut off at the end of the simulation.
        trip_stop_ids = [ss.facility_id for ss in pt_trip.stops]
        new_od_matrix = np.zeros(dist_matrix_km.shape)
        for ii, planned_stop in enumerate(route.stops):
            try:
                # an actual stop was made for this route stop, so copy the
                # corresponding row and column to new od matrix
                jj = trip_stop_ids.index(planned_stop.facility_id)
                new_od_matrix[ii, :] = stops_od_matrix[jj, :]
                new_od_matrix[:, ii] = stops_od_matrix[:, jj]
            except ValueError:
                # the planned stop was not actually made, so leave 0s here
                pass
        stops_od_matrix = new_od_matrix

    # multiply OD matrix by distance matrix and take the sum
    # units of "person km"
    person_dist_reward = np.sum(dist_matrix_km * stops_od_matrix)

    # compute a component due to the cost of driving
    vehicle = pt_system.get_vehicle(pt_trip.vehicle_id)
    avg_power_kW = vehicle.type.avg_power_kW
    duration_s = pt_trip.end_time - pt_trip.start_time
    energy_used_MJ = avg_power_kW * duration_s / 1000

    # weighted combination of components
    reward = pd_weight * person_dist_reward - energy_weight * energy_used_MJ
    return reward


def plot_learning_progress(learnlog_df):
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('episode')
    axes = [ax1, ax1.twinx()]
    colnames = ['return', 'avg loss']
    colours = ['tab:blue', 'tab:orange']
    for ax, colname, colour in zip(axes, colnames, colours):
        ax.set_ylabel(colname, color=colour)
        ax.plot(learnlog_df.index, learnlog_df[colname], color=colour)
        ax.tick_params(axis='y', labelcolor=colour)

    fig.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('desc_dir', help='\
Directory containing the xml files describing the simulation environment.')
    parser.add_argument('matsim_path', help='Path to the matsim .jar file.')
    parser.add_argument('--inmodel', help='\
                        Path from which to load starting model weights.')
    parser.add_argument('-o', '--out', help='\
                        Path to which to save the learned model weights.')
    parser.add_argument('-e', '--episodes', type=int, default=100,
                        help='Number of episodes to run.')
    parser.add_argument('-r', '--epsilon', type=float, default=0.05,
                        help='Probability of totally random action.')
    parser.add_argument('-v', '--verbose', action='store_true', help='\
Display matsim output.')
    parser.add_argument('--plot', action='store_true', help='\
Plot the rewards received over time.')
    args = parser.parse_args()
    # initialize a Simulation object
    sim = MatsimSimulator(env_dir=args.desc_dir, matsim_path=args.matsim_path,
                          show_output=args.verbose)
    initial_pt_system = sim.get_transit()
    # initialize the learnable model
    model = TorchModel(initial_pt_system)
    if args.inmodel:
        model.load_state_dict(torch.load(args.inmodel))
    learnlog = learn_schedule(sim, model, args.episodes, args.epsilon)

    # produce the final schedule from the model with epsilon = 0
    model.eval()
    with torch.no_grad():
        build_and_set_schedule(sim, model)
    # compute reward with final schedule
    sim.run()
    final_pt_system = sim.get_transit()
    final_return = sum([person_dist_trip_reward(final_pt_system, tt)
                        for tt in sim.get_pt_trips()])
    print('Final return:', final_return)
    if args.out:
        torch.save(model.state_dict(), args.out)
    if args.plot:
        plot_learning_progress(learnlog)

if __name__ == '__main__':
    main()
