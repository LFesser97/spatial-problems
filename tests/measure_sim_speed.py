from pathlib import Path
from simulation import DisaggregatedDemandSim

import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


LAVAL_ENV_DIR='/home/andrew/matsim/my_envs/laval'


def measure_trip_assignment_speed(env_dir):
    sim = DisaggregatedDemandSim(od_data='od.csv', basin_radius_m=500,
                                 delay_tolerance_s=1800, env_dir=env_dir)


measure_trip_assignment_speed(LAVAL_ENV_DIR)
