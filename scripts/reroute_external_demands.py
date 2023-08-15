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
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist

from analyze_network import parse_census_code, MTL_CSD, LAVAL_CSD


METRO_CODE = 4
LAVAL_BUS_CODE = 6
TRANSIT_CODES = set((3, 4, 5, 6))
CAR_CODES = set((1, 2))


LAVAL_EXITS = {
    MTL_CSD: {
        "in": [(7620610.413081295, 1263177.3381993298), # bus 52E
               (7618180.725586116, 1244748.6329287067), # bus 902N, 55N, 151N
               (7622352.269461416, 1253792.3049979685), # bus 52E
               (7621184.195696177, 1248459.6143421473), # orange line Cartier
               (7619019.206962035, 1247829.1273129827), # Orange Line de la Concorde
               (7618277.674835001, 1247295.8638324633), # Orange Line Montmorency
               ],
        "out": [(7623441.387349587, 1259015.3166801454), # bus 925S
                (7621871.942138646, 1253531.6440306604), # bus 904S
                (7618180.725586116, 1244748.6329287067), # bus 902S, 55S, 151S
                (7622150.156897315, 1253805.5341613956), # bus 52O
                (7621184.195696177, 1248459.6143421473), # Orange Line Cartier
                (7619019.206962035, 1247829.1273129827), # Orange Line de la Concorde
                (7618277.674835001, 1247295.8638324633), # Orange Line Montmorency
                ]
    },
    # would be nice to remap trips that aren't to/from Montreal as well, but
    # they're only about 3% of the total so it's low-priority.
    # TDB_CSD: {
    #     "in": [],
    #     "out": []
    # },
    # DEUXMONTAGNES_CSD: {
    #     "in": [],
    #     "out": []
    # },
    # LESMOULINS_CSD: {
    #     "in": [],
    #     "out": []
    # }
}

def reroute_external_demands(in_od_path, out_od_path):
    od_df = pd.read_csv(in_od_path)
    # for each trip:
    for row_idx, row in od_df.iterrows():
        ori_csd = parse_census_code(row["t_oricsd"])[1]
        des_csd = parse_census_code(row["t_descsd"])[1]
        end_csds = set((ori_csd, des_csd))
        is_intercity = LAVAL_CSD in end_csds and len(end_csds) > 1

        # determine if it uses transit
        # should contain 3, 4, 5, or 6, and not contain 1, 2
        modes = (row["t_mode1"], row["t_mode2"], row["t_mode3"],
                 row["t_mode4"], row["t_mode5"], row["t_mode6"],
                 row["t_mode7"], row["t_mode8"], row["t_mode9"])
        uses_transit = len(CAR_CODES & set(modes)) == 0 and \
            len(TRANSIT_CODES & set(modes)) > 0

        # if the trip starts or ends outside laval and uses transit:
        if is_intercity and uses_transit:
            # - retreive the set of "exit" coordinates based on the code and 
             # whether it's origin or destination
            if ori_csd != LAVAL_CSD and ori_csd in LAVAL_EXITS:
                cross_coords = LAVAL_EXITS[ori_csd]["in"]
            elif des_csd in LAVAL_EXITS:
                cross_coords = LAVAL_EXITS[des_csd]["out"]
            else:
                # the external csd isn't in our remapping dict, so just skip it
                continue

            # compute distance from the cross coords to the endpoints
            endpoints = ((row["t_orix"], row["t_oriy"]), 
                         (row["t_desx"], row["t_desy"]))
            distances = cdist(endpoints, cross_coords)
            _, cross_idx = np.unravel_index(distances.argmin(), 
                                            distances.shape)

            # remap row's external endpoint to the exit coordinate which is 
             # closest to any endpoint
            new_coord = cross_coords[cross_idx]
            if ori_csd != LAVAL_CSD:
                od_df.loc[row_idx, "t_orix"] = new_coord[0]
                od_df.loc[row_idx, "t_oriy"] = new_coord[1]
            else:
                od_df.loc[row_idx, "t_desx"] = new_coord[0]
                od_df.loc[row_idx, "t_desy"] = new_coord[1]

    od_df.to_csv(out_od_path, index=False)


if __name__ == "__main__":
    assert len(sys.argv) == 3, "Must supply input and output file paths!"
    reroute_external_demands(sys.argv[1], sys.argv[2])