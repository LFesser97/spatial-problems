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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
from pathlib import Path

SIM_SYSTEM = pd.DataFrame({"ridership": [21897.3], "cost": [10035]})
REAL_SYSTEM = pd.DataFrame({"ridership": [20316.6], "cost": [10035]})
BAGLOEE = pd.DataFrame({"ridership": [21499], "cost": [4206]})

def plot_transit_performance_curves(data_csvs):

    for path in data_csvs:
        path = Path(path)
        df = pd.read_csv(path)
        label = path.stem
        plt.plot(df["cost"], df["ridership"], marker="o", label=label)
        for _, row in df.iterrows():
            label = "$\omega = {}$".format(row["weight"])
            loc = np.array((row["cost"], row["ridership"]))
            plt.annotate(label, xy=loc, xytext=loc + (300, -300))

    # draw the real system's performance as a dot
    plt.scatter(REAL_SYSTEM["cost"], REAL_SYSTEM["ridership"], color="black",
                label="existing bus network")
    plt.scatter(SIM_SYSTEM["cost"], SIM_SYSTEM["ridership"], color="gray",
                label="existing bus network (simulated)")
    plt.scatter(BAGLOEE["cost"], BAGLOEE["ridership"], color="red",
                label="best from Bagloee")

    # draw an arrow indicating which direction is improvement.
    arrow_loc = np.array((0.2, 0.7))
    plt.annotate("this direction\n is better", xycoords="figure fraction",
                 textcoords="figure fraction",
                 xy=arrow_loc, xytext = arrow_loc + (0.05, -0.15),
                 arrowprops=dict(facecolor="none", shrink=0.05))

    plt.legend()
    plt.xlabel("cost (kW)")
    plt.ylabel("ridership (# boardings)")
    plt.show()


if __name__ == "__main__":
    plot_transit_performance_curves(sys.argv[1:])