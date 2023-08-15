import sys
import pickle
from pathlib import Path

from tqdm import tqdm
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
import matplotlib.pyplot as plt


def main(in_folder, out_folder):
    data_paths = {}
    for subfolder in Path(in_folder).iterdir():
        # find the newest file in the sub-directory
        scalar_files = list(subfolder.glob('events.out.tfevents.*'))
        scalar_files.sort(key=lambda x: x.stat().st_mtime_ns)
        if scalar_files:
            data_paths[subfolder] = scalar_files[-1].as_posix()

    out_folder = Path(out_folder)
    if not out_folder.exists():
        out_folder.mkdir(parents=True)

    for dir_path, data_path in tqdm(data_paths.items()):
        out_path = (out_folder / dir_path.stem).with_suffix('.pkl')
        if out_path.exists():
            continue

        event_acc = EventAccumulator(data_path)
        event_acc.Reload()
        # Show all tags in the log file
        print(event_acc.Tags())

        # E. g. get wall clock, number of steps and value for mean cost
        try:
            w_times, step_nums, vals = zip(*event_acc.Scalars('mean cost'))
        except KeyError:
            # No mean cost in this file
            continue

        step_nums = np.array(step_nums)
        vals = np.array(vals)

        with out_path.open('wb') as ff:
            pickle.dump({'step_nums': step_nums, 'vals': vals}, ff)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])