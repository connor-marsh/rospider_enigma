import os

import numpy as np
this_file_dir = os.path.dirname(os.path.abspath(__file__))
base_gait_names = [
    "tripod", "tripod_huge", "tripod_right", "tripod_huge_right",
    "ripple", "ripple_tiny", "ripple_right", "ripple_tiny_right",
]
mirrored_gait_names = [
    "tripod_backwards", "tripod_huge_backwards", "tripod_left", "tripod_huge_left",
    "ripple_backwards", "ripple_tiny_backwards", "ripple_left", "ripple_tiny_left",
]
gait_names = base_gait_names + mirrored_gait_names


gait_tables_orig = []

for name in mirrored_gait_names:
    base_name = name.replace("_backwards", "").replace("_left", "_right")
    gait_table = np.loadtxt(f"{this_file_dir}/gaits/{base_name}.csv",
                            delimiter=",", dtype=np.float32)
    np.savetxt(f"{this_file_dir}/gaits/{name}.csv", np.flip(gait_table, axis=0), delimiter=",")