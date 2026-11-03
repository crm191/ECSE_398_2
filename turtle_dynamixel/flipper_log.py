"""
Minimal logger for system identification runs (tools/fit_flipper_params.py).

    log = FlipperLog()
    ...
    # in the control loop, after sending the currents:
    log.add(time.time(), [positions[mid] for mid in ALL_IDS], [velocities[mid] for mid in ALL_IDS],
            [torque_cmds[mid] for mid in ALL_IDS])
    ...
    log.save("sysid_air_motor1.csv")
"""
import csv
from pathlib import Path


class FlipperLog:
    """Rows of time (s), motor positions (rad), motor velocities (rad/s) and Goal Current steps."""

    HEADER = ["t", "q1", "q2", "q3", "dq1", "dq2", "dq3", "i1", "i2", "i3"]

    def __init__(self):
        self.rows = []

    def add(self, t, q, dq, current_steps):
        self.rows.append([float(t), *map(float, q), *map(float, dq), *map(int, current_steps)])

    def save(self, path):
        with open(Path(path), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.HEADER)
            writer.writerows(self.rows)
