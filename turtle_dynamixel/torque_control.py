"""
Drop-in replacement for torque_control.py in the turtle_dynamixel package: model-based control of the front
flipper(s) of the current turtle design. flipper_trajectory.py and flipper_main_multi.py can call it unchanged.

Differences from the old version:
    - The model is the current limb from mujoco_models/turtle.xml, not the old mass.py.
    - Motor angles are mapped to the model's joints with the offsets and signs in flipper_params.json.
    - The return value is already in Goal Current steps (torque constant, current limit). The old `tau * 150`
      scaling is gone, and Kp / KD stay in Nm/rad and Nm s/rad.

Parameters are read from flipper_params.json next to this file, or from the path in the FLIPPER_PARAMS
environment variable. See README.md.
"""
import json
import os
from pathlib import Path
import sys
import numpy as np

try:
    from flipper_model_control import controller_from_params
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from turtle_dynamixel.flipper_model_control import controller_from_params

PARAMS_PATH = Path(os.environ.get("FLIPPER_PARAMS", Path(__file__).with_name("flipper_params.json")))
_flippers = None


def load_params(path=None):
    """(Re)loads the parameter file and builds one controller per enabled flipper."""
    global _flippers
    path = PARAMS_PATH if path is None else Path(path)
    params = json.loads(Path(path).read_text())
    _flippers = {}
    for side, flipper in params["flippers"].items():
        if flipper.get("enabled", False):
            _flippers[side] = controller_from_params(params, side)
    return _flippers


def torque_control(q, dq, qd, dqd, ddqd, Kp=None, KD=None):
    """
    Arguments:
        q, dq: measured motor positions (rad) and velocities (rad/s), shape (6, 1) or (6,)
        qd, dqd, ddqd: desired motor positions, velocities and accelerations, same shape
        Kp, KD: feedback gains in the motors' convention, shape (6, 6). None uses the gains in flipper_params.json.
    Returns:
        Goal Current register steps as an int array of shape (6, 1). Rows of disabled flippers are zero.
    """
    flippers = load_params() if _flippers is None else _flippers
    q, dq, qd, dqd, ddqd = (np.asarray(x, dtype=float).reshape(-1) for x in (q, dq, qd, dqd, ddqd))
    steps = np.zeros((6, 1), dtype=int)
    for controller, joint_map, flipper in flippers.values():
        slots = np.asarray(flipper["slots"])
        kp = None if Kp is None else joint_map.gains_to_model(np.asarray(Kp)[np.ix_(slots, slots)])
        kd = None if KD is None else joint_map.gains_to_model(np.asarray(KD)[np.ix_(slots, slots)])
        tau, _ = controller.torque(
            joint_map.position_to_model(q[slots]),
            joint_map.velocity_to_model(dq[slots]),
            joint_map.position_to_model(qd[slots]),
            joint_map.velocity_to_model(dqd[slots]),
            joint_map.velocity_to_model(ddqd[slots]),
            kp=kp,
            kd=kd,
        )
        steps[slots, 0] = controller.actuator.torque_to_steps(joint_map.velocity_to_model(tau))
    return steps
