#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
flipper_trajectory.py  —  Automated Trajectory Edition
-------------------------------------------------------
Moves each motor of the sea turtle flipper through an automated
position trajectory using the computed torque + PD controller.

Trajectory for each motor (executed one motor at a time):

    home ──► lower_limit ──► upper_limit ──► home

Dense intermediate setpoints are generated between each bound using
cosine interpolation for smooth, jerk-minimised motion. Desired
velocity and acceleration profiles are derived analytically from the
position trajectory and fed directly into the torque controller as
feedforward terms.

Sequence:
    Motor 1 (X) completes full trajectory
    Motor 2 (Y) completes full trajectory
    Motor 3 (Z) completes full trajectory
    All motors return to home -> automatic shutdown

Emergency stop:
    Press ESC at any time to immediately zero all currents and shut down.
    Any trajectory data recorded before ESC is plotted automatically.

File dependencies:
    Mod.py            — GroupSyncRead / GroupSyncWrite
    Constants.py      — port, baud rate, hardware constants
    torque_control.py — computed torque + PD controller
    mass.py           — dynamic model (M, C, G matrices)
    rft_sensor.py     — Robotous RFT60-HA01 (USB) driver + logger + live plot
"""
import os
from datetime import datetime
import csv
import sys
import time
import math
import json
from pathlib import Path
import threading
import keyboard
import numpy as np
import matplotlib.pyplot as plt

from dynamixel_sdk import *
from Mod import *
from Constants import *
from torque_control import torque_control

from rft_sensor import RFTSensor, CHANNELS

# ══════════════════════════════════════════════════════════════════════════════
#  EMERGENCY STOP
# ══════════════════════════════════════════════════════════════════════════════

class EmergencyStop(Exception):
    """
    Raised at any point in the program when ESC is pressed.
    Caught in main() which guarantees shutdown() is always called.
    """
    pass


# Threading event — set directly by keyboard hotkey callback.
# Checked at the top of every control loop iteration.
_estop_flag = threading.Event()


def _on_esc() -> None:
    """
    Callback registered with keyboard.add_hotkey().
    Called directly by the keyboard library's internal thread.
    No extra thread required — reuses keyboard's existing monitor.
    """
    _estop_flag.set()
    print("\n[ESTOP] *** ESC detected — emergency stop triggered! ***")


def setup_estop() -> None:
    """
    Register the ESC hotkey callback with the keyboard library.
    Uses keyboard's existing internal thread — no new thread created.
    suppress=True prevents ESC from propagating to the terminal.
    """
    keyboard.add_hotkey('esc', _on_esc, suppress=True)
    print("[OK] Emergency stop monitor active (press ESC at any time).\n")


def teardown_estop() -> None:
    """
    Unregister the ESC hotkey on exit.
    Prevents the callback from firing after the program ends.
    """
    try:
        keyboard.remove_hotkey('esc')
    except Exception:
        pass    # already removed or never registered — safe to ignore


def is_estop() -> bool:
    """Return True if ESC has been pressed."""
    return _estop_flag.is_set()


def handle_estop(mod: 'Mod') -> None:
    """
    Immediately zero all motor currents.
    Always called before raising EmergencyStop so motors are
    safe before the exception unwinds the call stack.
    """
    print("[ESTOP] Zeroing all motor currents immediately ...")
    try:
        mod.send_torque_cmd([HOLD_CURRENT] * N_MOTORS)
    except Exception as e:
        print(f"[ESTOP] Warning — could not zero currents: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Motor IDs ─────────────────────────────────────────────────────────────────
MOTOR_ID_X = 1          # Controls rotation around x-axis (roll)
MOTOR_ID_Y = 2          # Controls rotation around y-axis (pitch)
MOTOR_ID_Z = 3          # Controls rotation around z-axis (yaw)
ALL_IDS    = [MOTOR_ID_X, MOTOR_ID_Y, MOTOR_ID_Z]
N_MOTORS   = len(ALL_IDS)
N_DOF      = 6          # full system DOF — second flipper padded with zeros

# ── Home Positions in RADIANS ───────────────────────────────────────────────
# Home positions are set at startup and used for soft limits and collision checks.
HOME_POSITIONS_RAD = [(16/9)*math.pi, 1*math.pi, 2*math.pi]  

# ── Unit conversions ──────────────────────────────────────────────────────────
STEPS_TO_RAD = (2 * math.pi) / 4096
RAD_TO_STEPS = 4096 / (2 * math.pi)

# ── Position limits (radians) ─────────────────────────────────────────────────
MAX_DELTA_RAD      = 114 * 2 * STEPS_TO_RAD    # ±10° x 2 soft travel limit
RAMP_ZONE_RAD      =  50 * STEPS_TO_RAD    # deceleration band near limit
HOME_THRESHOLD_RAD = DXL_MOVING_STATUS_THRESHOLD * STEPS_TO_RAD

# ── Current / hardware limits ─────────────────────────────────────────────────
# XW540-T260: 2.69 mA per register step
CURRENT_UNIT_MA    = 2.69
CURRENT_CEILING_MA = 300
CURRENT_CEILING    = int(CURRENT_CEILING_MA / CURRENT_UNIT_MA)  # ≈ 186 steps
HOLD_CURRENT       = 0

# ── PD Controller gains ───────────────────────────────────────────────────────
'''
# For manual input not using json

KP_VAL = 4        # proportional gain (Nm/rad)
KD_VAL = 0.3        # derivative gain   (Nm·s/rad)
Kp = np.diag([KP_VAL] * N_DOF)
Kd = np.diag([KD_VAL] * N_DOF)
'''

# ── Trajectory parameters ─────────────────────────────────────────────────────
LOOP_HZ          = 100
LOOP_DT          = 1.0 / LOOP_HZ
SEGMENT_DURATION = 1.0                              # seconds per segment
N_SEGMENT_POINTS = int(SEGMENT_DURATION * LOOP_HZ) # setpoints per segment
STATUS_INTERVAL  = 0.5                              # status print interval (s)

# ── Force-Torque Sensor Configuration ────────────────────────────────────────
FT_PORT        = "COM4"         # Change to your F/T sensor COM port
FT_BAUD        = 115200
FT_RATE_HZ     = LOOP_HZ * 2    # 2× control loop for good coverage
FT_FILTER_HZ   = 50             # 50 Hz low-pass — removes high-freq vibration

# ── Homing ────────────────────────────────────────────────────────────────────
HOME_NUDGE_VEL      = 20
HOME_SETTLE_TIMEOUT = 5.0

# ── Display ───────────────────────────────────────────────────────────────────
MOTOR_LABELS = {
    MOTOR_ID_X: "X-axis (forward/backward)",
    MOTOR_ID_Y: "Y-axis (lateral)",
    MOTOR_ID_Z: "Z-axis (vertical)",
}

PHASE_LABELS = [
    "home -> lower limit",
    "lower limit -> upper limit",
    "upper limit -> home",
]

# ════════════════════════════════════════════════════════════════════════════
# FLIPPER PARAMETER LOADING
# ════════════════════════════════════════════════════════════════════════════

PARAMS_PATH = Path(os.environ.get("FLIPPER_PARAMS", Path(__file__).with_name("flipper_params.json")))

# If wanting to use this method, place in main loop
def load_flipper_params(path=None) -> dict:
    """
    Load flipper controller parameters from flipper_params.json.

    Uses the same path resolution as torque_control.py:
        Path(__file__).with_name("flipper_params.json")
    so the file is always found relative to flipper_trajectory.py
    regardless of the working directory Python was launched from.

    The returned dict recreates the JSON exactly:
        - Outer parameters (kp, kd, medium, gravity, etc.)
          are extracted OUTSIDE the flipper loop.
        - Flipper-specific parameters (slots, offset_rad, sign, actuator)
          are extracted INSIDE the flipper loop — no outer parameters
          are repeated there.

    Returns
    -------
    dict  — all parameters from the JSON, or None if file not found.
    """
    resolved_path = PARAMS_PATH if path is None else Path(path)

    if not resolved_path.is_file():
        print(f"[PARAMS] WARNING — {resolved_path} not found. "
              f"Using hardcoded defaults.")
        return None

    raw = json.loads(resolved_path.read_text())

    # ── Outer parameters — outside flippers:{} in the JSON ───────────────────
    # These apply to the whole system regardless of which flipper is enabled.
    # Extracted here, NOT inside the flipper loop below.
    params = {
        'backend'         : raw.get('backend',           'closed_form'),
        'mode'            : raw.get('mode',              'pd_plus'),
        'medium'          : raw.get('medium',            'air'),
        'gravity'         : np.array(raw.get('gravity',
                            [0.0, 0.0, -9.81]),           dtype=float),
        'buoyancy_scale'  : float(raw.get('buoyancy_scale',  1.0)),
        'added_mass_scale': float(raw.get('added_mass_scale', 1.0)),
        'kp'              : np.array(raw['kp'],           dtype=float),
        'kd'              : np.array(raw['kd'],           dtype=float),
    }

    # ── Flipper-specific parameters — inside flippers:{} in the JSON ──────────
    # Only parameters that belong to the enabled flipper's own block.
    # Outer parameters (kp, kd, medium, etc.) are NOT repeated here.
    for side, flipper in raw['flippers'].items():
        if not flipper.get('enabled', False):
            continue

        act = flipper['actuator']

        params.update({
            # Flipper identity
            'flipper_key'     : side,
            'slots'           : list(flipper['slots']),

            # Joint mapping
            'offset_rad'      : np.array(flipper['offset_rad'], dtype=float),
            'sign'            : np.array(flipper['sign'],        dtype=float),

            # Hydrodynamics (None in air — fit_flipper_params.py fills this)
            'quadratic_drag'  : flipper.get('quadratic_drag'),

            # Actuator model
            'kt'              : np.array(act['kt'],               dtype=float),
            'current_unit_ma' : float(act['current_unit_ma']),
            'current_limit_ma': np.array(act['current_limit_ma'], dtype=float),
            'armature'        : np.array(act['armature'],         dtype=float),
            'viscous'         : np.array(act['viscous'],          dtype=float),
            'coulomb'         : np.array(act['coulomb'],          dtype=float),
            'coulomb_velocity': float(act['coulomb_velocity']),
        })

        print(
            f"\n[PARAMS] Loaded from : {resolved_path}"
            f"\n  Flipper      : {side}  (slots {params['slots']})"
            f"\n  Backend      : {params['backend']}"
            f"\n  Mode         : {params['mode']}"
            f"\n  Medium       : {params['medium']}"
            f"\n  Kp (Nm/rad)  : {params['kp']}"
            f"\n  Kd (Nm*s/r)  : {params['kd']}"
            f"\n  Kt (Nm/A)    : {params['kt']}"
            f"\n  I_lim (mA)   : {params['current_limit_ma']}"
            f"\n  Offset (rad) : {np.round(params['offset_rad'], 4)}"
            f"\n  Sign         : {params['sign']}"
            f"\n  Gravity      : {params['gravity']}\n"
        )

        # Return immediately after the first enabled flipper is found
        # Matches the behaviour of torque_control.py load_params()
        return params

    print("[PARAMS] WARNING — No enabled flipper found in flipper_params.json.")
    return None

# ── Model Parameters ─────────────────────────────────────────────────────────
# Manual update with json until load_flipper_params() is integrated into main loop
# Left flipper  → slots [0, 1, 2]
# Right flipper → slots [3, 4, 5]
# Kp → [5.0, 4.0, 5.0]
# Kd → [0.3, 0.3, 0.4]

# For mismatch of motor ids and slots currently
MOTOR_SLOTS = [3, 4, 5]  # Right flipper slots for torque_control()

# Specifically for printing in graph titles and logs, not for controller input
KP = [5.0, 4.0, 5.0]  # Proportional gains (Nm/rad)
KD = [0.3, 0.3, 0.4]  # Derivative gains (Nm·s/rad)

print(f"[PARAMS] Active flipper slots: {MOTOR_SLOTS}")


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees for display."""
    return round(math.degrees(rad), 2)


def to_unsigned_16(val: int) -> int:
    """
    Convert signed int to unsigned 16-bit two's complement.
    Required because GroupSyncWrite expects unsigned values.
    e.g. -56 -> 65536 + (-56) = 65480
    """
    if val < 0:
        return 65536 + val
    return val


def clamp(value: int, lo: int, hi: int) -> int:
    """Clamp integer between lo and hi."""
    return max(lo, min(hi, value))


def clamp_torque_steps(steps: int) -> int:
    """Clamp signed torque register steps to CURRENT_CEILING. Preserves sign."""
    magnitude = min(abs(steps), CURRENT_CEILING)
    return magnitude if steps >= 0 else -magnitude


def build_state_vector(values: list) -> np.ndarray:
    """
    Build a (6,1) column vector from 3 motor values.
    Indices 3-5 (second flipper) are padded with zeros.
    """
    vec = np.zeros((N_DOF, 1))
    for slot_idx, v in zip(MOTOR_SLOTS, values):
        # Place each motor value at its correct slot position
        vec[slot_idx][0] = v
    return vec


def ramp_factor(pos: float, motor_lims: dict, direction: int) -> float:
    """
    Direction-aware ramp scalar [0.0, 1.0].
    Ramps only toward the limit being approached.
    """
    dist = (
        motor_lims['max'] - pos if direction > 0
        else pos - motor_lims['min']
    )
    if dist <= 0:
        return 0.0
    if dist >= RAMP_ZONE_RAD:
        return 1.0
    return dist / RAMP_ZONE_RAD

# ══════════════════════════════════════════════════════════════════════════════
#  TRAJECTORY GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def cosine_segment(start: float, end: float, n_points: int) -> np.ndarray:
    """
    Smooth position profile from start to end using cosine interpolation.
    Zero velocity at both endpoints.

    q(t) = start + (end - start) x (1 - cos(π·t)) / 2
    """
    t     = np.linspace(0.0, 1.0, n_points)
    alpha = (1.0 - np.cos(np.pi * t)) / 2.0
    return start + alpha * (end - start)


def generate_trajectory(home: float, lower: float,
                        upper: float) -> tuple:
    """
    Generate full motor trajectory with position, velocity, and
    acceleration profiles.

    Path: home ──► lower ──► upper ──► home
    Each segment uses cosine interpolation over N_SEGMENT_POINTS.

    Cosine derivatives:
        q(t)   = start + (end-start)(1-cos(πt))/2
        q̇(t)   = (end-start)·π·sin(πt) / (2T)
        q̈(t)   = (end-start)·π²·cos(πt) / (2T²)

    Returns
    -------
    q           : (N,) position     (rad)
    dq          : (N,) velocity     (rad/s)
    ddq         : (N,) acceleration (rad/s²)
    phase_indices : list of start indices per phase
    """
    T = SEGMENT_DURATION
    N = N_SEGMENT_POINTS

    segments_bounds = [
        (home,  lower),
        (lower, upper),
        (upper, home),
    ]

    q_parts   = []
    dq_parts  = []
    ddq_parts = []
    phase_indices = []
    current_idx   = 0

    for (start, end) in segments_bounds:
        phase_indices.append(current_idx)
        t = np.linspace(0.0, 1.0, N)

        q_seg   = start + (end - start) * (1.0 - np.cos(np.pi * t)) / 2.0
        dq_seg  = (end - start) * np.pi * np.sin(np.pi * t) / (2.0 * T)
        ddq_seg = (end - start) * (np.pi ** 2) * np.cos(np.pi * t) / (2.0 * T ** 2)

        q_parts.append(q_seg)
        dq_parts.append(dq_seg)
        ddq_parts.append(ddq_seg)
        current_idx += N

    return (
        np.concatenate(q_parts),
        np.concatenate(dq_parts),
        np.concatenate(ddq_parts),
        phase_indices,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init_port():
    """Open serial port. Exits on failure."""
    port   = portHandlerJoint
    packet = packetHandlerJoint

    if not port.openPort():
        print("[ERROR] Cannot open port — check USB cable and COM port.")
        sys.exit(1)
    print(f"[OK] Port opened : {JOINTS}")

    if not port.setBaudRate(BAUDRATE):
        print("[ERROR] Cannot set baud rate.")
        port.closePort()
        sys.exit(1)
    print(f"[OK] Baud rate   : {BAUDRATE}")

    return port, packet


def init_mod(port, packet) -> 'Mod':
    """Create Mod, switch to current control, enable torque."""
    print(f"\n[INIT] Initialising Mod with motors {ALL_IDS} ...")
    mod = Mod(packet, port, ALL_IDS)
    mod.disable_torque()
    mod.set_current_cntrl_mode()
    mod.enable_torque()
    mod.send_torque_cmd([HOLD_CURRENT] * N_MOTORS)
    print("[OK] Mod ready in current-control mode.\n")
    return mod


def read_all_positions(mod: 'Mod', last_positions: dict) -> dict:
    """GroupSyncRead positions (rad). Falls back on failure."""
    try:
        pos_list = mod.get_position()
        if pos_list is None:
            return last_positions
        result = {}
        for i, mid in enumerate(ALL_IDS):
            result[mid] = (
                pos_list[i] if (i < len(pos_list) and pos_list[i] is not None)
                else last_positions.get(mid, 0.0)
            )
        return result
    except Exception as e:
        print(f"[WARNING] Position read exception ({e})")
        return last_positions


def read_all_velocities(mod: 'Mod', last_velocities: dict) -> dict:
    """GroupSyncRead velocities (rad/s). Falls back on failure."""
    try:
        vel_list = mod.get_velocity()
        if vel_list is None:
            return last_velocities
        result = {}
        for i, mid in enumerate(ALL_IDS):
            result[mid] = (
                vel_list[i] if (i < len(vel_list) and vel_list[i] is not None)
                else last_velocities.get(mid, 0.0)
            )
        return result
    except Exception as e:
        print(f"[WARNING] Velocity read exception ({e})")
        return last_velocities


def init_dynamic_limits(mod: 'Mod') -> dict:
    """Read startup positions and build per-motor soft limits (rad)."""
    all_limits = {}
    positions  = read_all_positions(mod, {mid: 0.0 for mid in ALL_IDS})
    for mid in ALL_IDS:
        #home = positions[mid]
        all_limits[mid] = {
            'home' : HOME_POSITIONS_RAD[mid - 1],  # use predefined home positions
            'lower': HOME_POSITIONS_RAD[mid - 1] - MAX_DELTA_RAD,
            'upper': HOME_POSITIONS_RAD[mid - 1] + MAX_DELTA_RAD,
            'min'  : HOME_POSITIONS_RAD[mid - 1] - MAX_DELTA_RAD,
            'max'  : HOME_POSITIONS_RAD[mid - 1] + MAX_DELTA_RAD,
        }
        print(
            f"[LIMITS] Motor {mid} ({MOTOR_LABELS[mid]}) — "
            f"Home: {rad_to_deg(HOME_POSITIONS_RAD[mid - 1]):>8.2f}°  "
            f"Lower: {rad_to_deg(HOME_POSITIONS_RAD[mid - 1] - MAX_DELTA_RAD):>8.2f}°  "
            f"Upper: {rad_to_deg(HOME_POSITIONS_RAD[mid - 1] + MAX_DELTA_RAD):>8.2f}°"
        )
    return all_limits

def init_ft_sensor() -> RFTSensor:
    """
    Initialise, configure, and tare the RFT60-HA01 F/T sensor.
    Returns a streaming RFTSensor instance ready to collect data.

    The sensor is attached at the base of Motor 1. It measures the
    total reaction wrench of the entire 3-motor flipper chain:

        Fx, Fy, Fz : reaction forces  (N)
        Tx, Ty, Tz : reaction torques (N·m)

    Taring removes gravity + static preload so only dynamic forces
    from motor motion are recorded.
    """
    print(f"\n[F/T SENSOR] Connecting to {FT_PORT} @ {FT_BAUD} baud ...")
    sensor = RFTSensor(FT_PORT, FT_BAUD)

    model, sn, fw = sensor.info()
    print(f"[F/T SENSOR] Model: {model}  S/N: {sn}  FW: {fw}")
    print(f"[F/T SENSOR] DF={sensor.df:.0f}  DT={sensor.dt:.0f}")

    sensor.configure(rate_hz=FT_RATE_HZ, filter_hz=FT_FILTER_HZ)
    sensor.start()
    time.sleep(0.3)
    sensor.drain()              # discard startup transient

    # ── Software tare — remove static preload ────────────────────────────────
    print("[F/T SENSOR] Taring — keep motors STATIONARY ...")
    tare_samples = sensor.collect(n=200, timeout=5.0)
    if not tare_samples:
        raise RuntimeError("[F/T SENSOR] No data received during tare — "
                           "check USB connection and COM port.")

    tare_arr           = np.array([s.ft for s in tare_samples])
    sensor.tare_offset = tare_arr.mean(axis=0)     # (6,) offset vector
    sensor.tare_std    = tare_arr.std(axis=0)

    print(
        f"[F/T SENSOR] Tare offset  (N / N·m): "
        f"{np.round(sensor.tare_offset, 4)}"
    )
    print(
        f"[F/T SENSOR] Noise std    (N / N·m): "
        f"{np.round(sensor.tare_std, 4)}"
    )
    print("[F/T SENSOR] Ready.\n")
    return sensor


def close_ft_sensor(sensor: RFTSensor) -> None:
    """Stop streaming and close the F/T sensor connection."""
    print("[F/T SENSOR] Shutting down ...")
    sensor.close()
    print("[F/T SENSOR] Closed.\n")


def home_all_motors(mod: 'Mod', port, packet,
                    all_limits: dict) -> dict:
    """
    Switch to Extended Position Mode, drive all motors to home,
    then return to Current Control Mode.
    Raises EmergencyStop if ESC is pressed during homing.
    """
    print("\n[HOME] Switching to position mode for homing ...")
    mod.disable_torque()

    for mid in ALL_IDS:
        result, _ = packet.write1ByteTxRx(
            port, mid, ADDR_OPERATING_MODE, EXT_POSITION_CONTROL_MODE
        )
        if result != COMM_SUCCESS:
            print(f"[ERROR] Motor {mid} mode switch failed.")

    mod.set_max_velocity(HOME_NUDGE_VEL)
    mod.enable_torque()

    for mid in ALL_IDS:
        home_steps = int(all_limits[mid]['home'] * RAD_TO_STEPS)
        result, _ = packet.write4ByteTxRx(
            port, mid, ADDR_GOAL_POSITION, home_steps
        )
        if result != COMM_SUCCESS:
            print(f"[ERROR] Motor {mid} home command failed.")

    positions = {mid: all_limits[mid]['home'] for mid in ALL_IDS}
    deadline  = time.time() + HOME_SETTLE_TIMEOUT

    while time.time() < deadline:

        #   Check ESC every iteration of the settle loop
        if is_estop():
            handle_estop(mod)
            raise EmergencyStop("ESC pressed during homing.")

        positions = read_all_positions(mod, positions)
        if all(
            abs(positions[mid] - all_limits[mid]['home'])
            <= HOME_THRESHOLD_RAD for mid in ALL_IDS
        ):
            break
        time.sleep(LOOP_DT)

    print("[HOME] Switching back to current control ...")
    mod.disable_torque()
    for mid in ALL_IDS:
        packet.write1ByteTxRx(
            port, mid, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE
        )
    mod.enable_torque()
    mod.send_torque_cmd([HOLD_CURRENT] * N_MOTORS)
    positions = read_all_positions(mod, positions)
    print("[HOME] Done.\n")
    return positions


# ══════════════════════════════════════════════════════════════════════════════
#  SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

def shutdown(mod: 'Mod', port) -> None:
    """Zero currents, disable torques, close port."""
    print("\n[SHUTDOWN] Zeroing currents ...")
    mod.send_torque_cmd([HOLD_CURRENT] * N_MOTORS)
    time.sleep(0.1)
    print("[SHUTDOWN] Disabling torques ...")
    mod.disable_torque()
    port.closePort()
    print("[SHUTDOWN] Port closed. Goodbye!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  TORQUE COMMAND PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def compute_single_motor_torque(active_mid: int,
                                positions: dict,
                                velocities: dict,
                                qd_active: float,
                                dqd_active: float,
                                ddqd_active: float,
                                all_limits: dict) -> dict:
    """
    Compute torque commands for one active motor using the computed
    torque + PD controller. All inactive motors receive zero current.

    Parameters
    ----------
    active_mid    : motor ID being driven this step
    positions     : { motor_id: rad }   current positions
    velocities    : { motor_id: rad/s } current velocities
    qd_active     : desired position    for active motor (rad)
    dqd_active    : desired velocity    for active motor (rad/s)
    ddqd_active   : desired accel       for active motor (rad/s²)
    all_limits    : { motor_id: limit dict }

    Returns
    -------
    { motor_id: signed int steps }
    """
    active_idx = ALL_IDS.index(active_mid)
    active_slot = MOTOR_SLOTS[active_idx]

    # Inactive motors track their current position -> zero tracking error
    qd_list = [
        qd_active if mid == active_mid else positions[mid]
        for mid in ALL_IDS
    ]

    # Build (6,1) state vectors
    q_vec    = build_state_vector([positions[mid]  for mid in ALL_IDS])
    dq_vec   = build_state_vector([velocities[mid] for mid in ALL_IDS])
    qd_vec   = build_state_vector(qd_list)

    # Feedforward velocity and acceleration — active motor only
    dqd_vec                    = np.zeros((N_DOF, 1))
    ddqd_vec                   = np.zeros((N_DOF, 1))
    dqd_vec[active_slot][0]     = dqd_active
    ddqd_vec[active_slot][0]    = ddqd_active

    # Test zeroes for position tracking against gravity. Want to stay stable
    zeroes6 = np.zeros((6,6))
    # Call torque controller
    try:
        tau_steps = torque_control(
            q_vec, dq_vec, qd_vec, dqd_vec, ddqd_vec, Kp=None, KD=None
        )
    except Exception as e:
        print(f"[ERROR] torque_control() failed: {e} — zeroing all currents.")
        return {mid: 0 for mid in ALL_IDS}

    # Build per-motor command dict with safety checks
    torque_cmds = {}
    for i, mid in enumerate(ALL_IDS):

        # Inactive motors always get zero
        if mid != active_mid:
            torque_cmds[mid] = 0
            continue

        slot      = MOTOR_SLOTS[i]
        raw        = int(tau_steps[slot][0])
        direction  = +1 if raw >= 0 else -1
        pos        = positions[mid]
        motor_lims = all_limits[mid]

        # Per-motor hard limit check
        at_min = (pos <= motor_lims['min'] and direction < 0)
        at_max = (pos >= motor_lims['max'] and direction > 0)

        if at_min or at_max:
            torque_cmds[mid] = 0
            print(
                f"[LIMIT] Motor {mid} at soft limit "
                f"({rad_to_deg(pos):.1f}°) — torque zeroed."
            )
        else:
            factor           = ramp_factor(pos, motor_lims, direction)
            scaled           = int(raw * factor)
            torque_cmds[mid] = clamp_torque_steps(scaled)

    return torque_cmds


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE MOTOR TRAJECTORY EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def run_motor_trajectory(mod: 'Mod',
                         motor_id: int,
                         trajectory_q:   np.ndarray,
                         trajectory_dq:  np.ndarray,
                         trajectory_ddq: np.ndarray,
                         phase_indices:  list,
                         positions:      dict,
                         velocities:     dict,
                         all_limits:     dict,
                         sensor:         'RFTSensor' = None) -> tuple:
    """
    Execute a full position trajectory for one motor.
    Records position, torque, and F/T sensor data each loop iteration.

    F/T sensor note:
        The sensor runs at FT_RATE_HZ (200 Hz) independently of the
        100 Hz control loop. Each loop iteration drains all new F/T
        samples and records their mean to align with the control timestamp.
        This gives one averaged F/T reading per control step.
    """
    n_steps          = len(trajectory_q)
    current_phase    = 0
    last_status_time = 0.0
    start_time       = time.time()

    # ── Preallocate control loop recording arrays ──────────────────────────────
    log_time     = np.zeros(n_steps)
    log_q_actual = np.zeros(n_steps)
    log_q_des    = np.zeros(n_steps)
    log_q_error  = np.zeros(n_steps)
    log_dq_actual = np.zeros(n_steps)
    log_dq_des    = np.zeros(n_steps)
    log_dq_error  = np.zeros(n_steps)
    log_torque   = np.zeros(n_steps)
    phase_times  = []

    # ── Preallocate F/T recording arrays — one row per control step ───────────
    # Columns: [Fx, Fy, Fz, Tx, Ty, Tz]  (N and N·m, tare-corrected)
    log_ft = np.zeros((n_steps, 6))     # F/T data aligned to control steps

    print(
        f"\n{'═'*65}\n"
        f"  MOTOR {motor_id} — {MOTOR_LABELS[motor_id]}\n"
        f"  Total steps : {n_steps}  ({n_steps * LOOP_DT:.1f} s)\n"
        f"  F/T sensor  : {'ACTIVE' if sensor else 'NOT CONNECTED'}\n"
        f"{'═'*65}"
    )

    home  = all_limits[motor_id]['home']
    lower = all_limits[motor_id]['lower']
    upper = all_limits[motor_id]['upper']
    print(
        f"  Home:  {rad_to_deg(home):>8.2f}°\n"
        f"  Lower: {rad_to_deg(lower):>8.2f}°\n"
        f"  Upper: {rad_to_deg(upper):>8.2f}°\n"
    )

    for step in range(n_steps):
        step_start = time.time()
        elapsed    = step_start - start_time

        # ── ESC check ─────────────────────────────────────────────────────────
        if is_estop():
            handle_estop(mod)
            log = {
                'time_s'      : log_time[:step],
                'q_actual'    : log_q_actual[:step],
                'q_desired'   : log_q_des[:step],
                'q_error'     : log_q_error[:step],
                'dq_actual'   : log_dq_actual[:step],
                'dq_desired'  : log_dq_des[:step],
                'dq_error'    : log_dq_error[:step],
                'torque'      : log_torque[:step],
                'ft'          : log_ft[:step],          
                'phase_times' : phase_times,
                'motor_id'    : motor_id,
                'label'       : MOTOR_LABELS[motor_id],
                'aborted'     : True,
                'aborted_step': step,
            }
            raise EmergencyStop(
                f"ESC pressed during Motor {motor_id} trajectory "
                f"at step {step}/{n_steps} (t={elapsed:.2f}s)."
            )

        # ── Phase tracking ────────────────────────────────────────────────────
        for p_idx in range(len(phase_indices) - 1, -1, -1):
            if step >= phase_indices[p_idx]:
                if p_idx != current_phase:
                    current_phase = p_idx
                    phase_times.append(elapsed)
                    print(
                        f"\n  [PHASE {current_phase + 1}/3] "
                        f"{PHASE_LABELS[current_phase]}  "
                        f"(step {step}/{n_steps}  t={elapsed:.2f}s)"
                    )
                break

        if step == 0:
            phase_times.append(0.0)
            print(
                f"\n  [PHASE 1/3] {PHASE_LABELS[0]}  "
                f"(step 0/{n_steps}  t=0.00s)"
            )

        # ── Read motor state ──────────────────────────────────────────────────
        positions  = read_all_positions(mod, positions)
        velocities = read_all_velocities(mod, velocities)

        # ── Drain all new F/T samples and average them ──────────────────────
        # The sensor produces FT_RATE_HZ / LOOP_HZ ≈ 2 samples per loop step.
        # Averaging gives one aligned, noise-reduced reading per control step.
        if sensor is not None:
            ft_batch = sensor.drain()
            if ft_batch:
                ft_raw  = np.array([s.ft for s in ft_batch])
                ft_mean = ft_raw.mean(axis=0) - sensor.tare_offset
            else:
                # No new samples this step — carry forward last reading
                ft_mean = log_ft[step - 1] if step > 0 else np.zeros(6)
            log_ft[step] = ft_mean
        else:
            log_ft[step] = np.zeros(6)

        # ── Compute and send torque ───────────────────────────────────────────
        qd_step   = trajectory_q[step]
        dqd_step  = trajectory_dq[step]
        ddqd_step = trajectory_ddq[step]

        torque_cmds = compute_single_motor_torque(
            motor_id, positions, velocities,
            qd_step, dqd_step, ddqd_step,
            all_limits
        )
        mod.send_torque_cmd([
            to_unsigned_16(torque_cmds[mid]) for mid in ALL_IDS
        ])

        # ── Record motor data ─────────────────────────────────────────────────
        pos_actual         = positions[motor_id]
        vel_actual         = velocities[motor_id]
        q_error            = qd_step - pos_actual
        dq_error           = dqd_step - vel_actual
    
        log_time[step]     = elapsed
        log_q_actual[step] = rad_to_deg(pos_actual)
        log_q_des[step]    = rad_to_deg(qd_step)
        log_q_error[step]  = rad_to_deg(q_error)
        log_dq_actual[step] = rad_to_deg(vel_actual)
        log_dq_des[step]    = rad_to_deg(dqd_step)
        log_dq_error[step]  = rad_to_deg(dq_error)
        log_torque[step]   = torque_cmds[motor_id]

        # ── Periodic status print ─────────────────────────────────────────────
        now = time.time()
        if now - last_status_time >= STATUS_INTERVAL:
            pct = 100.0 * step / max(n_steps - 1, 1)
            ft  = log_ft[step]
            print(
                f"    step {step:>4}/{n_steps} ({pct:>5.1f}%) "
                f"t={elapsed:>5.1f}s | "
                f"q={rad_to_deg(pos_actual):>8.2f}° "
                f"q_err={rad_to_deg(q_error):>+7.2f}° "
                f"dq={rad_to_deg(vel_actual):>+6.2f}deg/s "
                f"dq_err={rad_to_deg(dq_error):>+6.2f}deg/s "
                f"I={torque_cmds[motor_id]:>4} | "
                #f"Fz={ft[2]:>+6.2f}N "
                #f"Tz={ft[5]:>+6.3f}N·m"
            )
            last_status_time = now

        # ── Pace loop ─────────────────────────────────────────────────────────
        elapsed_step = time.time() - step_start
        sleep_t      = LOOP_DT - elapsed_step
        if sleep_t > 0:
            time.sleep(sleep_t)

    mod.send_torque_cmd([HOLD_CURRENT] * N_MOTORS)
    total_time = time.time() - start_time
    print(
        f"\n  [COMPLETE] Motor {motor_id} done. "
        f"Total time: {total_time:.2f} s"
    )

    log = {
        'time_s'      : log_time,
        'q_actual'    : log_q_actual,
        'q_desired'   : log_q_des,
        'q_error'     : log_q_error,
        'dq_actual'   : log_dq_actual,
        'dq_desired'  : log_dq_des,
        'dq_error'    : log_dq_error,
        'torque'      : log_torque,
        'ft'          : log_ft,             
        'phase_times' : phase_times,
        'motor_id'    : motor_id,
        'label'       : MOTOR_LABELS[motor_id],
        'aborted'     : False,
        'aborted_step': None,
    }

    return positions, velocities, log

# ══════════════════════════════════════════════════════════════════════════════
#  SAVING RESULTS AS CSV
# ══════════════════════════════════════════════════════════════════════════════

def save_results_csv(all_logs: list, Kp: np.array, Kd: np.array) -> str:
    """
    Save all trajectory data for every motor into a single CSV file.

    One row per control loop step per motor. Columns include:
        - Run metadata  : motor_id, motor_label, phase, aborted
        - Time          : time_s
        - Position      : q_actual_deg, q_desired_deg, q_error_deg
        - Velocity      : qd_actual, qd_desired, qd_error
        - Current       : current_steps, current_mA
        - F/T sensor    : Fx_N, Fy_N, Fz_N, Tx_Nm, Ty_Nm, Tz_Nm
        - F/T magnitude : force_mag_N, torque_mag_Nm
        - Gain settings : Kp, Kd

    A second summary CSV is also written with per-motor statistics:
        - RMS and peak values for position error, current, and F/T channels

    Parameters
    ----------
    all_logs : list of log dicts from run_motor_trajectory()

    Returns
    -------
    str — path to the main CSV file
    """
    import os
    from datetime import datetime

    if not all_logs:
        print("[CSV] No data to save.")
        return ""

    # ── Build output folder and base filename ─────────────────────────────────
    RESULTS_DIR = "flipper_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_flipper_trajectory_Kp{Kp}_Kd{Kd}"

    main_csv_path    = os.path.join(RESULTS_DIR, f"{base_name}_data.csv")
    summary_csv_path = os.path.join(RESULTS_DIR, f"{base_name}_summary.csv")

    # ── Phase lookup helper ───────────────────────────────────────────────────
    def get_phase_label(step: int, phase_indices: list) -> str:
        """Return the phase label for a given step index."""
        current_phase = 0
        for p_idx in range(len(phase_indices) - 1, -1, -1):
            if step >= phase_indices[p_idx]:
                current_phase = p_idx
                break
        return PHASE_LABELS[current_phase] if current_phase < len(PHASE_LABELS) else "unknown"

    # ══════════════════════════════════════════════════════════════════════════
    #  MAIN DATA CSV — one row per step per motor
    # ══════════════════════════════════════════════════════════════════════════
    main_headers = [
        # ── Run metadata ──────────────────────────────────────────────────────
        "motor_id", "motor_label", "Kp_Nm_per_rad", "Kd_Nm_s_per_rad",
        "aborted", "phase",

        # ── Time ──────────────────────────────────────────────────────────────
        "time_s",

        # ── Position ──────────────────────────────────────────────────────────
        "q_actual_deg", "q_desired_deg", "q_error_deg",

        # ── Velocity ─────────────────────────────────────────────────────────
        "dq_actual_deg_s", "dq_desired_deg_s", "dq_error_deg_s",

        # ── Commanded current ─────────────────────────────────────────────────
        "current_steps", "current_mA",

        # ── F/T sensor — individual channels ─────────────────────────────────
        "Fx_N", "Fy_N", "Fz_N", "Tx_Nm", "Ty_Nm", "Tz_Nm",

        # ── F/T sensor — resultant magnitudes ─────────────────────────────────
        "force_mag_N", "torque_mag_Nm",
    ]

    total_rows = 0

    with open(main_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(main_headers)

        for log in all_logs:
            mid          = log['motor_id']
            label        = log['label']
            aborted      = log.get('aborted', False)
            phase_idx    = log.get('phase_times', [])
            t            = log['time_s']
            q_act        = log['q_actual']
            q_des        = log['q_desired']
            q_err        = log['q_error']
            dq_act       = log['dq_actual']
            dq_des       = log['dq_desired']
            dq_err       = log['dq_error']
            torque_steps = log['torque']
            ft_data      = log.get('ft', np.zeros((len(t), 6)))

            # ── Ensure ft_data is the right shape ─────────────────────────────
            if ft_data is None or ft_data.ndim != 2 or ft_data.shape[1] != 6:
                ft_data = np.zeros((len(t), 6))

            n_steps = len(t)

            for step in range(n_steps):
                # Current in mA
                current_mA_val = torque_steps[step] * CURRENT_UNIT_MA

                # F/T channels
                fx  = ft_data[step, 0]
                fy  = ft_data[step, 1]
                fz  = ft_data[step, 2]
                tx  = ft_data[step, 3]
                ty  = ft_data[step, 4]
                tz  = ft_data[step, 5]

                # Resultant magnitudes
                force_mag  = math.sqrt(fx**2 + fy**2 + fz**2)
                torque_mag = math.sqrt(tx**2 + ty**2 + tz**2)

                # Phase label for this step
                phase_label = get_phase_label(step, phase_idx)

                writer.writerow([
                    # metadata
                    mid,
                    label,
                    Kp,
                    Kd,
                    int(aborted),
                    phase_label,

                    # time
                    f"{t[step]:.5f}",

                    # position
                    f"{q_act[step]:.4f}",
                    f"{q_des[step]:.4f}",
                    f"{q_err[step]:.4f}",

                    # velocity
                    f"{dq_act[step]:.4f}",
                    f"{dq_des[step]:.4f}",
                    f"{dq_err[step]:.4f}",

                    # current
                    f"{int(torque_steps[step])}",
                    f"{current_mA_val:.4f}",

                    # F/T channels
                    f"{fx:.6f}",
                    f"{fy:.6f}",
                    f"{fz:.6f}",
                    f"{tx:.6f}",
                    f"{ty:.6f}",
                    f"{tz:.6f}",

                    # magnitudes
                    f"{force_mag:.6f}",
                    f"{torque_mag:.6f}",
                ])

                total_rows += 1

    print(f"[CSV] Main data saved  : {main_csv_path}")
    print(f"[CSV] Total rows       : {total_rows}")

    # ══════════════════════════════════════════════════════════════════════════
    #  SUMMARY CSV — one row per motor with statistics
    # ══════════════════════════════════════════════════════════════════════════
    summary_headers = [
        # ── Run metadata ──────────────────────────────────────────────────────
        "motor_id", "motor_label", "Kp_Nm_per_rad", "Kd_Nm_s_per_rad",
        "aborted", "n_steps", "duration_s",

        # ── Position error statistics ─────────────────────────────────────────
        "pos_error_rms_deg", "pos_error_peak_deg", "pos_error_mean_deg", "pos_final_error_deg",

        # ── Velocity error statistics ─────────────────────────────────────────
        "dq_error_rms_rad_s", "dq_error_peak_rad_s", "dq_error_mean_rad_s",

        # ── Current statistics ────────────────────────────────────────────────
        "current_rms_mA", "current_peak_mA", "current_mean_abs_mA",

        # ── F/T force statistics ──────────────────────────────────────────────
        "Fx_rms_N", "Fx_peak_N", "Fy_rms_N", "Fy_peak_N", "Fz_rms_N", "Fz_peak_N",
        "force_mag_rms_N", "force_mag_peak_N",

        # ── F/T torque statistics ─────────────────────────────────────────────
        "Tx_rms_Nm", "Tx_peak_Nm", "Ty_rms_Nm",  "Ty_peak_Nm", "Tz_rms_Nm", "Tz_peak_Nm",
        "torque_mag_rms_Nm", "torque_mag_peak_Nm",
    ]

    with open(summary_csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(summary_headers)

        for log in all_logs:
            mid          = log['motor_id']
            label        = log['label']
            aborted      = log.get('aborted', False)
            t            = log['time_s']
            q_err        = log['q_error']
            dq_err       = log['dq_error']
            torque_steps = log['torque']
            ft_data      = log.get('ft', np.zeros((len(t), 6)))

            if ft_data is None or ft_data.ndim != 2 or ft_data.shape[1] != 6:
                ft_data = np.zeros((len(t), 6))

            n_steps    = len(t)
            duration_s = float(t[-1]) if n_steps > 0 else 0.0

            # ── Position error statistics ─────────────────────────────────────
            err_rms   = float(np.sqrt(np.mean(q_err**2))) if n_steps > 0 else 0.0
            err_peak  = float(np.max(np.abs(q_err)))       if n_steps > 0 else 0.0
            err_mean  = float(np.mean(np.abs(q_err)))      if n_steps > 0 else 0.0
            err_final = float(q_err[-1])                   if n_steps > 0 else 0.0

            # ── Velocity error statistics ─────────────────────────────────────
            dq_err_rms   = float(np.sqrt(np.mean(dq_err**2))) if n_steps > 0 else 0.0
            dq_err_peak  = float(np.max(np.abs(dq_err)))       if n_steps > 0 else 0.0
            dq_err_mean  = float(np.mean(np.abs(dq_err)))      if n_steps > 0 else 0.0

            # ── Current statistics ────────────────────────────────────────────
            cur_mA      = torque_steps * CURRENT_UNIT_MA
            cur_rms     = float(np.sqrt(np.mean(cur_mA**2))) if n_steps > 0 else 0.0
            cur_peak    = float(np.max(np.abs(cur_mA)))       if n_steps > 0 else 0.0
            cur_mean    = float(np.mean(np.abs(cur_mA)))      if n_steps > 0 else 0.0

            # ── F/T channel statistics ────────────────────────────────────────
            def rms(arr):  return float(np.sqrt(np.mean(arr**2))) if len(arr) > 0 else 0.0
            def peak(arr): return float(np.max(np.abs(arr)))      if len(arr) > 0 else 0.0

            force_mag  = np.linalg.norm(ft_data[:, :3], axis=1)
            torque_mag = np.linalg.norm(ft_data[:, 3:], axis=1)

            writer.writerow([
                # metadata
                mid,
                label,
                Kp,
                Kd,
                int(aborted),
                n_steps,
                f"{duration_s:.3f}",

                # position error
                f"{err_rms:.4f}",
                f"{err_peak:.4f}",
                f"{err_mean:.4f}",
                f"{err_final:.4f}",

                # velocity error
                f"{dq_err_rms:.4f}",
                f"{dq_err_peak:.4f}",
                f"{dq_err_mean:.4f}",

                # current
                f"{cur_rms:.4f}",
                f"{cur_peak:.4f}",
                f"{cur_mean:.4f}",

                # F/T forces
                f"{rms(ft_data[:,0]):.6f}",  f"{peak(ft_data[:,0]):.6f}",
                f"{rms(ft_data[:,1]):.6f}",  f"{peak(ft_data[:,1]):.6f}",
                f"{rms(ft_data[:,2]):.6f}",  f"{peak(ft_data[:,2]):.6f}",
                f"{rms(force_mag):.6f}",     f"{peak(force_mag):.6f}",

                # F/T torques
                f"{rms(ft_data[:,3]):.6f}",  f"{peak(ft_data[:,3]):.6f}",
                f"{rms(ft_data[:,4]):.6f}",  f"{peak(ft_data[:,4]):.6f}",
                f"{rms(ft_data[:,5]):.6f}",  f"{peak(ft_data[:,5]):.6f}",
                f"{rms(torque_mag):.6f}",    f"{peak(torque_mag):.6f}",
            ])

    print(f"[CSV] Summary saved    : {summary_csv_path}")
    return main_csv_path

# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(all_logs: list, Kp: np.array, Kd: np.array) -> None:
    """
    Generate two separate figures.

    Figure 1 — Position and Velocity Tracking (n_motors rows × 2 columns):
        Col 1 : actual vs desired position  (deg)   vs time (s)
        Col 2 : actual vs desired velocity  (deg/s) vs time (s)

    Figure 2 — Force, Torque, and Current (n_motors rows × 3 columns):
        Col 1 : Forces  Fx, Fy, Fz + |F| resultant  (N)
        Col 2 : Torques Tx, Ty, Tz + |T| resultant  (N·m)
        Col 3 : Commanded current                    (mA)
    """
    import os
    from datetime import datetime
    from scipy.signal import butter, filtfilt

    n_motors = len(all_logs)
    if n_motors == 0:
        print("[PLOT] No data to plot.")
        return

    RESULTS_DIR = "flipper_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_flipper_trajectory_Kp{Kp}_Kd{Kd}"

    # ── Colour scheme ──────────────────────────────────────────────────────────
    colours = {
        MOTOR_ID_X: '#2196F3',
        MOTOR_ID_Y: '#4CAF50',
        MOTOR_ID_Z: '#FF5722',
    }

    FT_COLOURS = {
        'Fx': '#E53935', 'Fy': '#8E24AA', 'Fz': '#1565C0',
        'Tx': '#F4511E', 'Ty': '#6D4C41', 'Tz': '#00838F',
    }

    FT_PLOT_CUTOFF_HZ = 10.0

    def lowpass_filter(data, cutoff_hz, sample_hz, order=4):
        nyq  = sample_hz / 2.0
        if cutoff_hz >= nyq or len(data) < 15:
            return data
        b, a = butter(order, cutoff_hz / nyq, btype='low')
        if data.ndim == 1:
            return filtfilt(b, a, data)
        return np.column_stack([filtfilt(b, a, data[:, i])
                                for i in range(data.shape[1])])

    def draw_phase_lines(ax, phase_times, t, aborted):
        for pt in phase_times:
            ax.axvline(x=pt, color='grey', linewidth=0.8,
                       linestyle=':', zorder=1)
        if aborted and len(t) > 0:
            ax.axvline(x=t[-1], color='red', linewidth=1.5,
                       linestyle='--', zorder=4)

    def annotate_abort(ax, t, aborted):
        if aborted and len(t) > 0:
            ax.text(
                t[-1], ax.get_ylim()[1],
                ' ESC STOP', fontsize=8,
                color='red', va='top', fontweight='bold'
            )

    def stat_box(ax, text):
        ax.text(
            0.02, 0.95, text,
            transform=ax.transAxes,
            fontsize=8, va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8)
        )

    # ══════════════════════════════════════════════════════════════════════════
    #  FIGURE 1 — POSITION AND VELOCITY TRACKING
    #  n_motors rows × 2 columns
    # ══════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(
        n_motors, 2,
        figsize=(14, 4 * n_motors),
        sharex='row'
    )
    if n_motors == 1:
        axes1 = np.array([axes1])

    fig1.suptitle(
        "Sea Turtle Flipper — Position and Velocity Tracking\n"
        f"Kp={Kp} Nm/rad   Kd={Kd} Nm·s/rad",
        fontsize=13, fontweight='bold', y=1.01
    )

    for row, log in enumerate(all_logs):
        mid         = log['motor_id']
        label       = log['label']
        colour      = colours.get(mid, '#9C27B0')
        t           = log['time_s']
        q_act       = log['q_actual']
        q_des       = log['q_desired']
        q_err       = log['q_error']
        dq_act      = log['dq_actual']
        dq_des      = log['dq_desired']
        dq_err      = log['dq_error']
        phase_times = log['phase_times']
        aborted     = log.get('aborted', False)

        title_suffix = " [ABORTED — ESC]" if aborted else ""
        title_colour = 'red' if aborted else 'black'

        ax_pos = axes1[row, 0]
        ax_vel = axes1[row, 1]

        # ── Col 1: Position Tracking ──────────────────────────────────────────
        ax_pos.plot(t, q_des, color='black', linewidth=1.5,
                    linestyle='--', label='Desired $q_d$', zorder=3)
        ax_pos.plot(t, q_act, color=colour, linewidth=1.5,
                    label='Actual $q$', zorder=2)
        ax_pos.fill_between(t, q_des, q_act, alpha=0.15,
                             color=colour, label='Error region')
        draw_phase_lines(ax_pos, phase_times, t, aborted)
        annotate_abort(ax_pos, t, aborted)

        if len(q_des) > 0:
            for val, lbl, ls in [
                (q_des[0],    'home',  '-'),
                (q_des.min(), 'lower', '--'),
                (q_des.max(), 'upper', '--'),
            ]:
                ax_pos.axhline(y=val, color='grey', linewidth=0.6,
                               linestyle=ls, alpha=0.5)
                ax_pos.text(
                    t[-1] if len(t) > 0 else 0, val,
                    f' {lbl}\n {val:.1f}°',
                    fontsize=7, color='grey', va='center'
                )

        if len(q_err) > 0:
            stat_box(ax_pos,
                f"RMS  : {float(np.sqrt(np.mean(q_err**2))):.3f}°\n"
                f"Peak : {float(np.max(np.abs(q_err))):.3f}°"
            )

        ax_pos.set_title(
            f"Motor {mid} — {label}{title_suffix}\nPosition Tracking",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_pos.set_ylabel("Position (°)", fontsize=9)
        ax_pos.legend(fontsize=8, loc='upper right')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.set_xlim(left=0)

        # ── Col 2: Velocity Tracking ──────────────────────────────────────────
        ax_vel.plot(t, dq_des, color='black', linewidth=1.5,
                    linestyle='--', label='Desired $\dot{q}_d$', zorder=3)
        ax_vel.plot(t, dq_act, color=colour, linewidth=1.5,
                    label='Actual $\dot{q}$', zorder=2)
        ax_vel.fill_between(t, dq_des, dq_act, alpha=0.15,
                             color=colour, label='Error region')
        ax_vel.axhline(y=0, color='black', linewidth=0.6,
                       linestyle='--', alpha=0.4)
        draw_phase_lines(ax_vel, phase_times, t, aborted)
        annotate_abort(ax_vel, t, aborted)

        if len(dq_err) > 0:
            stat_box(ax_vel,
                f"RMS  : {float(np.sqrt(np.mean(dq_err**2))):.3f}°/s\n"
                f"Peak : {float(np.max(np.abs(dq_err))):.3f}°/s"
            )

        ax_vel.set_title(
            f"Motor {mid} — {label}{title_suffix}\nVelocity Tracking",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_vel.set_ylabel("Velocity (°/s)", fontsize=9)
        ax_vel.legend(fontsize=8, loc='upper right')
        ax_vel.grid(True, alpha=0.3)
        ax_vel.set_xlim(left=0)

        if row == n_motors - 1:
            ax_pos.set_xlabel("Time (s)", fontsize=9)
            ax_vel.set_xlabel("Time (s)", fontsize=9)

    plt.figure(fig1.number)
    plt.tight_layout()
    fig1_path = os.path.join(RESULTS_DIR, f"{base_name}_position_velocity.png")
    fig1.savefig(fig1_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Figure 1 saved to: {fig1_path}")

    # ══════════════════════════════════════════════════════════════════════════
    #  FIGURE 2 — FORCE, TORQUE, AND CURRENT
    #  n_motors rows × 3 columns
    # ══════════════════════════════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(
        n_motors, 3,
        figsize=(18, 4 * n_motors),
        sharex='row'
    )
    if n_motors == 1:
        axes2 = np.array([axes2])

    fig2.suptitle(
        "Sea Turtle Flipper — Force, Torque and Current\n"
        f"RFT60-HA01 @ {FT_RATE_HZ} Hz  |  "
        f"F/T filter: {FT_PLOT_CUTOFF_HZ} Hz  |  "
        f"Kp={Kp}  Kd={Kd}",
        fontsize=13, fontweight='bold', y=1.01
    )

    for row, log in enumerate(all_logs):
        mid         = log['motor_id']
        label       = log['label']
        colour      = colours.get(mid, '#9C27B0')
        t           = log['time_s']
        current_mA  = log['torque'] * CURRENT_UNIT_MA
        ft_data     = log.get('ft', np.zeros((len(t), 6)))
        phase_times = log['phase_times']
        aborted     = log.get('aborted', False)

        title_suffix = " [ABORTED — ESC]" if aborted else ""
        title_colour = 'red' if aborted else 'black'

        ax_frc = axes2[row, 0]
        ax_trq = axes2[row, 1]
        ax_cur = axes2[row, 2]

        ft_has_data = (
            ft_data is not None
            and ft_data.ndim == 2
            and ft_data.shape[0] > 0
            and ft_data.shape[1] == 6
            and np.any(ft_data != 0)
        )

        # ── Col 1: Forces ─────────────────────────────────────────────────────
        if ft_has_data:
            ft_filtered = lowpass_filter(ft_data, FT_PLOT_CUTOFF_HZ, FT_RATE_HZ)
            force_mag   = np.linalg.norm(ft_filtered[:, :3], axis=1)
            torque_mag  = np.linalg.norm(ft_filtered[:, 3:], axis=1)

            for i, ch in enumerate(['Fx', 'Fy', 'Fz']):
                ax_frc.plot(t, ft_data[:, i], color=FT_COLOURS[ch],
                            linewidth=0.5, alpha=0.25, zorder=1)
                ax_frc.plot(t, ft_filtered[:, i], color=FT_COLOURS[ch],
                            linewidth=1.2, label=f'{ch} (N)', zorder=2)

            ax_frc.plot(t, force_mag, color='black', linewidth=2.0,
                        label='|F| resultant', zorder=5)
            ax_frc.axhline(y=0, color='black', linewidth=0.6,
                           linestyle='--', alpha=0.4)
            draw_phase_lines(ax_frc, phase_times, t, aborted)
            annotate_abort(ax_frc, t, aborted)
            stat_box(ax_frc,
                f"Peak |F| : {float(np.max(force_mag)):.3f} N\n"
                f"RMS  |F| : {float(np.sqrt(np.mean(force_mag**2))):.3f} N"
            )

            # ── Col 2: Torques ────────────────────────────────────────────────
            for i, ch in enumerate(['Tx', 'Ty', 'Tz']):
                ax_trq.plot(t, ft_data[:, i+3], color=FT_COLOURS[ch],
                            linewidth=0.5, alpha=0.25, linestyle='--', zorder=1)
                ax_trq.plot(t, ft_filtered[:, i+3], color=FT_COLOURS[ch],
                            linewidth=1.2, linestyle='--',
                            label=f'{ch} (N·m)', zorder=2)

            ax_trq.plot(t, torque_mag, color='black', linewidth=2.0,
                        linestyle='--', label='|T| resultant', zorder=5)
            ax_trq.axhline(y=0, color='black', linewidth=0.6,
                           linestyle='--', alpha=0.4)
            draw_phase_lines(ax_trq, phase_times, t, aborted)
            annotate_abort(ax_trq, t, aborted)
            stat_box(ax_trq,
                f"Peak |T| : {float(np.max(torque_mag)):.4f} N·m\n"
                f"RMS  |T| : {float(np.sqrt(np.mean(torque_mag**2))):.4f} N·m"
            )

        else:
            for ax in (ax_frc, ax_trq):
                ax.text(0.5, 0.5,
                        'No F/T data\n(sensor not connected or all zeros)',
                        transform=ax.transAxes,
                        ha='center', va='center',
                        fontsize=9, color='grey')

        ax_frc.set_title(
            f"Motor {mid} — {label}{title_suffix}\nForces at Base",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_frc.set_ylabel("Force (N)", fontsize=9)
        ax_frc.legend(fontsize=8, loc='upper right')
        ax_frc.grid(True, alpha=0.3)
        ax_frc.set_xlim(left=0)

        ax_trq.set_title(
            f"Motor {mid} — {label}{title_suffix}\nTorques at Base",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_trq.set_ylabel("Torque (N·m)", fontsize=9)
        ax_trq.legend(fontsize=8, loc='upper right')
        ax_trq.grid(True, alpha=0.3)
        ax_trq.set_xlim(left=0)

        # ── Col 3: Commanded Current ──────────────────────────────────────────
        ax_cur.plot(t, current_mA, color=colour, linewidth=1.2,
                    label='Current (mA)')
        ax_cur.fill_between(t, current_mA, 0, alpha=0.15, color=colour)
        ax_cur.axhline(y=0, color='black', linewidth=0.8,
                       linestyle='--', alpha=0.6)
        ax_cur.axhline(y=+CURRENT_CEILING_MA, color='red', linewidth=0.8,
                       linestyle=':', alpha=0.7,
                       label=f'±{CURRENT_CEILING_MA:.0f} mA ceiling')
        ax_cur.axhline(y=-CURRENT_CEILING_MA, color='red', linewidth=0.8,
                       linestyle=':', alpha=0.7)
        draw_phase_lines(ax_cur, phase_times, t, aborted)
        annotate_abort(ax_cur, t, aborted)

        if len(current_mA) > 0:
            stat_box(ax_cur,
                f"Peak : {float(np.max(np.abs(current_mA))):.1f} mA\n"
                f"RMS  : {float(np.sqrt(np.mean(current_mA**2))):.1f} mA\n"
                f"Mean : {float(np.mean(np.abs(current_mA))):.1f} mA"
            )

        ax_cur.set_title(
            f"Motor {mid} — {label}{title_suffix}\nCommanded Current",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_cur.set_ylabel("Current (mA)", fontsize=9)
        ax_cur.legend(fontsize=8, loc='upper right')
        ax_cur.grid(True, alpha=0.3)
        ax_cur.set_xlim(left=0)

        if row == n_motors - 1:
            ax_frc.set_xlabel("Time (s)", fontsize=9)
            ax_trq.set_xlabel("Time (s)", fontsize=9)
            ax_cur.set_xlabel("Time (s)", fontsize=9)

    plt.figure(fig2.number)
    plt.tight_layout()
    fig2_path = os.path.join(RESULTS_DIR, f"{base_name}_force_torque_current.png")
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f"[PLOT] Figure 2 saved to: {fig2_path}")

    plt.show()

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── 1. Start ESC monitor FIRST — before any hardware is touched ───────────
    #   keyboard.add_hotkey() reuses keyboard's internal thread — no new thread
    setup_estop()

    # ── 2. Hardware setup ─────────────────────────────────────────────────────
    port, packet = init_port()
    mod          = init_mod(port, packet)
    params       = load_flipper_params()

    if params is not None:
        kp = params['kp']
        kd = params['kd']
    else:
        kp = KP
        kd = KD
        print(f"[PARAMS] WARNING — Could not load flipper_params.json, using defaults: Kp={kp}, Kd={kd}")

    sensor = None
    try:
        sensor = init_ft_sensor()
    except Exception as e:
        print(f"[F/T SENSOR] WARNING — Could not connect: {e}")
        print("[F/T SENSOR] Continuing WITHOUT sensor data.\n")
    
    print("=" * 65)
    print("   SEA TURTLE FLIPPER — AUTOMATED TRAJECTORY CONTROL")
    print("=" * 65)
    print(f"""
  Control Law:
    τ = C(q,q̇)q̇_d + M(q)q̈_d + G(q) + Kp(q_d-q) + Kd(q̇_d-q̇)
    Kp=({kp}) Nm/rad   Kd=({kd}) Nm·s/rad

  Trajectory per motor:
    home ──► lower_limit ──► upper_limit ──► home
    Segment duration : {SEGMENT_DURATION:.1f} s x 3 = {3*SEGMENT_DURATION:.1f} s/motor
    Total duration   : {3 * 3 * SEGMENT_DURATION:.1f} s ({3} motors x {3*SEGMENT_DURATION:.1f} s)
    Loop rate        : {LOOP_HZ} Hz  ({N_SEGMENT_POINTS} setpoints/segment)

  Press ESC at any time to immediately stop all motors and exit.
    """)

    all_logs      = []
    results       = {}
    program_start = time.time()

    #   try/except/finally guarantees shutdown() is ALWAYS called regardless
    # of whether the program exits normally, via ESC, or via unexpected error.
    try:

        # ── 3. Dynamic limits and initial homing ──────────────────────────────
        limits     = init_dynamic_limits(mod)
        positions  = home_all_motors(mod, port, packet, limits)
        velocities = {mid: 0.0 for mid in ALL_IDS}

        # ── 4. Print trajectory summary ───────────────────────────────────────
        print("\n[SUMMARY] Planned trajectory bounds:")
        for mid in ALL_IDS:
            home  = limits[mid]['home']
            lower = limits[mid]['lower']
            upper = limits[mid]['upper']
            print(
                f"  Motor {mid} ({MOTOR_LABELS[mid]:<26}) "
                f"home={rad_to_deg(home):>8.2f}°  "
                f"lower={rad_to_deg(lower):>8.2f}°  "
                f"upper={rad_to_deg(upper):>8.2f}°"
            )

        # ── 5. Execute trajectory for each motor sequentially ─────────────────
        for motor_id in ALL_IDS:

            #   Check ESC before starting each motor's trajectory
            if is_estop():
                handle_estop(mod)
                raise EmergencyStop(
                    f"ESC pressed before Motor {motor_id} trajectory."
                )

            home  = limits[motor_id]['home']
            lower = limits[motor_id]['lower']
            upper = limits[motor_id]['upper']

            traj_q, traj_dq, traj_ddq, phase_idx = generate_trajectory(
                home, lower, upper
            )

            print(
                f"\n[TRAJECTORY] Motor {motor_id} — "
                f"{len(traj_q)} setpoints generated  "
                f"({rad_to_deg(traj_q.min()):.2f}° to "
                f"{rad_to_deg(traj_q.max()):.2f}°)"
            )

            # EmergencyStop propagates up from run_motor_trajectory()
            # if ESC is pressed mid-trajectory — caught below in except block
            positions, velocities, log = run_motor_trajectory(
                mod,
                motor_id,
                traj_q,
                traj_dq,
                traj_ddq,
                phase_idx,
                positions,
                velocities,
                limits,
                sensor=sensor
            )

            all_logs.append(log)

            final_pos         = positions[motor_id]
            final_err         = abs(final_pos - home)
            results[motor_id] = {
                'final_pos_deg' : rad_to_deg(final_pos),
                'home_deg'      : rad_to_deg(home),
                'q_error_deg'     : rad_to_deg(final_err),
                'q_rms_error_deg' : float(np.sqrt(np.mean(log['q_error'] ** 2)))
                                  if len(log['q_error']) > 0 else 0.0,
                'q_peak_error_deg': float(np.max(np.abs(log['q_error'])))
                                  if len(log['q_error']) > 0 else 0.0,
                'dq_rms_error_deg': float(np.sqrt(np.mean(log['dq_error'] ** 2)))
                                    if len(log['dq_error']) > 0 else 0.0,
                'dq_peak_error_deg': float(np.max(np.abs(log['dq_error'])))
                                      if len(log['dq_error']) > 0 else 0.0
            }

            #   Check ESC after each motor completes
            if is_estop():
                handle_estop(mod)
                raise EmergencyStop(
                    f"ESC pressed after Motor {motor_id} trajectory."
                )

            print(f"\n[PAUSE] Pausing 1.0 s before next motor ...")
            mod.send_torque_cmd([HOLD_CURRENT] * N_MOTORS)

            #   Check ESC every 100 ms during inter-motor pause
            for _ in range(10):
                if is_estop():
                    handle_estop(mod)
                    raise EmergencyStop(
                        "ESC pressed during inter-motor pause."
                    )
                time.sleep(0.1)

        # ── 6. Return all motors to home ──────────────────────────────────────
        print("\n[HOME] Returning all motors to home positions ...")
        home_all_motors(mod, port, packet, limits)

        # ── 7. Print final summary ────────────────────────────────────────────
        total_time = time.time() - program_start
        print("\n" + "=" * 70)
        print("   TRAJECTORY COMPLETE — RESULTS SUMMARY")
        print("=" * 70)
        print(
            f"  {'Motor':<8} {'Axis':<28} "
            f"{'Final(°)':>10}  {'Home(°)':>10}  "
            f"{'Err(°)':>8}  {'RMS(°)':>8}  {'Peak(°)':>8}"
        )
        print("─" * 70)
        for mid in results:
            r = results[mid]
            print(
                f"  ID {mid:<5} {MOTOR_LABELS[mid]:<28} "
                f"{r['final_pos_deg']:>9.2f}°  "
                f"{r['home_deg']:>9.2f}°  "
                f"{r['q_error_deg']:>7.2f}°  "
                f"{r['q_rms_error_deg']:>7.3f}°  "
                f"{r['q_peak_error_deg']:>7.3f}°"
                f"{r['dq_rms_error_deg']:>7.3f}°/s  "
                f"{r['dq_peak_error_deg']:>7.3f}°/s"
            )
        print("─" * 70)
        print(f"  Total program time: {total_time:.2f} s")
        print("=" * 70)

    #   ESC was pressed — clean message, fall through to finally
    except EmergencyStop as e:
        print(f"\n[ESTOP] Program stopped: {e}")
        total_time = time.time() - program_start
        print(f"[ESTOP] Elapsed time at stop: {total_time:.2f} s")
        if all_logs:
            print(
                f"[ESTOP] {len(all_logs)} motor trajectory(s) recorded — "
                f"partial data will be plotted."
            )

    #   Catch any unexpected exception — still shuts down safely
    except Exception as e:
        print(f"\n[ERROR] Unexpected exception: {e}")

    #   ALWAYS runs — guarantees hardware is safe regardless of exit path
    finally:
        teardown_estop()        # unregister ESC hotkey
        if sensor is not None:
            close_ft_sensor(sensor)
        shutdown(mod, port)     # zero currents, disable torque, close port

    # ── 8. Save CSV data ──────────────────────────────────────────────────────
    # Called after hardware shutdown — purely offline file write
    if all_logs:
        print("\n[CSV] Saving trajectory data ...")
        csv_path = save_results_csv(all_logs, kp, kd)

    # ── 9. Plot whatever data was recorded ────────────────────────────────────
    # Hardware is already shut down — plotting is purely offline.
    # Partial logs (aborted mid-trajectory) are annotated on the graph.
    if all_logs:
        print("\n[PLOT] Generating position tracking graphs ...")
        plot_results(all_logs, kp, kd)
    else:
        print("[PLOT] No trajectory data recorded — nothing to plot.")

# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()