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
    All motors return to home → automatic shutdown

Emergency stop:
    Press ESC at any time to immediately zero all currents and shut down.
    Any trajectory data recorded before ESC is plotted automatically.

File dependencies:
    Mod.py            — GroupSyncRead / GroupSyncWrite
    Constants.py      — port, baud rate, hardware constants
    torque_control.py — computed torque + PD controller
    mass.py           — dynamic model (M, C, G matrices)
"""
import os
from datetime import datetime
import sys
import time
import math
import threading
import keyboard
import numpy as np
import matplotlib.pyplot as plt

from dynamixel_sdk import *
from Mod import *
from Constants import *
from torque_control import torque_control

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
HOME_POSITIONS_RAD = [-2*math.pi, 2*math.pi, 2*math.pi]  # 360° for all three motors

# ── Unit conversions ──────────────────────────────────────────────────────────
STEPS_TO_RAD = (2 * math.pi) / 4096
RAD_TO_STEPS = 4096 / (2 * math.pi)

# ── Position limits (radians) ─────────────────────────────────────────────────
MAX_DELTA_RAD      = 114 * 3 * STEPS_TO_RAD    # ±10° x 3 soft travel limit
RAMP_ZONE_RAD      =  50 * STEPS_TO_RAD    # deceleration band near limit
HOME_THRESHOLD_RAD = DXL_MOVING_STATUS_THRESHOLD * STEPS_TO_RAD

# ── Current / hardware limits ─────────────────────────────────────────────────
# XW540-T260: 2.69 mA per register step
CURRENT_UNIT_MA    = 2.69
CURRENT_CEILING_MA = 300
CURRENT_CEILING    = int(CURRENT_CEILING_MA / CURRENT_UNIT_MA)  # ≈ 186 steps
HOLD_CURRENT       = 0

# ── PD Controller gains ───────────────────────────────────────────────────────
# KP_VAL = 4        # proportional gain (Nm/rad)
# KD_VAL = 0.25        # derivative gain   (Nm·s/rad)
Kp = np.diag([5, 4, 5, 0, 0, 0])
Kd = np.diag([0.3, 0.3, 0.4, 0, 0, 0])

# ── Trajectory parameters ─────────────────────────────────────────────────────
LOOP_HZ          = 100
LOOP_DT          = 1.0 / LOOP_HZ
SEGMENT_DURATION = 1.0                              # seconds per segment
N_SEGMENT_POINTS = int(SEGMENT_DURATION * LOOP_HZ) # setpoints per segment
STATUS_INTERVAL  = 0.5                              # status print interval (s)

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
    "home → lower limit",
    "lower limit → upper limit",
    "upper limit → home",
]

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
    e.g. -56 → 65536 + (-56) = 65480
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
    for i, v in enumerate(values):
        vec[i][0] = v
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

    # Inactive motors track their current position → zero tracking error
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
    dqd_vec[active_idx][0]     = dqd_active
    ddqd_vec[active_idx][0]    = ddqd_active

    # Call torque controller
    try:
        tau_steps = torque_control(
            q_vec, dq_vec, qd_vec, dqd_vec, ddqd_vec, Kp, Kd
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

        raw        = int(tau_steps[i][0])
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
                         all_limits:     dict) -> tuple:
    """
    Execute a full position trajectory for one motor.
    Checks ESC at the top of every iteration — maximum response
    latency is one LOOP_DT (10 ms at 100 Hz).

    Returns
    -------
    (positions, velocities, log)
    log contains time, position, desired, error, torque arrays
    and aborted flag if ESC was pressed mid-trajectory.
    """
    n_steps          = len(trajectory_q)
    current_phase    = 0
    last_status_time = 0.0
    start_time       = time.time()

    # Preallocate recording arrays
    log_time     = np.zeros(n_steps)
    log_q_actual = np.zeros(n_steps)
    log_q_des    = np.zeros(n_steps)
    log_q_error  = np.zeros(n_steps)
    log_torque   = np.zeros(n_steps)
    phase_times  = []

    print(
        f"\n{'═'*65}\n"
        f"  MOTOR {motor_id} — {MOTOR_LABELS[motor_id]}\n"
        f"  Total steps : {n_steps}  ({n_steps * LOOP_DT:.1f} s)\n"
        f"  Trajectory  : home → lower → upper → home\n"
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

        #   Check ESC at the top of every iteration
        # Maximum response latency = one LOOP_DT (10 ms at 100 Hz)
        if is_estop():
            handle_estop(mod)

            #   Save partial log so plotting still works on abort
            log = {
                'time_s'      : log_time[:step],
                'q_actual'    : log_q_actual[:step],
                'q_desired'   : log_q_des[:step],
                'q_error'     : log_q_error[:step],
                'torque'      : log_torque[:step],
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

        # Phase tracking
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

        # Read current state
        positions  = read_all_positions(mod, positions)
        velocities = read_all_velocities(mod, velocities)

        # Trajectory setpoints for this step
        qd_step   = trajectory_q[step]
        dqd_step  = trajectory_dq[step]
        ddqd_step = trajectory_ddq[step]

        # Compute torque via PD + inverse dynamics
        torque_cmds = compute_single_motor_torque(
            motor_id, positions, velocities,
            qd_step, dqd_step, ddqd_step,
            all_limits
        )

        # Send all motors via GroupSyncWrite
        mod.send_torque_cmd([
            to_unsigned_16(torque_cmds[mid]) for mid in ALL_IDS
        ])

        # Record data
        pos_actual         = positions[motor_id]
        error              = qd_step - pos_actual
        log_time[step]     = elapsed
        log_q_actual[step] = rad_to_deg(pos_actual)
        log_q_des[step]    = rad_to_deg(qd_step)
        log_q_error[step]  = rad_to_deg(error)
        log_torque[step]   = torque_cmds[motor_id]

        # Periodic status print
        now = time.time()
        if now - last_status_time >= STATUS_INTERVAL:
            pct_done = 100.0 * step / max(n_steps - 1, 1)
            print(
                f"    step {step:>4}/{n_steps}  "
                f"({pct_done:>5.1f}%)  "
                f"t={elapsed:>5.1f}s  |  "
                f"q={rad_to_deg(pos_actual):>8.2f}°  "
                f"qd={rad_to_deg(qd_step):>8.2f}°  "
                f"err={rad_to_deg(error):>+7.2f}°  "
                f"I={torque_cmds[motor_id]:>5} steps"
            )
            last_status_time = now

        # Pace loop
        elapsed_step = time.time() - step_start
        sleep_t      = LOOP_DT - elapsed_step
        if sleep_t > 0:
            time.sleep(sleep_t)

    # Zero this motor after trajectory completes
    mod.send_torque_cmd([HOLD_CURRENT] * N_MOTORS)
    total_time = time.time() - start_time

    print(
        f"\n  [COMPLETE] Motor {motor_id} trajectory done. "
        f"Total time: {total_time:.2f} s"
    )
    print(
        f"  Final position : {rad_to_deg(positions[motor_id]):.2f}°  "
        f"Home           : {rad_to_deg(all_limits[motor_id]['home']):.2f}°  "
        f"Final error    : "
        f"{rad_to_deg(positions[motor_id] - all_limits[motor_id]['home']):+.2f}°"
    )

    # Pack full log
    log = {
        'time_s'      : log_time,
        'q_actual'    : log_q_actual,
        'q_desired'   : log_q_des,
        'q_error'     : log_q_error,
        'torque'      : log_torque,
        'phase_times' : phase_times,
        'motor_id'    : motor_id,
        'label'       : MOTOR_LABELS[motor_id],
        'aborted'     : False,
        'aborted_step': None,
    }

    return positions, velocities, log


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_results(all_logs: list) -> None:
    """
    Plot position tracking, tracking error, and commanded current
    for all motors after the trajectory program completes.

    Layout (one row per motor, three columns):
        Left   : actual vs desired position  (deg)  vs time (s)
        Centre : position tracking error     (deg)  vs time (s)
        Right  : commanded current           (mA)   vs time (s)

    Current values are converted from register steps to mA:
        I_mA = steps x CURRENT_UNIT_MA  (2.69 mA/step for XW540-T260)

    Parameters
    ----------
    all_logs : list of log dicts returned by run_motor_trajectory()
    """
    import os
    from datetime import datetime

    n_motors = len(all_logs)
    if n_motors == 0:
        print("[PLOT] No data to plot.")
        return

    # ── Figure layout — n_motors rows × 3 columns ─────────────────────────────
    fig, axes = plt.subplots(
        n_motors, 3,
        figsize=(18, 4 * n_motors),     # wider to accommodate 3 columns
        sharex='row'
    )

    # Ensure axes is always 2D even for a single motor
    if n_motors == 1:
        axes = np.array([axes])

    # ── Colour scheme — one colour per motor ──────────────────────────────────
    colours = {
        MOTOR_ID_X: '#2196F3',      # blue
        MOTOR_ID_Y: '#4CAF50',      # green
        MOTOR_ID_Z: '#FF5722',      # orange-red
    }

    fig.suptitle(
        "Sea Turtle Flipper — Trajectory Results\n"
        f"Computed Torque + PD Control  |  "
        f"Kp=(x={Kp[0][0]}, y={Kp[1][1]}, z={Kp[2][2]}) Nm/rad   Kd=(x={Kd[0][0]}, y={Kd[1][1]}, z={Kd[2][2]}) Nm·s/rad  |  "
        f"Current unit: {CURRENT_UNIT_MA} mA/step",
        fontsize=13,
        fontweight='bold',
        y=1.01
    )

    for row, log in enumerate(all_logs):
        mid         = log['motor_id']
        label       = log['label']
        colour      = colours.get(mid, '#9C27B0')
        t           = log['time_s']
        q_act       = log['q_actual']
        q_des       = log['q_desired']
        q_err       = log['q_error']
        phase_times = log['phase_times']
        aborted     = log.get('aborted', False)

        # Convert register steps → mA for display
        # Preserves sign so direction of current is visible
        current_mA  = log['torque'] * CURRENT_UNIT_MA

        title_suffix = " [ABORTED — ESC]" if aborted else ""
        title_colour = 'red' if aborted else 'black'

        ax_pos = axes[row, 0]   # left   — position tracking
        ax_err = axes[row, 1]   # centre — tracking error
        ax_cur = axes[row, 2]   # right  — commanded current

        # ══════════════════════════════════════════════════════════════════════
        #  LEFT SUBPLOT — Position Tracking
        # ══════════════════════════════════════════════════════════════════════
        ax_pos.plot(
            t, q_des,
            color='black', linewidth=1.5, linestyle='--',
            label='Desired $q_d$', zorder=3
        )
        ax_pos.plot(
            t, q_act,
            color=colour, linewidth=1.5,
            label='Actual $q$', zorder=2
        )
        ax_pos.fill_between(
            t, q_des, q_act,
            alpha=0.15, color=colour,
            label='Tracking error region'
        )

        # Phase boundary vertical lines
        for p_idx, pt in enumerate(phase_times):
            ax_pos.axvline(
                x=pt, color='grey', linewidth=0.8,
                linestyle=':', zorder=1
            )
            ax_pos.text(
                pt + 0.05, ax_pos.get_ylim()[0],
                f'P{p_idx + 1}', fontsize=7,
                color='grey', va='bottom'
            )

        # Home / lower / upper reference lines
        if len(q_des) > 0:
            home_deg  = q_des[0]
            lower_deg = q_des.min()
            upper_deg = q_des.max()
            for val, lbl, ls in [
                (home_deg,  'home',  '-'),
                (lower_deg, 'lower', '--'),
                (upper_deg, 'upper', '--'),
            ]:
                ax_pos.axhline(
                    y=val, color='grey',
                    linewidth=0.6, linestyle=ls, alpha=0.5
                )
                ax_pos.text(
                    t[-1] if len(t) > 0 else 0, val,
                    f' {lbl}\n {val:.1f}°',
                    fontsize=7, color='grey', va='center'
                )

        # ESC abort annotation
        if aborted and len(t) > 0:
            ax_pos.axvline(
                x=t[-1], color='red', linewidth=1.5,
                linestyle='--', zorder=4
            )
            ax_pos.text(
                t[-1], ax_pos.get_ylim()[1],
                ' ESC STOP', fontsize=8,
                color='red', va='top', fontweight='bold'
            )

        ax_pos.set_title(
            f"Motor {mid} — {label}{title_suffix}\nPosition Tracking",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_pos.set_ylabel("Position (°)", fontsize=9)
        ax_pos.legend(fontsize=8, loc='upper right')
        ax_pos.grid(True, alpha=0.3)
        ax_pos.set_xlim(left=0)

        # ══════════════════════════════════════════════════════════════════════
        #  CENTRE SUBPLOT — Tracking Error
        # ══════════════════════════════════════════════════════════════════════
        ax_err.plot(
            t, q_err,
            color=colour, linewidth=1.2,
            label='Error $q_d - q$'
        )
        ax_err.fill_between(t, q_err, 0, alpha=0.15, color=colour)
        ax_err.axhline(
            y=0, color='black', linewidth=0.8,
            linestyle='--', alpha=0.6
        )

        for pt in phase_times:
            ax_err.axvline(
                x=pt, color='grey', linewidth=0.8,
                linestyle=':', zorder=1
            )

        if aborted and len(t) > 0:
            ax_err.axvline(
                x=t[-1], color='red', linewidth=1.5,
                linestyle='--', zorder=4,
                label='ESC stop'
            )

        if len(q_err) > 0:
            rms_err  = float(np.sqrt(np.mean(q_err ** 2)))
            peak_err = float(np.max(np.abs(q_err)))
            ax_err.text(
                0.02, 0.95,
                f"RMS  : {rms_err:.3f}°\nPeak : {peak_err:.3f}°",
                transform=ax_err.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='grey',
                    alpha=0.8
                )
            )

        ax_err.set_title(
            f"Motor {mid} — {label}{title_suffix}\nTracking Error ($q_d - q$)",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_err.set_ylabel("Error (°)", fontsize=9)
        ax_err.legend(fontsize=8, loc='upper right')
        ax_err.grid(True, alpha=0.3)
        ax_err.set_xlim(left=0)

        # ══════════════════════════════════════════════════════════════════════
        #  RIGHT SUBPLOT — Commanded Current (mA)
        # ══════════════════════════════════════════════════════════════════════
        ax_cur.plot(
            t, current_mA,
            color=colour, linewidth=1.2,
            label='Commanded current (mA)'
        )
        ax_cur.fill_between(
            t, current_mA, 0,
            alpha=0.15, color=colour
        )

        # Zero line — separates positive and negative current
        ax_cur.axhline(
            y=0, color='black', linewidth=0.8,
            linestyle='--', alpha=0.6
        )

        # Hardware ceiling lines — shows when current is being clamped
        ax_cur.axhline(
            y=+CURRENT_CEILING_MA,
            color='red', linewidth=0.8,
            linestyle=':', alpha=0.7,
            label=f'Ceiling ±{CURRENT_CEILING_MA} mA'
        )
        ax_cur.axhline(
            y=-CURRENT_CEILING_MA,
            color='red', linewidth=0.8,
            linestyle=':', alpha=0.7
        )

        # Phase boundary vertical lines
        for pt in phase_times:
            ax_cur.axvline(
                x=pt, color='grey', linewidth=0.8,
                linestyle=':', zorder=1
            )

        # ESC abort annotation
        if aborted and len(t) > 0:
            ax_cur.axvline(
                x=t[-1], color='red', linewidth=1.5,
                linestyle='--', zorder=4,
                label='ESC stop'
            )

        # Annotate peak and RMS current
        if len(current_mA) > 0:
            peak_cur = float(np.max(np.abs(current_mA)))
            rms_cur  = float(np.sqrt(np.mean(current_mA ** 2)))
            mean_cur = float(np.mean(np.abs(current_mA)))
            ax_cur.text(
                0.02, 0.95,
                f"Peak : {peak_cur:.1f} mA\n"
                f"RMS  : {rms_cur:.1f} mA\n"
                f"Mean : {mean_cur:.1f} mA",
                transform=ax_cur.transAxes,
                fontsize=8, verticalalignment='top',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='white',
                    edgecolor='grey',
                    alpha=0.8
                )
            )

        ax_cur.set_title(
            f"Motor {mid} — {label}{title_suffix}\n"
            f"Commanded Current  ({CURRENT_UNIT_MA} mA/step)",
            fontsize=10, fontweight='bold', color=title_colour
        )
        ax_cur.set_ylabel("Current (mA)", fontsize=9)
        ax_cur.legend(fontsize=8, loc='upper right')
        ax_cur.grid(True, alpha=0.3)
        ax_cur.set_xlim(left=0)

        # ── X-axis labels on bottom row only ──────────────────────────────────
        if row == n_motors - 1:
            ax_pos.set_xlabel("Time (s)", fontsize=9)
            ax_err.set_xlabel("Time (s)", fontsize=9)
            ax_cur.set_xlabel("Time (s)", fontsize=9)

    plt.tight_layout()

    # ── Build results folder and filename ─────────────────────────────────────
    RESULTS_DIR = "flipper_results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = os.path.join(
        RESULTS_DIR,
        f"flipper_trajectory_Kp(x={Kp[0][0]}_y={Kp[1][1]}_z={Kp[2][2]})_Kd(x={Kd[0][0]}_y={Kd[1][1]}_z={Kd[2][2]})_{timestamp}.png"
    )

    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n[PLOT] Figure saved to: {filename}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("   SEA TURTLE FLIPPER — AUTOMATED TRAJECTORY CONTROL")
    print("=" * 65)
    print(f"""
  Control Law:
    τ = C(q,q̇)q̇_d + M(q)q̈_d + G(q) + Kp(q_d-q) + Kd(q̇_d-q̇)
    Kp=(x={Kp[0][0]}, y={Kp[1][1]}, z={Kp[2][2]}) Nm/rad   Kd=(x={Kd[0][0]}, y={Kd[1][1]}, z={Kd[2][2]}) Nm·s/rad

  Trajectory per motor:
    home ──► lower_limit ──► upper_limit ──► home
    Segment duration : {SEGMENT_DURATION:.1f} s x 3 = {3*SEGMENT_DURATION:.1f} s/motor
    Total duration   : {3 * 3 * SEGMENT_DURATION:.1f} s ({3} motors x {3*SEGMENT_DURATION:.1f} s)
    Loop rate        : {LOOP_HZ} Hz  ({N_SEGMENT_POINTS} setpoints/segment)

  Press ESC at any time to immediately stop all motors and exit.
    """)

    # ── 1. Start ESC monitor FIRST — before any hardware is touched ───────────
    #   keyboard.add_hotkey() reuses keyboard's internal thread — no new thread
    setup_estop()

    # ── 2. Hardware setup ─────────────────────────────────────────────────────
    port, packet = init_port()
    mod          = init_mod(port, packet)

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
            )

            all_logs.append(log)

            final_pos         = positions[motor_id]
            final_err         = abs(final_pos - home)
            results[motor_id] = {
                'final_pos_deg' : rad_to_deg(final_pos),
                'home_deg'      : rad_to_deg(home),
                'error_deg'     : rad_to_deg(final_err),
                'rms_error_deg' : float(np.sqrt(np.mean(log['q_error'] ** 2)))
                                  if len(log['q_error']) > 0 else 0.0,
                'peak_error_deg': float(np.max(np.abs(log['q_error'])))
                                  if len(log['q_error']) > 0 else 0.0,
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
                f"{r['error_deg']:>7.2f}°  "
                f"{r['rms_error_deg']:>7.3f}°  "
                f"{r['peak_error_deg']:>7.3f}°"
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
        shutdown(mod, port)     # zero currents, disable torque, close port

    # ── 8. Plot whatever data was recorded ────────────────────────────────────
    # Hardware is already shut down — plotting is purely offline.
    # Partial logs (aborted mid-trajectory) are annotated on the graph.
    if all_logs:
        print("\n[PLOT] Generating position tracking graphs ...")
        plot_results(all_logs)
    else:
        print("[PLOT] No trajectory data recorded — nothing to plot.")


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()