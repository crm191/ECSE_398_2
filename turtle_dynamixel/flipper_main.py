#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
flipper_main.py  —  Current-Control Edition
--------------------------------------------
Sea Turtle Flipper controller using CURRENT (torque) control mode.

Motors are driven by a commanded current (mA).  Position is read back
continuously and used ONLY as a safety guard — if a motor approaches its
soft limit the current is zeroed automatically.

Hardware:
    Motor 1 (ID=1) → X-axis  (forward / backward stroke)
    Motor 2 (ID=2) → Y-axis  (lateral sweep)
    Motor 3 (ID=3) → Z-axis  (vertical pitch / twist)

Controls:
    W / S   →  Motor 1 (X)   positive / negative current
    A / D   →  Motor 2 (Y)   positive / negative current
    I / K   →  Motor 3 (Z)   positive / negative current
    SPACE   →  Zero ALL currents (coast to stop)
    P       →  Print live positions + active currents
    Q / ESC →  Safe shutdown
    H     →  Home all motors (position mode) then return to current control
"""

import os
import sys
import time
import keyboard
import math

'''
# ─── Path setup ───────────────────────────────────────────────────────────────
submodule = (
    os.path.expanduser("~")
    + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel"
)
sys.path.append(submodule)
'''

# ────── Path Setup - Windows ──────────────────────────────────────────────────
# sys.path.append(r"C:\Users\charr\OneDrive - Case Western Reserve University\Documents\ECSE 398\venv\Lib\site-packages")

# ─── Platform-safe keyboard input ────────────────────────────────────────────
if os.name == "nt":
    import msvcrt

    def getch():
        return msvcrt.getch().decode()

    def kbhit():
        return msvcrt.kbhit()

else:
    import termios
    import fcntl
    from select import select

    fd = sys.stdin.fileno()
    _old_term = termios.tcgetattr(fd)
    _new_term = termios.tcgetattr(fd)

    def getch():
        _new_term[3] = _new_term[3] & ~termios.ICANON & ~termios.ECHO
        termios.tcsetattr(fd, termios.TCSANOW, _new_term)
        try:
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, _old_term)
        return ch

    def kbhit():
        _new_term[3] = _new_term[3] & ~(termios.ICANON | termios.ECHO)
        termios.tcsetattr(fd, termios.TCSANOW, _new_term)
        try:
            dr, _, _ = select([sys.stdin], [], [], 0)
            return bool(dr)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, _old_term)

# ─── Project imports ──────────────────────────────────────────────────────────
from dynamixel_sdk import *
from Dynamixel import *
from Constants import *
# from dyn_functions import to_radians

# ══════════════════════════════════════════════════════════════════════════════
#  FLIPPER CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Motor IDs ─────────────────────────────────────────────────────────────────
MOTOR_ID_X = 1      # Forward / backward stroke
MOTOR_ID_Y = 2      # Lateral sweep
MOTOR_ID_Z = 3      # Vertical pitch

ALL_IDS = [MOTOR_ID_X, MOTOR_ID_Y, MOTOR_ID_Z]

# ── Position limits (steps, 4096 steps = 360°) ───────────────────────────────
# ±10° of travel per axis from startup position.
# 114 steps ≈ 10°
MAX_DELTA_STEPS = 114       # ±10° soft travel limit per axis

# RAMP_ZONE must be LESS than MAX_DELTA_STEPS so the ramp has room to act.
# With MAX_DELTA_STEPS=228, the furthest a motor can be from a limit is 228
# steps (at home). 
# Fixed: RAMP_ZONE reduced to 50 (≈ 4.4°), leaving a full-power band of 178
# steps either side of home.
RAMP_ZONE = 50              # steps — deceleration band near each soft limit

# ── Current commands (mA) ─────────────────────────────────────────────────────
# Current Limit register max = 2,047 → 2047 × 2.69 mA = ~5,506 mA (5.5 A stall)

# To send 150 mA:   150 / 2.69 ≈ 56 steps
# To send 300 mA:   300 / 2.69 ≈ 112 steps
# To send 500 mA:   500 / 2.69 ≈ 186 steps

CURRENT_UNIT_MA     = 2.69          # mA per register step
DRIVE_CURRENT_MA    = 50           # desired drive current in mA
CEILING_CURRENT_MA  = 150           # desired ceiling in mA

# Convert to register steps before sending
DRIVE_CURRENT    = int(DRIVE_CURRENT_MA  / CURRENT_UNIT_MA)   # = 56 steps
CURRENT_CEILING  = int(CEILING_CURRENT_MA / CURRENT_UNIT_MA)  # = 186 steps
HOLD_CURRENT     = 0                 # 0 mA to coast when idle

# ── Collision safety ──────────────────────────────────────────────────────────
# Sum of per-axis deviations from home must stay below this threshold.
COMBINED_DEV_LIMIT = int(6 * MAX_DELTA_STEPS)    #  steps

# ── Timing ────────────────────────────────────────────────────────────────────
LOOP_HZ = 20
LOOP_DT = 1.0 / LOOP_HZ    # 50 ms per iteration

# ── Homing parameters ────────────────────────────────────────────────────────
HOME_NUDGE_VEL      = 20    # Profile velocity used during homing
HOME_SETTLE_TIMEOUT = 5.0   # Seconds before homing gives up

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer between lo and hi."""
    return max(lo, min(hi, value))


def steps_to_deg(steps: int) -> float:
    """Convert motor steps to degrees for display."""
    return round(steps * (360.0 / 4096.0), 2)


def ramp_factor(pos: int, motor_lims: dict, direction: int) -> float:
    """
    Returns a scalar in [0.0, 1.0] that smoothly reduces commanded current
    as the motor approaches either soft limit.

    Parameters
    ----------
    pos        : current motor position in steps
    motor_lims : per-motor limit dict  {'home': x, 'min': y, 'max': z}

    Ramp profile:
         min          min+RAMP_ZONE     max-RAMP_ZONE      max
          |─── 0→1 ───|─────── 1.0 ──────────|─── 1→0 ───|
    """
    # Correct lowercase keys from init_dynamic_limits()
    if direction > 0:
        # Moving toward max — only ramp near max limit
        dist = motor_lims['max'] - pos
    else:
        # Moving toward min — only ramp near min limit
        dist = pos - motor_lims['min']

    if dist <= 0:
        return 0.0                  # at or past limit — zero current
    if dist >= RAMP_ZONE:
        return 1.0                  # well inside safe zone — full current
    return dist / RAMP_ZONE         # linear ramp


def safe_current(raw_mA: int, pos: int, motor_lims: dict, direction: int) -> int:
    """
    Scale raw_mA by the ramp factor, then clamp to the hardware ceiling.
    Sign is preserved so motor direction is maintained.

    Parameters
    ----------
    raw_mA     : signed current command (mA)
    pos        : current motor position in steps
    motor_lims : per-motor limit dict  {'home': x, 'min': y, 'max': z}
    direction  : +1 or -1
    """
    factor  = ramp_factor(pos, motor_lims, direction)      # per-motor sub-dict
    scaled  = int(raw_mA * factor)
    clamped = clamp(abs(scaled), 0, CURRENT_CEILING)
    return clamped if raw_mA >= 0 else -clamped


def combined_deviation(positions: dict, all_limits: dict) -> int:
    """
    Sum of absolute deviations from each motor's home position.
    Used for the geometric collision check.

    Parameters
    ----------
    positions  : { motor_id: steps }
    all_limits : full limits dict  { motor_id: {'home':x,'min':y,'max':z} }
    """
    # Access each motor's home from its own sub-dictionary
    return sum(abs(positions[mid] - all_limits[mid]['home']) for mid in ALL_IDS)


def collision_blocked(positions: dict, axis_id: int,
                      direction: int, all_limits: dict) -> bool:
    """
    Returns True if driving motor `axis_id` in `direction` would push the
    combined deviation over COMBINED_DEV_LIMIT.

    Parameters
    ----------
    positions  : { motor_id: steps }
    axis_id    : motor to move
    direction  : +1 or -1
    all_limits : full limits dict  { motor_id: {'home':x,'min':y,'max':z} }
    """
    projected = dict(positions)
    projected[axis_id] += direction * RAMP_ZONE
    # Pass full all_limits dict so combined_deviation can look up each motor
    return combined_deviation(projected, all_limits) > COMBINED_DEV_LIMIT


def print_status(positions: dict, currents: dict, all_limits: dict) -> None:
    """Pretty-print live motor state."""
    print("\n" + "─" * 65)
    print(f"  {'Motor':<10} {'Axis':<14} {'Steps':>7}  {'Angle':>9}  {'Current':>9}")
    print("─" * 65)
    labels = {
        MOTOR_ID_X: "X (fwd/bk)",
        MOTOR_ID_Y: "Y (lateral)",
        MOTOR_ID_Z: "Z (vertical)",
    }
    for mid in ALL_IDS:
        # Fall back to each motor's own home, not a shared 'HOME' key
        pos = positions.get(mid, all_limits[mid]['home'])
        cur = currents.get(mid, 0)
        print(
            f"  ID {mid:<7} {labels[mid]:<14} {pos:>6}   "
            f"{steps_to_deg(pos):>8}°  {cur:>7} mA"
        )
    print("─" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE INITIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def init_port():
    """Open serial port. Exits on failure."""
    port   = portHandlerJoint       # From Constants.py
    packet = packetHandlerJoint     # From Constants.py

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


def init_motors(port, packet) -> dict:
    """
    Set every motor to CURRENT CONTROL MODE and enable torque.
    Returns { motor_id: Dynamixel }.
    """
    motors = {}
    axes   = {MOTOR_ID_X: "X", MOTOR_ID_Y: "Y", MOTOR_ID_Z: "Z"}

    for mid in ALL_IDS:
        print(f"\n[INIT] Motor {mid} ({axes[mid]}-axis) ...")
        m = Dynamixel(packet, port, mid)

        # Disable torque before changing operating mode
        m.disable_torque()

        # Switch to current control mode BEFORE enabling torque
        m.current_control_mode()

        # Enable torque — motor is now live in current control
        m.enable_torque()

        # Start with zero current (coast)
        m.send_torque_cmd(HOLD_CURRENT)

        motors[mid] = m
        print(f"[OK]   Motor {mid} ready in current-control mode.")

    return motors


def init_dynamic_limits(motors: dict) -> dict:
    """
    Read each motor's actual startup position and build
    soft limits centred around it.
    """
    all_limits = {}
    # Pass zeros as initial fallback for very first read
    positions = read_all_positions(motors, {mid: 0 for mid in ALL_IDS})

    for mid in ALL_IDS:
        home = positions[mid]
        all_limits[mid] = {
            'home': home,
            'min' : home - MAX_DELTA_STEPS,
            'max' : home + MAX_DELTA_STEPS,
        }
        print(
            f"[LIMITS] Motor {mid} — "
            f"Home: {steps_to_deg(home):.2f}°  "
            f"Min: {steps_to_deg(home - MAX_DELTA_STEPS):.2f}°  "
            f"Max: {steps_to_deg(home + MAX_DELTA_STEPS):.2f}°"
        )
    return all_limits


def zero_all_currents(motors: dict, currents: dict) -> None:
    """Send zero current to every motor and update the tracking dict."""
    for mid, motor in motors.items():
        motor.send_torque_cmd(0)
        currents[mid] = 0


def read_all_positions(motors: dict, last_positions: dict) -> dict:
    """
    Read present position from every motor.
    Falls back to last known position if a read returns None.

    Parameters
    ----------
    motors         : { motor_id: Dynamixel }
    last_positions : { motor_id: steps } — fallback on read failure
    """
    result = {}
    for mid, motor in motors.items():
        pos = motor.get_present_pos()
        if pos is None:
            # Use last known position instead of crashing
            result[mid] = last_positions.get(mid, 0)
            print(f"[WARNING] Motor {mid} read failed — "
                  f"using last known position: {last_positions.get(mid, 0)}")
        else:
            result[mid] = pos
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  HOMING ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def home_all_motors(motors: dict, packet, port, all_limits: dict) -> dict:
    print("\n[HOME] Switching to position mode for homing ...")

    for mid, motor in motors.items():
        motor.disable_torque()
        motor.extended_pos_mode()
        motor.set_max_velocity(HOME_NUDGE_VEL)
        motor.enable_torque()
        motor.set_goal_position(all_limits[mid]['home'])

    # Initialise fallback positions before the settle loop
    positions = {mid: all_limits[mid]['home'] for mid in ALL_IDS}
    deadline  = time.time() + HOME_SETTLE_TIMEOUT

    while time.time() < deadline:
        # Pass current positions as fallback on read failure
        positions = read_all_positions(motors, positions)
        if all(
            abs(positions[mid] - all_limits[mid]['home']) <= DXL_MOVING_STATUS_THRESHOLD
            for mid in ALL_IDS
        ):
            break
        time.sleep(LOOP_DT)

    print("[HOME] Motors centered — switching back to current control ...")
    for mid, motor in motors.items():
        motor.disable_torque()
        motor.current_control_mode()
        motor.enable_torque()
        motor.send_torque_cmd(HOLD_CURRENT)

    # Pass positions as fallback for final read
    positions = read_all_positions(motors, positions)
    print("[HOME] Done.\n")
    return positions


# ══════════════════════════════════════════════════════════════════════════════
#  SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

def shutdown(motors: dict, port) -> None:
    """Zero currents, disable torques, close port."""
    print("\n[SHUTDOWN] Zeroing currents ...")
    for mid, motor in motors.items():
        motor.send_torque_cmd(0)

    time.sleep(0.1)

    print("[SHUTDOWN] Disabling torques ...")
    for mid, motor in motors.items():
        motor.disable_torque()
        print(f"  Motor {mid} disabled.")

    port.closePort()
    print("[SHUTDOWN] Port closed. Goodbye!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTROL LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("   SEA TURTLE FLIPPER — CURRENT CONTROL MODE")
    print("=" * 60)
    print("""
  Controls:
    W / S   →  Motor 1 (X-axis)   Forward  / Backward
    A / D   →  Motor 2 (Y-axis)   Left     / Right
    I / K   →  Motor 3 (Z-axis)   Up       / Down
    SPACE   →  Zero all currents  (coast to stop)
    P       →  Print live status
    Q / ESC →  Safe shutdown
    """)

    # ── 1. Hardware setup ─────────────────────────────────────────────────────
    port, packet = init_port()
    motors       = init_motors(port, packet)

    # ── 2. Build dynamic limits from actual startup positions ─────────────────
    limits = init_dynamic_limits(motors)

    # ── 3. State tracking ─────────────────────────────────────────────────────
    # Pass per-motor home as initial fallback for first read
    positions = read_all_positions(motors, {mid: limits[mid]['home'] for mid in ALL_IDS})
    currents  = {mid: 0 for mid in ALL_IDS}

    '''
    Old method that used constants
    # ── Key → (motor_id, direction, label) ───────────────────────────────────
    KEY_MAP = {
        chr(WKEY_ASCII_VALUE): (MOTOR_ID_X, 1, "X +"),
        chr(SKEY_ASCII_VALUE): (MOTOR_ID_X, -1, "X -"),
        chr(AKEY_ASCII_VALUE): (MOTOR_ID_Y, -1, "Y -"),
        chr(DKEY_ASCII_VALUE): (MOTOR_ID_Y, 1, "Y +"),
        chr(IKEY_ASCII_VALUE): (MOTOR_ID_Z, 1, "Z +"),
        chr(KKEY_ASCII_VALUE): (MOTOR_ID_Z, -1, "Z -"),
    }
    '''

    # New method for keys that uses keyboard library.
    # ── Key → (motor_id, direction, label) ───────────────────────────────────
    DRIVE_KEYS = {
        'w': (MOTOR_ID_X, 1, "X +"),
        's': (MOTOR_ID_X, -1, "X -"),
        'a': (MOTOR_ID_Y, -1, "Y -"),
        'd': (MOTOR_ID_Y, 1, "Y +"),
        'i': (MOTOR_ID_Z, 1, "Z +"),
        'k': (MOTOR_ID_Z, -1, "Z -"),
    }

    print("[READY] Flipper live. Hold a key to apply current.\n")

    # ── 4. Control loop ───────────────────────────────────────────────────────
    running         = True
    active_key      = None
    last_print_time = 0.0

    while running:
        loop_start = time.time()

        # ── 4a. Read fresh positions — fall back to last known on failure ─────
        positions = read_all_positions(motors, positions)   # pass last positions

        # ── 4b. Keypress handling ─────────────────────────────────────────────

        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            running = False
            continue

        if keyboard.is_pressed('space'):
            zero_all_currents(motors, currents)
            print("[COAST] All currents zeroed.")
            time.sleep(0.2)  # Debounce to avoid multiple prints

        if keyboard.is_pressed('p'):
            # Pass full limits dict
            print_status(positions, currents, limits)
            time.sleep(0.2)  # Debounce to avoid multiple prints

        if keyboard.is_pressed('h'):
            positions = home_all_motors(motors, packet, port, limits)
            # After homing, zero all currents and reset active key
            zero_all_currents(motors, currents)
            active_key = None

        # Check for drive keys being pressed or held down
        active_keys = {
            key for key in DRIVE_KEYS if keyboard.is_pressed(key)
        }

        # ── 4c. Apply current for the active key ──────────────────────────────
        driven_motors = set()

        if active_keys:
            for key in active_keys:
                axis_id, direction, label = DRIVE_KEYS[key]
                pos        = positions[axis_id]
                motor_lims = limits[axis_id]            # per-motor sub-dict

                # ── Per-motor hard limit check ────────────────────────────────────
                # Access lowercase keys from init_dynamic_limits()
                at_min = (pos <= motor_lims['min'] and direction < 0)
                at_max = (pos >= motor_lims['max'] and direction > 0)

                if at_min or at_max:
                    motors[axis_id].send_torque_cmd(0)
                    currents[axis_id] = 0
                    print(
                        f"[LIMIT] Motor {axis_id} at soft limit "
                        f"({steps_to_deg(pos):.1f}°) — current zeroed."
                    )

                # ── Geometric collision check ─────────────────────────────────────
                # Pass full limits dict so combined_deviation works correctly
                elif collision_blocked(positions, axis_id, direction, limits):
                    motors[axis_id].send_torque_cmd(0)
                    currents[axis_id] = 0
                    print(
                        f"[SAFETY] Combined deviation limit reached — "
                        f"move blocked. Press SPACE to coast."
                    )

                else:
                    # ── Compute and send safe current ─────────────────────────────
                    raw_mA = direction * DRIVE_CURRENT
                    # Pass per-motor sub-dict to safe_current
                    cmd_mA = safe_current(raw_mA, pos, motor_lims, direction)

                    motors[axis_id].send_torque_cmd(cmd_mA)
                    currents[axis_id] = cmd_mA

                    driven_motors.add(axis_id)

                    # Zero all other motors
                    for mid in ALL_IDS:
                        if mid not in driven_motors:
                            motors[mid].send_torque_cmd(0)
                            currents[mid] = 0
        else:
            # No drive keys pressed — zero all currents
            zero_all_currents(motors, currents)
            driven_motors.clear()

        # ── 4d. Periodic status print every 2 s ──────────────────────────────
        now = time.time()
        if now - last_print_time >= 2.0:
            print(
                f"[STATUS]  "
                f"X={steps_to_deg(positions[MOTOR_ID_X]):7.2f}°  "
                f"Y={steps_to_deg(positions[MOTOR_ID_Y]):7.2f}°  "
                f"Z={steps_to_deg(positions[MOTOR_ID_Z]):7.2f}°  |  "
                f"Currents: "
                f"M1={currents[MOTOR_ID_X]:>4} mA  "
                f"M2={currents[MOTOR_ID_Y]:>4} mA  "
                f"M3={currents[MOTOR_ID_Z]:>4} mA  "
                f"Driven motors: {', '.join([f'M{i+1}' for i in driven_motors])}"
            )
            last_print_time = now

        # ── 4e. Pace the loop ─────────────────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_t = LOOP_DT - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── 5. Clean shutdown ─────────────────────────────────────────────────────
    shutdown(motors, port)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()