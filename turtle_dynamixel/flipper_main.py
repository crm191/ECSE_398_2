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
    H       →  Home  (position-control nudge back to centre, then resume)
    P       →  Print live positions + active currents
    Q / ESC →  Safe shutdown
"""

import os
import sys
import time
import math

'''
# ─── Path setup ───────────────────────────────────────────────────────────────
submodule = (
    os.path.expanduser("~")
    + "/drl-turtle/ros2_ws/src/turtle_hardware/turtle_hardware/turtle_dynamixel"
)
sys.path.append(submodule)

'''

# ────── Path Setup - Windows ──────────────────────────────────────────────────────────────────
#sys.path.append(r"C:\Users\charr\OneDrive - Case Western Reserve University\Documents\ECSE 398\venv\Lib\site-packages")

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
# Home is the electrical mid-point of the motor.
# ±10° (1024 steps) either side keeps the flipper well clear of hard stops
# and prevents adjacent motors from clashing.
#HOME_STEPS      = 2048          # 180° — mechanical mid-point
MAX_DELTA_STEPS = 114          # ±10° soft travel limit per axis

#For static limits
#LIMIT_MIN = HOME_STEPS - MAX_DELTA_STEPS    # 1024 steps
#LIMIT_MAX = HOME_STEPS + MAX_DELTA_STEPS    # 3072 steps

# When a motor is within this many steps of a soft limit, current is ramped
# down linearly to zero so the motor coasts to a gentle stop.
RAMP_ZONE = 200     # steps — width of the deceleration band near each limit

# ── Current commands (mA) ─────────────────────────────────────────────────────
# Dynamixel XM/XH series: 1 mA resolution on Goal Current register.
# Keep well below the motor's rated stall current to protect the mechanism.
DRIVE_CURRENT   = 150           # mA applied while a key is held
HOLD_CURRENT    = 0             # mA when no key is pressed (coast / free)

# Absolute ceiling — hardware will never receive more than this value
# regardless of any calculation.  Matches xw_max_torque in Constants.py.
CURRENT_CEILING = 300           # mA

# ── Collision safety ──────────────────────────────────────────────────────────
# The sum of per-axis deviations from home must stay below this threshold.
# Prevents "corner" configurations where all three axes are simultaneously
# near their extremes and the flipper plates could contact each other.
COMBINED_DEV_LIMIT = int(1.5 * MAX_DELTA_STEPS)    # 1536 steps

# ── Timing ────────────────────────────────────────────────────────────────────
LOOP_HZ     = 20                # Control-loop rate
LOOP_DT     = 1.0 / LOOP_HZ    # 50 ms per iteration

# ── Home-nudge parameters (brief position-mode move to re-centre) ─────────────
HOME_NUDGE_VEL      = 20        # Profile velocity during homing
HOME_SETTLE_TIMEOUT = 5.0       # Seconds before homing gives up

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer between lo and hi."""
    return max(lo, min(hi, value))


def steps_to_deg(steps: int) -> float:
    """Convert motor steps to degrees for display."""
    return round(steps * (360.0 / 4096.0), 2)


def ramp_factor(pos: int, limits: dict) -> float:
    """
    Returns a scalar in [0.0, 1.0] that smoothly reduces the commanded
    current as the motor approaches either soft limit.

    The factor is 1.0 (full current) in the safe interior and ramps
    linearly to 0.0 at the limit boundary.

         LIMIT_MIN          LIMIT_MAX
             |  RAMP_ZONE  |           |  RAMP_ZONE  |
             0 ──── ramp ──── 1.0 ──────── ramp ──── 0
    """
    # Distance from each soft limit
    dist_lo = pos - limits['MIN']
    dist_hi = limits['MAX'] - pos

    nearest = min(dist_lo, dist_hi)        # distance to the closer limit

    if nearest <= 0:
        return 0.0                         # at or past the limit — zero current
    if nearest >= RAMP_ZONE:
        return 1.0                         # well inside safe zone — full current

    return nearest / RAMP_ZONE            # linear ramp


def safe_current(raw_mA: int, pos: int, limits: dict) -> int:
    """
    Scale raw_mA by the ramp factor at the current position, then clamp
    to the hardware ceiling.  Sign is preserved so direction is maintained.

    Parameters
    ----------
    raw_mA : signed current command (mA) — may be positive or negative
    pos    : current motor position in steps
    limits : dictionary containing soft limits for each motor

    Returns
    -------
    Signed, safety-clamped current in mA ready to send to send_torque_cmd().
    """
    factor  = ramp_factor(pos, limits)
    scaled  = int(raw_mA * factor)
    clamped = clamp(abs(scaled), 0, CURRENT_CEILING)
    return clamped if raw_mA >= 0 else -clamped


def combined_deviation(positions: dict, limits: dict) -> int:
    """
    Sum of absolute deviations from HOME across all three axes.
    Used for the geometric collision check.
    """
    return sum(abs(positions[mid] - limits['HOME']) for mid in ALL_IDS)


def collision_blocked(positions: dict, axis_id: int, direction: int, limits: dict) -> bool:
    """
    Returns True if driving motor `axis_id` in `direction` (+1 or -1)
    would push the combined deviation over the safety threshold.

    Parameters
    ----------
    positions : dict  { motor_id: current_steps }
    axis_id   : which motor we want to move
    direction : +1 (away from home on that axis) or -1 (toward home)
    limits    : dictionary containing soft limits for each motor
    """
    # Estimate where the motor will be after one RAMP_ZONE worth of travel
    projected = dict(positions)
    projected[axis_id] += direction * RAMP_ZONE
    return combined_deviation(projected, limits) > COMBINED_DEV_LIMIT


def print_status(positions: dict, currents: dict, limits: dict) -> None:
    """Pretty-print live motor state."""
    print("\n" + "─" * 65)
    print(f"  {'Motor':<10} {'Axis':<14} {'Steps':>7}  {'Angle':>9}  {'Current':>9}")
    print("─" * 65)
    labels = {MOTOR_ID_X: "X (fwd/bk)",
              MOTOR_ID_Y: "Y (lateral)",
              MOTOR_ID_Z: "Z (vertical)"}
    for mid in ALL_IDS:
        pos = positions.get(mid, limits['HOME'])
        cur = currents.get(mid, 0)
        print(
            f"  ID {mid:<7} {labels[mid]:<14} {pos:>6}   "
            f"{steps_to_deg(pos):>8}°  {cur:>7} mA"
        )
    print("─" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
#  HARDWARE INITIALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def init_port():
    """Open serial port.  Exits on failure."""
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

        # Switch to current control mode BEFORE enabling torque
        m.current_control_mode()

        # Enable torque — motor is now live in current control
        m.enable_torque()

        # Start with zero current (coast)
        m.send_torque_cmd(HOLD_CURRENT)

        motors[mid] = m
        print(f"[OK]   Motor {mid} ready in current-control mode.")

    return motors

# Figure out if needed. Depends on home position
def init_dynamic_limits(motors: dict) -> dict:
    """
    Read each motor's actual startup position and build
    soft limits centered around it.
    """
    limits = {}
    positions = read_all_positions(motors)
    for mid in ALL_IDS:
        home = positions[mid]
        limits[mid] = {
            "home"  : home,
            "min"   : home - MAX_DELTA_STEPS,
            "max"   : home + MAX_DELTA_STEPS,
        }
        print(
            f"[LIMITS] Motor {mid} — "
            f"Home: {steps_to_deg(home):.2f}°  "
            f"Min: {steps_to_deg(home - MAX_DELTA_STEPS):.2f}°  "
            f"Max: {steps_to_deg(home + MAX_DELTA_STEPS):.2f}°"
        )
    return limits


def zero_all_currents(motors: dict, currents: dict) -> None:
    """Send zero current to every motor and update the tracking dict."""
    for mid, motor in motors.items():
        motor.send_torque_cmd(0)
        currents[mid] = 0


def read_all_positions(motors: dict) -> dict:
    """Read present position from every motor. Returns { id: steps }."""
    return {mid: motor.get_present_pos() for mid, motor in motors.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  HOMING ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def home_all_motors(motors: dict, packet, port, limits: dict) -> dict:
    """
    Briefly switch to Extended Position Mode to drive all motors back to
    HOME_STEPS, then return to Current Control Mode.

    Returns the refreshed positions dict (all ≈ HOME_STEPS).
    """
    print("\n[HOME] Switching to position mode for homing ...")

    for mid, motor in motors.items():
        motor.disable_torque()
        motor.extended_pos_mode()
        motor.set_max_velocity(HOME_NUDGE_VEL)
        motor.enable_torque()
        motor.set_goal_position(limits['HOME'])

    # Wait for all motors to settle
    deadline = time.time() + HOME_SETTLE_TIMEOUT
    while time.time() < deadline:
        positions = read_all_positions(motors)
        if all(abs(positions[mid] - limits['HOME']) <= DXL_MOVING_STATUS_THRESHOLD
               for mid in ALL_IDS):
            break
        time.sleep(LOOP_DT)

    print("[HOME] Motors centred — switching back to current control ...")

    for mid, motor in motors.items():
        motor.disable_torque()
        motor.current_control_mode()
        motor.enable_torque()
        motor.send_torque_cmd(HOLD_CURRENT)

    positions = read_all_positions(motors)
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
    H       →  Home all motors    (re-centre)
    P       →  Print live status
    Q / ESC →  Safe shutdown
    """)

    # ── 1. Hardware setup ─────────────────────────────────────────────────────
    port, packet = init_port()
    motors       = init_motors(port, packet)
    limits = init_dynamic_limits(motors)

    # ── 2. State tracking ─────────────────────────────────────────────────────
    # positions: last-known motor positions in steps
    # currents : last commanded current per motor in mA
    positions = read_all_positions(motors)
    currents  = {mid: 0 for mid in ALL_IDS}

    # Key → (motor_id, sign) mapping
    # sign +1 means positive current (motor moves in + direction)
    # sign -1 means negative current
    KEY_MAP = {
        chr(WKEY_ASCII_VALUE): (MOTOR_ID_X, +1, "X +"),
        chr(SKEY_ASCII_VALUE): (MOTOR_ID_X, -1, "X −"),
        chr(AKEY_ASCII_VALUE): (MOTOR_ID_Y, -1, "Y −"),
        chr(DKEY_ASCII_VALUE): (MOTOR_ID_Y, +1, "Y +"),
        chr(IKEY_ASCII_VALUE): (MOTOR_ID_Z, +1, "Z +"),
        chr(CKEY_ASCII_VALUE): (MOTOR_ID_Z, -1, "Z −"),
    }

    print("[READY] Flipper live. Hold a key to apply current.\n")

    # ── 3. Control loop ───────────────────────────────────────────────────────
    running         = True
    active_key      = None      # key currently being held
    last_print_time = 0.0

    while running:
        loop_start = time.time()

        # ── 3a. Read fresh positions from all motors ──────────────────────────
        positions = read_all_positions(motors)

        # ── 3b. Check for a new keypress ──────────────────────────────────────
        if kbhit():
            key = getch()

            # ── Quit ──────────────────────────────────────────────────────────
            if key in (chr(QKEY_ASCII_VALUE), chr(ESC_ASCII_VALUE)):
                running = False
                continue

            # ── Coast / stop ──────────────────────────────────────────────────
            elif key == chr(SPACE_ASCII_VALUE):
                zero_all_currents(motors, currents)
                active_key = None
                print("[COAST] All currents zeroed.")

            # ── Home ──────────────────────────────────────────────────────────
            #elif key == chr(BKEY_ASCII_VALUE):      # 'b' mapped to home (H)
            #    zero_all_currents(motors, currents)
            #    active_key = None
            #    positions  = home_all_motors(motors, packet, port)

            # ── Status print ──────────────────────────────────────────────────
            elif key == chr(PKEY_ASCII_VALUE):
                print_status(positions, currents, limits)

            # ── Drive key ─────────────────────────────────────────────────────
            elif key in KEY_MAP:
                active_key = key

            # ── Any other key releases the active drive ───────────────────────
            else:
                active_key = None
                zero_all_currents(motors, currents)

        # ── 3c. Apply current for the active key ──────────────────────────────
        if active_key and active_key in KEY_MAP:
            axis_id, direction, label = KEY_MAP[active_key]

            # ── Per-motor hard limit check ────────────────────────────────────
            pos = positions[axis_id]
            at_min = (pos <= limits['MIN'] and direction < 0)
            at_max = (pos >= limits['MAX'] and direction > 0)

            if at_min or at_max:
                # Motor is at its individual soft limit — zero its current
                motors[axis_id].send_torque_cmd(0)
                currents[axis_id] = 0
                print(
                    f"[LIMIT] Motor {axis_id} at soft limit "
                    f"({steps_to_deg(pos):.1f}°) — current zeroed."
                )

            # ── Geometric collision check ─────────────────────────────────────
            elif collision_blocked(positions, axis_id, direction, limits):
                motors[axis_id].send_torque_cmd(0)
                currents[axis_id] = 0
                print(
                    f"[SAFETY] Combined deviation limit reached — "
                    f"move blocked. Press SPACE or H to re-centre."
                )

            else:
                # ── Compute and send safe current ─────────────────────────────
                raw_mA  = direction * DRIVE_CURRENT
                cmd_mA  = safe_current(raw_mA, pos, limits)

                motors[axis_id].send_torque_cmd(cmd_mA)
                currents[axis_id] = cmd_mA

                # Keep all OTHER motors at zero (coast)
                for other_id in ALL_IDS:
                    if other_id != axis_id:
                        motors[other_id].send_torque_cmd(0)
                        currents[other_id] = 0

        # ── 3d. Periodic auto-print every 2 s ─────────────────────────────────
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
                f"M3={currents[MOTOR_ID_Z]:>4} mA"
            )
            last_print_time = now

        # ── 3e. Pace the loop ─────────────────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_t = LOOP_DT - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── 4. Clean shutdown ─────────────────────────────────────────────────────
    shutdown(motors, port)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()