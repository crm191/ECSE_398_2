#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
flipper_main_multi.py  —  Multi-Motor Current-Control Edition
--------------------------------------------------------------
Sea Turtle Flipper controller using the Mod class for GroupSyncRead
and GroupSyncWrite. All motor positions are read simultaneously and
all torque commands are sent simultaneously each loop iteration.

This program utilizes current control mode to drive each Dynamixel motor with a set current (mA). 
Position is read back continuously and used ONLY as a safety guard — if a motor approaches its
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
"""

import os
import sys
import time
import math
import keyboard

# ─── Project imports ──────────────────────────────────────────────────────────
from dynamixel_sdk import *
from Mod import *           # Mod class + all address constants
from Constants import *     # portHandlerJoint, packetHandlerJoint, JOINTS, etc.

# ══════════════════════════════════════════════════════════════════════════════
#  FLIPPER CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

# ── Motor IDs (front flipper only) ───────────────────────────────────────────
MOTOR_ID_X  = 1        # Forward / backward stroke
MOTOR_ID_Y  = 2        # Lateral sweep
MOTOR_ID_Z  = 3        # Vertical pitch

ALL_IDS     = [MOTOR_ID_X, MOTOR_ID_Y, MOTOR_ID_Z]

# ── Position limits in RADIANS ────────────────────────────────────────────────
# Mod.get_position() returns radians so all limits must match.
# Conversion: steps × (2π / 4096) = radians
#   114 steps ≈ 10°  ≈ 0.1749 rad
#    50 steps ≈  4.4° ≈ 0.0767 rad
STEPS_TO_RAD  = (2 * math.pi) / 4096
RAD_TO_STEPS  = 4096 / (2 * math.pi)

MAX_DELTA_RAD = 114 * STEPS_TO_RAD     # ±10° soft travel limit per axis
RAMP_ZONE_RAD =  50 * STEPS_TO_RAD     # deceleration band near each limit

# Home threshold for settling check during homing
HOME_THRESHOLD_RAD = DXL_MOVING_STATUS_THRESHOLD * STEPS_TO_RAD

# ── Current commands ──────────────────────────────────────────────────────────
# XW540-T260: 1 register step = 2.69 mA
CURRENT_UNIT_MA  = 2.69         # mA per register step
DRIVE_CURRENT_MA = 50          # desired drive current  (mA)
CEILING_MA       = 150          # desired ceiling        (mA)

DRIVE_CURRENT    = int(DRIVE_CURRENT_MA / CURRENT_UNIT_MA)  # ≈ 56 steps
CURRENT_CEILING  = int(CEILING_MA       / CURRENT_UNIT_MA)  # ≈ 186 steps
HOLD_CURRENT     = 0

# ── Collision safety ──────────────────────────────────────────────────────────
# Each motor can use its full individual range before combined check triggers.
COMBINED_DEV_LIMIT = 3.0 * MAX_DELTA_RAD   # ≈ 0.5236 rad

# ── Timing ────────────────────────────────────────────────────────────────────
LOOP_HZ = 500
LOOP_DT = 1.0 / LOOP_HZ            # 2.0 ms per iteration

# ── Homing ────────────────────────────────────────────────────────────────────
HOME_NUDGE_VEL      = 20            # profile velocity during homing
HOME_SETTLE_TIMEOUT = 5.0           # seconds before homing gives up

# ── Key → (index in ALL_IDS, direction, label) ───────────────────────────────
# Index is used to look up motor ID from ALL_IDS list.
DRIVE_KEYS = {
    'w': (0, +1, "X +"),   # Motor 1 forward
    's': (0, -1, "X -"),   # Motor 1 backward
    'a': (1, -1, "Y -"),   # Motor 2 left
    'd': (1, +1, "Y +"),   # Motor 2 right
    'i': (2, +1, "Z +"),   # Motor 3 up
    'k': (2, -1, "Z -"),   # Motor 3 down
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer between lo and hi."""
    return max(lo, min(hi, value))


def rad_to_deg(rad: float) -> float:
    """Convert radians to degrees for display."""
    return round(math.degrees(rad), 2)


def to_unsigned_16(val: int) -> int:
    """
    Convert a signed integer to its unsigned 16-bit two's complement.
    Required because GroupSyncWrite expects unsigned values.
        e.g.  -56 → 65536 + (-56) = 65480
    """
    if val < 0:
        return 65536 + val
    return val


def ramp_factor(pos: float, motor_lims: dict, direction: int) -> float:
    """
    Direction-aware ramp scalar in [0.0, 1.0].
    Only ramps near the limit being approached so the motor can
    always move freely away from a limit.

    Parameters
    ----------
    pos        : current motor position  (radians)
    motor_lims : {'home': r, 'min': r, 'max': r}
    direction  : +1 or -1
    """
    if direction > 0:
        dist = motor_lims['max'] - pos      # distance to max limit
    else:
        dist = pos - motor_lims['min']      # distance to min limit

    if dist <= 0:
        return 0.0                          # at or past the limit
    if dist >= RAMP_ZONE_RAD:
        return 1.0                          # well clear — full current
    return dist / RAMP_ZONE_RAD            # linear ramp


def safe_current(raw: int, pos: float, motor_lims: dict, direction: int) -> int:
    """
    Scale raw current by the direction-aware ramp factor then clamp
    to the hardware ceiling. Sign is preserved.

    Parameters
    ----------
    raw        : signed current command (register steps)
    pos        : current motor position (radians)
    motor_lims : {'home': r, 'min': r, 'max': r}
    direction  : +1 or -1
    """
    factor  = ramp_factor(pos, motor_lims, direction)
    scaled  = int(raw * factor)
    clamped = clamp(abs(scaled), 0, CURRENT_CEILING)
    return clamped if raw >= 0 else -clamped


def combined_deviation(positions: dict, all_limits: dict) -> float:
    """
    Sum of absolute deviations from each motor's home position (radians).
    Used for the geometric collision check.
    """
    return sum(abs(positions[mid] - all_limits[mid]['home']) for mid in ALL_IDS)


def collision_blocked(positions: dict, idx: int,
                      direction: int, all_limits: dict) -> bool:
    """
    Returns True if moving the motor at ALL_IDS[idx] in direction would
    push the combined deviation over COMBINED_DEV_LIMIT.

    Parameters
    ----------
    positions  : { motor_id: radians }
    idx        : index into ALL_IDS (0=X, 1=Y, 2=Z)
    direction  : +1 or -1
    all_limits : { motor_id: {'home':r, 'min':r, 'max':r} }
    """
    mid       = ALL_IDS[idx]
    projected = dict(positions)
    projected[mid] += direction * RAMP_ZONE_RAD
    return combined_deviation(projected, all_limits) > COMBINED_DEV_LIMIT


def print_status(positions: dict, currents: dict, all_limits: dict) -> None:
    """Pretty-print live motor state."""
    print("\n" + "─" * 72)
    print(f"  {'Motor':<10} {'Axis':<14} {'Angle':>10}  {'Home':>10}  {'Current':>12}")
    print("─" * 72)
    labels = {
        MOTOR_ID_X: "X (fwd/bk)",
        MOTOR_ID_Y: "Y (lateral)",
        MOTOR_ID_Z: "Z (vertical)",
    }
    for mid in ALL_IDS:
        pos  = positions.get(mid, all_limits[mid]['home'])
        cur  = currents.get(mid, 0)
        home = all_limits[mid]['home']
        cur_mA = round(cur * CURRENT_UNIT_MA, 1)
        print(
            f"  ID {mid:<7} {labels[mid]:<14} "
            f"{rad_to_deg(pos):>9}°  "
            f"{rad_to_deg(home):>9}°  "
            f"{cur:>5} ({cur_mA:>6} mA)"
        )
    print("─" * 72 + "\n")


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


def init_mod(port, packet) -> 'Mod':
    """
    Create a Mod instance for all front flipper motors.
    Disables torque first, sets current control mode, then enables torque.

    Returns
    -------
    Mod instance ready for current control.
    """
    print(f"\n[INIT] Initialising Mod with motors {ALL_IDS} ...")
    mod = Mod(packet, port, ALL_IDS)

    # Always disable torque before changing operating mode
    mod.disable_torque()

    # Set all motors to current control mode
    mod.set_current_cntrl_mode()

    # Enable torque — motors are now live
    mod.enable_torque()

    # Start at zero current (coast)
    mod.send_torque_cmd([HOLD_CURRENT] * len(ALL_IDS))

    print("[OK] Mod ready in current-control mode.\n")
    return mod
    
def read_all_positions(mod: 'Mod', last_positions: dict) -> dict:
    """
    Read all motor positions via GroupSyncRead.
    Falls back per-motor to last_positions on any failure.
    
    Parameters
    ----------
    mod            : Mod instance
    last_positions : { motor_id: radians } — fallback on failure
    """
    try:
        pos_list = mod.get_position()

        # Check 1 — entire read failed
        if pos_list is None:
            print("[WARNING] GroupSyncRead returned None — "
                  "using all last known positions.")
            return last_positions

        # Check 2 — partial failure — some motors returned None
        result = {}
        for i, mid in enumerate(ALL_IDS):
            if i >= len(pos_list) or pos_list[i] is None:
                print(f"[WARNING] Motor {mid} position unavailable — "
                      f"using last known: "
                      f"{rad_to_deg(last_positions.get(mid, 0.0)):.2f}°")
                result[mid] = last_positions.get(mid, 0.0)
            else:
                result[mid] = pos_list[i]
        return result

    except Exception as e:
        print(f"[WARNING] Position read exception ({e}) — "
              f"using all last known positions.")
        return last_positions

def init_dynamic_limits(mod: 'Mod') -> dict:
    """
    Read each motor's actual startup position and build soft limits
    centred around it (in radians).

    Returns
    -------
    { motor_id: {'home': radians, 'min': radians, 'max': radians} }
    """
    all_limits = {}
    positions  = read_all_positions(mod, {mid: 0.0 for mid in ALL_IDS})

    for mid in ALL_IDS:
        home = positions[mid]
        all_limits[mid] = {
            'home': home,
            'min' : home - MAX_DELTA_RAD,
            'max' : home + MAX_DELTA_RAD,
        }
        print(
            f"[LIMITS] Motor {mid} — "
            f"Home: {rad_to_deg(home):.2f}°  "
            f"Min:  {rad_to_deg(home - MAX_DELTA_RAD):.2f}°  "
            f"Max:  {rad_to_deg(home + MAX_DELTA_RAD):.2f}°"
        )
    return all_limits


def zero_all_currents(mod: 'Mod', currents: dict) -> None:
    """Send zero current to all motors via GroupSyncWrite."""
    mod.send_torque_cmd([HOLD_CURRENT] * len(ALL_IDS))
    for mid in ALL_IDS:
        currents[mid] = 0


# ══════════════════════════════════════════════════════════════════════════════
#  HOMING ROUTINE
# ══════════════════════════════════════════════════════════════════════════════

def home_all_motors(mod: 'Mod', port, packet, all_limits: dict) -> dict:
    """
    Briefly switch to Extended Position Mode to drive all motors back to
    their individual home positions, then return to Current Control Mode.

    Uses direct packet writes for mode switching to avoid the hardcoded
    ID bugs in Mod.set_extended_pos_mode() and Mod.send_pos_cmd().

    Returns
    -------
    Refreshed positions dict { motor_id: radians }.
    """
    print("\n[HOME] Switching to position mode for homing ...")

    mod.disable_torque()

    # Set extended position mode directly — avoids hardcoded IDs in Mod
    for mid in ALL_IDS:
        dxl_comm_result, dxl_error = packet.write1ByteTxRx(
            port, mid, ADDR_OPERATING_MODE, EXT_POSITION_CONTROL_MODE
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ERROR] Motor {mid} mode switch failed: "
                  f"{packet.getTxRxResult(dxl_comm_result)}")

    # Limit homing speed
    mod.set_max_velocity(HOME_NUDGE_VEL)
    mod.enable_torque()

    # Send home position to each motor directly — avoids num_motors=6 bug
    for mid in ALL_IDS:
        home_steps = int(all_limits[mid]['home'] * RAD_TO_STEPS)
        dxl_comm_result, dxl_error = packet.write4ByteTxRx(
            port, mid, ADDR_GOAL_POSITION, home_steps
        )
        if dxl_comm_result != COMM_SUCCESS:
            print(f"[ERROR] Motor {mid} home command failed: "
                  f"{packet.getTxRxResult(dxl_comm_result)}")

    # Wait for all motors to settle at their home positions
    positions = {mid: all_limits[mid]['home'] for mid in ALL_IDS}
    deadline  = time.time() + HOME_SETTLE_TIMEOUT

    while time.time() < deadline:
        positions = read_all_positions(mod, positions)
        if all(
            abs(positions[mid] - all_limits[mid]['home']) <= HOME_THRESHOLD_RAD
            for mid in ALL_IDS
        ):
            break
        time.sleep(LOOP_DT)

    print("[HOME] Motors centred — switching back to current control ...")

    mod.disable_torque()

    # Switch back to current control directly — avoids hardcoded IDs in Mod
    for mid in ALL_IDS:
        packet.write1ByteTxRx(
            port, mid, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE
        )

    mod.enable_torque()
    mod.send_torque_cmd([HOLD_CURRENT] * len(ALL_IDS))

    positions = read_all_positions(mod, positions)
    print("[HOME] Done.\n")
    return positions


# ══════════════════════════════════════════════════════════════════════════════
#  SHUTDOWN
# ══════════════════════════════════════════════════════════════════════════════

def shutdown(mod: 'Mod', port) -> None:
    """Zero all currents, disable all torques, close port."""
    print("\n[SHUTDOWN] Zeroing currents ...")
    mod.send_torque_cmd([HOLD_CURRENT] * len(ALL_IDS))
    time.sleep(0.1)

    print("[SHUTDOWN] Disabling torques ...")
    mod.disable_torque()

    port.closePort()
    print("[SHUTDOWN] Port closed. Goodbye!\n")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CONTROL LOOP
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("   SEA TURTLE FLIPPER — MULTI-MOTOR CURRENT CONTROL")
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
    mod          = init_mod(port, packet)

    # ── 2. Build dynamic limits from actual startup positions ─────────────────
    limits = init_dynamic_limits(mod)

    # ── 3. State tracking ─────────────────────────────────────────────────────
    # positions : { motor_id: radians } — updated each loop via GroupSyncRead
    # currents  : { motor_id: steps   } — last commanded value per motor
    positions = read_all_positions(mod, {mid: limits[mid]['home'] for mid in ALL_IDS})
    currents  = {mid: 0 for mid in ALL_IDS}

    print("[READY] Flipper live. Hold a key to apply current.\n")

    # ── 4. Control loop ───────────────────────────────────────────────────────
    running         = True
    active_keys      = None
    last_print_time = 0.0

    while running:
        loop_start = time.time()

        # ── 4a. Read ALL positions via GroupSyncRead ──────────────────────────
        positions = read_all_positions(mod, positions)

        # ── 4b. Check quit and special keys ──────────────────────────────────
        if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
            running = False
            continue

        if keyboard.is_pressed('space'):
            zero_all_currents(mod, currents)
            print("[COAST] All currents zeroed.")
            time.sleep(0.2)     # debounce
            continue

        if keyboard.is_pressed('p'):
            print_status(positions, currents, limits)
            time.sleep(0.2)     # debounce
            continue

        # ── 4c. Build active keys from currently held keys ────────────────────
        # keyboard.is_pressed() reflects real-time physical key state
        active_keys = {
            key for key in DRIVE_KEYS
            if keyboard.is_pressed(key)
        }

        # ── 4d. Build full torque command dict for ALL motors ─────────────────
        # GroupSyncWrite requires a command for every motor simultaneously.
        # Default all to zero — override below for driven motors.
        torque_cmds   = {mid: 0 for mid in ALL_IDS}
        driven_motors = set()

        # ── 4e. Process active keys ───────────────────────────────────────────
        if active_keys:
            for key in active_keys:
                idx, direction, label = DRIVE_KEYS[key]
                mid        = ALL_IDS[idx]
                pos        = positions[mid]
                motor_lims = limits[mid]

                # ── Per-motor hard limit check ────────────────────────────────
                at_min = (pos <= motor_lims['min'] and direction < 0)
                at_max = (pos >= motor_lims['max'] and direction > 0)

                if at_min or at_max:
                    torque_cmds[mid] = 0
                    print(
                        f"[LIMIT] Motor {mid} at soft limit "
                        f"({rad_to_deg(pos):.1f}°) — current zeroed."
                    )

                # ── Geometric collision check ─────────────────────────────────
                elif collision_blocked(positions, idx, direction, limits):
                    torque_cmds[mid] = 0
                    print(
                        f"[SAFETY] Combined deviation limit reached — "
                        f"Motor {mid} blocked."
                    )

                else:
                    # ── Compute safe current ──────────────────────────────────
                    raw_mA = direction * DRIVE_CURRENT
                    cmd_mA = safe_current(raw_mA, pos, motor_lims, direction)
                    torque_cmds[mid] = cmd_mA
                    driven_motors.add(mid)

        # ── 4f. Limit enforcement — runs every loop regardless of key state ───
        # ── 4f. No keys pressed — zero all currents ───────────────────────────
        else:
            # No drive keys pressed — zero all torque commands and clear tracking
            torque_cmds = {mid: 0 for mid in ALL_IDS}
            driven_motors.clear()

        # ── 4g. Convert signed to unsigned and send ALL motors at once ────────
        # GroupSyncWrite sends one packet for all motors simultaneously
        # to_unsigned_16 converts negative steps to unsigned 16-bit two's
        # complement so GroupSyncWrite transmits the correct signed value.
        torque_list = [
            to_unsigned_16(torque_cmds[mid]) for mid in ALL_IDS
        ]
        mod.send_torque_cmd(torque_list)

        # Update current tracking dict (store signed values for display)
        for mid in ALL_IDS:
            currents[mid] = torque_cmds[mid]

        # ── 4h. Periodic status print every 2 s ──────────────────────────────
        now = time.time()
        if now - last_print_time >= 2.0:
            print(
                f"[STATUS]  "
                f"X={rad_to_deg(positions[MOTOR_ID_X]):7.2f}°  "
                f"Y={rad_to_deg(positions[MOTOR_ID_Y]):7.2f}°  "
                f"Z={rad_to_deg(positions[MOTOR_ID_Z]):7.2f}°  |  "
                f"Currents (steps): "
                f"M1={currents[MOTOR_ID_X]:>5}  "
                f"M2={currents[MOTOR_ID_Y]:>5}  "
                f"M3={currents[MOTOR_ID_Z]:>5}"
            )
            last_print_time = now

        # ── 4i. Pace the loop ─────────────────────────────────────────────────
        elapsed = time.time() - loop_start
        sleep_t = LOOP_DT - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    # ── 5. Clean shutdown ─────────────────────────────────────────────────────
    shutdown(mod, port)


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()