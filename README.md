# ECSE 398 Team 3 - Sea Turtle Flipper Hardware Implementation
## Project Overview

This project implements model-based current control of a three-motor
sea turtle flipper using Dynamixel XW540-T260 servo motors. The system
supports keyboard operation, automated position trajectory execution,
real-time force/torque sensing, and system identification for actuator
parameter fitting.

## File Descriptions

### `flipper_params.json`
Central parameter file for the entire control system. Defines all
hardware and controller settings in one place so no code changes are
needed when tuning the system. Contains:
- **Control settings**: backend (`closed_form` or `mujoco`), control
  mode (`pd_plus`, `computed_torque`, `desired_ff`), operating medium
  (`air` or `water`), gravity vector, and buoyancy/added-mass scales.
- **PD gains**: per-joint proportional (`kp`) and derivative (`kd`)
  gains in Nm/rad and Nm·s/rad.
- **Flipper blocks** (`left` and `right`): joint slot assignments,
  home position offsets (`offset_rad`), joint direction signs, and
  per-joint actuator parameters (`kt`, `current_unit_ma`,
  `current_limit_ma`, `armature`, `viscous`, `coulomb` friction).

Set `"enabled": true` on the flipper side matching your physical
hardware. All other files read from this file at startup — it is the
single source of truth for every gain, limit, and hardware constant
in the project.

---

### `Dynamixel.py`
Low-level driver class for a single Dynamixel XW540-T260 motor.
Wraps the Dynamixel SDK to provide a clean Python interface for one
motor at a time. Key responsibilities:
- Setting operating modes: extended position, current control,
  velocity control.
- Enabling and disabling motor torque.
- Writing goal position, goal velocity, and goal current commands.
- Reading present position (with two's complement sign handling and
  communication error protection) and present velocity.
- Setting PID gains and profile velocity limits.

Used by `flipper_main.py` for single-motor control. Superseded by
`Mod.py` for multi-motor simultaneous control, but retained for
individual motor testing and diagnostics.

---

### `Mod.py`
Group synchronisation driver for controlling multiple Dynamixel motors
simultaneously using `GroupSyncRead` and `GroupSyncWrite`. Unlike
`Dynamixel.py` which commands one motor per packet, `Mod` batches all
reads and writes into single bus transactions, reducing latency and
ensuring all motors receive their commands at the same time. Key
responsibilities:
- Initialising `GroupSyncRead` for simultaneous position and velocity
  reads across all motors.
- Sending current commands to all motors in one `GroupSyncWrite` packet
  via `send_torque_cmd()`.
- Setting operating modes (current, velocity, extended position) and
  torque enable/disable for all motors.
- Reading present positions in radians and velocities in rad/s with
  retry logic and byte-level sign correction.

Used by `flipper_main_multi.py` and `flipper_trajectory.py` as the
hardware communication layer.

---

### `Constants.py`
Hardware configuration file containing all Dynamixel register addresses,
operating mode values, and physical constants for the turtle robot. Also
initialises the serial port handler (`portHandlerJoint`) and packet
handler (`packetHandlerJoint`) that are shared across all control files.
Key contents:
- Register addresses for torque enable, operating mode, goal current,
  goal position, present position, PID gains, and watchdog.
- Operating mode constants: extended position, current control, velocity
  control.
- Physical robot constants: hexagon plate circumradius, spool radius,
  link lengths, mass properties, and reference motor angles.
- Per-joint torque limits (`max_torque`, `xw_max_torque`) and keyboard
  ASCII key codes for teleoperation.

---

### `dyn_functions.py`
Utility library of mathematical and kinematic helper functions used by
the cable-driven continuum robot control system. Key functions:
- `grab_motor_lengths()`: converts state variables to desired cable
  lengths with clamping to physical limits.
- `grab_q()` / `grab_arm_q()`: compute generalised coordinates from
  cable lengths using the Piecewise Constant Curvature (PCC) model.
- `grab_qd()`: builds the desired generalised coordinate vector from
  curvature, bending angle, and arc length inputs.
- `grab_cable_lens()` / `grab_arm_cable_lens()`: compute current cable
  lengths from motor positions using spool kinematics.
- `grab_current()` / `grab_arm_current()`: clamp raw torque commands
  to safe hardware limits.
- `to_motor_steps()` / `to_radians()` / `to_rad_secs()`: unit
  conversion utilities between motor steps, radians, and rad/s.
- `get_time()` / `diff()` / `get_qindex()`: timing and numerical
  differentiation utilities for the control loop.
- `mat2np()`: loads `.mat` files into NumPy arrays for trajectory
  reference data.

---

### `flipper_main.py`
Keyboard teleoperation program for the sea turtle flipper using
**individual motor current control**. Each motor is assigned to one
axis and driven by a commanded current while position is monitored as
a safety guard only. Key features:
- Each `Dynamixel` instance is set to current control mode at startup.
- Soft position limits are built dynamically from each motor's actual
  startup position (`init_dynamic_limits()`).
- A three-layer safety system prevents collisions: per-motor hard
  limits, a direction-aware current ramp zone near limits, and a
  geometric combined-deviation check across all three axes.
- Key release is detected in real time using the `keyboard` library so
  current is zeroed immediately when a key is released.
- A homing routine temporarily switches to extended position mode,
  drives all motors to their home positions, then returns to current
  control mode.

**Controls**: `W/S` Motor 1, `A/D` Motor 2, `I/K` Motor 3,
`SPACE` coast, `Q/ESC` shutdown.

---

### `flipper_main_multi.py`
Keyboard teleoperation program using the `Mod` class for simultaneous
multi-motor current control via `GroupSyncRead` and `GroupSyncWrite`.
Functionally equivalent to `flipper_main.py` but all three motor
positions are read in a single bus transaction and all current commands
are sent in a single bus transaction each loop iteration. Key features:
- Positions are returned in radians directly from `Mod.get_position()`.
- All soft limits, ramp factors, and collision checks operate in
  radians to match the position units.
- Multiple keys can be held simultaneously to drive two or three motors
  at once. Conflicting keys on the same axis are detected and resolved
  by `resolve_conflicts()`, zeroing that motor while leaving others
  unaffected.
- The `to_unsigned_16()` helper converts signed current steps to the
  unsigned 16-bit two's complement format required by `GroupSyncWrite`.
- A `2 ms` bus settling delay between write and the next read prevents
  packet collisions at 4,000,000 baud.

**Controls**: same as `flipper_main.py`. Hold multiple keys for
simultaneous multi-motor motion.

---

### `flipper_trajectory.py`
Automated trajectory execution program that moves each motor through a
pre-planned position trajectory using the computed torque + PD
controller. No keyboard input is required during the run. Key features:
- Generates smooth `home → lower → upper → home` trajectories for each
  motor using cosine interpolation with analytically derived velocity
  and acceleration feedforward profiles.
- Calls `torque_control()` every loop iteration to compute physically
  informed current commands from the full inverse dynamics model plus
  PD feedback.
- Records position, velocity, current, and F/T sensor data at every
  control step and saves them to timestamped CSV files.
- Produces two separate figures after each run: one for position and
  velocity tracking, one for force, torque, and current.
- All parameters (gains, limits, motor offsets, medium) are read from
  `flipper_params.json` at startup via `load_flipper_params()`.
- An ESC emergency stop using `keyboard.add_hotkey()` immediately zeros
  all currents and initiates a safe shutdown at any point during the
  run. Any data recorded before the stop is still saved and plotted.

**Run order**: Motor 1 → Motor 2 → Motor 3 → home → shutdown → plot.

---

