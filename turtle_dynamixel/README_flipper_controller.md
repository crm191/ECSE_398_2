# New model-based flipper controller: drop-in for `turtle_dynamixel`

This replaces `torque_control.py` with a controller built on the **current** flipper design. It compensates the
inertia, Coriolis and gravity of the new limb, and in water also buoyancy, added mass and drag. The model is
generated from the CAD of the new limb. The old `mass.py` described an older design and is no longer used.

`flipper_trajectory.py` and `flipper_main_multi.py` keep working: `torque_control(q, dq, qd, dqd, ddqd, Kp, KD)`
has the same signature and still returns a (6, 1) array.

## What's in the zip

| File | Purpose |
|---|---|
| `torque_control.py` | Replaces your `torque_control.py`. Returns Goal Current **steps**, ready to send. |
| `flipper_model_control.py` | Controller, joint map, motor model |
| `flipper_dynamics_generated.py` | Dynamics of the new limb, generated from CAD (don't edit) |
| `flipper_params.json` | All settings: calibration, gains, air/water, motor parameters |
| `flipper_log.py` | Logger for the identification runs (step 5) |
| `fit_flipper_params.py` | Fits the motor parameters from your logs (step 5) |
| `zero_pose_left.png`, `zero_pose_right.png` | Reference pictures for calibration (step 2) |

Requirements: numpy and scipy, which you already have. No new packages.

## 1. Install

1. Back up your current `torque_control.py`, e.g. rename it to `torque_control_old.py`.
2. Unzip everything into the `turtle_dynamixel/` folder, next to `flipper_trajectory.py`.

`mass.py` can stay; nothing imports it anymore except `mass_torque_test.py`.

## 2. Calibrate the joint map (do this before running in current mode)

The controller works in the joint angles of the CAD model. `flipper_params.json` tells it how your motor
readings map to those angles. Edit the `"left"` entry under `"flippers"`. If you're testing the right flipper,
set `"left"` to `"enabled": false` and fill in `"right"`.

Open `zero_pose_left.png`. In the **zero pose** the flipper sticks straight out sideways, with the blade flat and
horizontal. The three joints are:
- **joint 1** turns about the vertical axis (sweeps the flipper forward/back);
- **joint 2** turns about the fore-aft axis (flaps it up/down);
- **joint 3** twists the flipper about its own length.

1. **`slots`**: for joints 1, 2, 3 in that order, which row of `q` the motor sits in. With
   `ALL_IDS = [1, 2, 3]`, row 0 is motor 1, row 1 is motor 2 and row 2 is motor 3. If motor 1 is joint 1,
   motor 2 is joint 2 and motor 3 is joint 3, keep `[0, 1, 2]`. If instead motor 3 is the vertical-axis joint,
   motor 1 the fore-aft one and motor 2 the twist, use `[2, 0, 1]`.
2. **`offset_rad`**: with torque disabled, put the flipper in the zero pose by hand. Read the three positions
   the way `flipper_trajectory.py` does (radians, from `read_all_positions`) and enter them in joint order 1, 2, 3.
3. **`sign`**: from the zero pose, turn each motor so that its reading increases. The caption of the picture
   gives the positive direction of each joint (right-hand rule about the listed axis). If the flipper moves that
   way, the sign is `1`; if it moves the opposite way, it is `-1`.

## 3. Gravity check (first time in current mode)

Before tracking anything, check that the model holds the flipper up on its own. Keep your 300 mA ceiling.
In your control loop, call the controller with the desired state equal to the current state and zero gains:

```python
zeros6 = np.zeros((6, 1))
steps = torque_control(q_vec, dq_vec, q_vec, zeros6, zeros6, np.zeros((6, 6)), np.zeros((6, 6)))
# send steps[i, 0] to motor ALL_IDS[i] (through your clamp / ramp checks as usual)
```

Move the flipper by hand to a few poses and let go:
- **Correct:** it stays roughly where you leave it and drifts only slowly, since friction isn't compensated
  yet. Holding it horizontal takes about 0.5 N·m on joint 2, roughly 215 mA.
- **Wrong:** it drops, or pushes itself upward. Then an offset, a sign or the slot order is wrong. Stop
  (ESC zeroes the currents) and re-check step 2.

## 4. Run your trajectories

`flipper_trajectory.py` runs as is:
- `Kp`/`Kd` keep their meaning (N·m/rad, N·m·s/rad). Your `diag(5, 4, 5)` / `diag(0.3, 0.3, 0.4)` worked well
  in simulation. Don't multiply by 150 anymore; `torque_control` already converts to current steps.
- **Recommended change:** `compute_single_motor_torque` currently sends **zero** current to the two motors
  that aren't being swept, so those joints aren't held against gravity. Send them the computed command
  instead: delete the `if mid != active_mid: ... continue` block and keep the limit checks. Their `qd` is
  already their current position, so they will just hold still.
- About your note on `C·dq` vs `C·dq_d` in the old code: using the desired velocity is correct. That
  controller is known as "PD+", and it is stable when C is computed properly from M. It's the default mode here
  (`"mode": "pd_plus"`).

## 5. Identify the motor parameters (air)

The defaults for motor inertia and friction are estimates. Fitting them from a few runs makes tracking
noticeably better.

1. Log the sweeps with `flipper_log.py`:
   ```python
   from flipper_log import FlipperLog
   log = FlipperLog()
   # each loop, after sending the currents:
   log.add(time.time(), [positions[m] for m in ALL_IDS], [velocities[m] for m in ALL_IDS],
           [torque_cmds[m] for m in ALL_IDS])
   # at the end:
   log.save("air_sweep_1s.csv")
   ```
2. Repeat with `SEGMENT_DURATION` = 0.5, 1 and 2 s. Different speeds are needed to separate the two kinds of
   friction. If the current hits the ceiling often, reduce the amplitude.
3. Fit and save into `flipper_params.json`:
   ```bash
   python fit_flipper_params.py --medium air air_sweep_*.csv --write
   ```
   It prints the fitted values and the fit quality (R² should be above about 0.9). Joint 2 carries the gravity
   load, so its torque constant is the reliable one; the other joints reuse it.

## 6. In water

Set `"medium": "water"` in `flipper_params.json`. This adds buoyancy, which carries most of the limb's weight,
plus the added mass of the blade and drag. After the air fit, log the same sweeps in water and run:

```bash
python fit_flipper_params.py --medium water water_sweep_*.csv --write
```

Fast strokes in water need more torque than 300 mA gives. In simulation the swimming stroke saturated 10–15% of
the time at that ceiling. Raise it carefully once everything above works.

## Settings reference (`flipper_params.json`)

- `mode`: `pd_plus` (default) or `desired_ff`, which evaluates the model only along the reference and is
  slightly less sensitive to velocity noise. `computed_torque` uses different gain units; don't use it without
  retuning.
- `medium`: `air` or `water`.
- `gravity`: leave as is unless your test rig holds the shoulder tilted.
- `kp`, `kd`: used only if you call `torque_control(..., Kp=None, KD=None)`.
- `actuator.current_limit_ma`: commands are clipped to this, in addition to your own clamp.
