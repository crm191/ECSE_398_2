"""
Identifies the actuator (and, in water, the hydrodynamic) parameters of a flipper from logged runs, and writes them
into the controller's parameter file.

Log runs with hardware_adapter/flipper_log.py (time, motor positions, commanded Goal Current steps). The
flipper_trajectory.py sweeps work well: joint 2 moving about the horizontal carries a large gravity load, which is
what makes the torque constant identifiable. Repeat the sweeps at two or three segment durations (e.g. 0.5, 1 and
2 s) so that viscous and Coulomb friction can be told apart. Keep the commanded current below the ceiling;
saturated samples are dropped.

    # in air: torque constant, armature, viscous and Coulomb friction per joint
    python tools/fit_flipper_params.py --params hardware_adapter/flipper_params.json --side left --medium air \\
        logs/air_*.csv --write

    # in water (after the air fit): added-mass scale and quadratic drag per joint
    python tools/fit_flipper_params.py --params hardware_adapter/flipper_params.json --side left --medium water \\
        logs/water_*.csv --write

Copied next to flipper_params.json (as in the turtle_dynamixel package), --params can be left out:

    python fit_flipper_params.py --medium air logs/air_*.csv --write

Model, per joint j, with the rigid-body torques tau_rb = M ddq + C dq + g of the limb:
    air:   kt * i = tau_rb + J_a ddq + b dq + f_c tanh(dq / v)       (linear in 1/kt, J_a/kt, b/kt, f_c/kt)
    water: kt * i - tau_rb(water, without added mass) - actuator(air fit) = alpha (M_add ddq) + d |dq| dq
For joints whose motion is dominated by inertia (little gravity load), 1/kt and J_a/kt are nearly collinear; there
kt is fixed to the mean of the identifiable joints (the motors are identical) or to the value in the file.
kt also absorbs errors in the link masses of the model (only their ratio to the current is observable), which is
what the controller needs anyway.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from scipy.signal import savgol_filter

# runs from the repository, or standalone next to flipper_model_control.py in the turtle_dynamixel package
try:
    from flipper_model_control import ActuatorModel, ClosedFormDynamics, JointMap
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from turtle_dynamixel.flipper_model_control import ActuatorModel, ClosedFormDynamics, JointMap

LOCAL_PARAMS = Path(__file__).with_name("flipper_params.json")
DEFAULT_PARAMS = (
    LOCAL_PARAMS if LOCAL_PARAMS.exists()
    else Path(__file__).resolve().parents[1] / "hardware_adapter" / "flipper_params.json"
)

MAX_COLLINEARITY = 0.97  # |corr(tau_rb, ddq)| above which kt is not fitted for a joint


def load_log(path: Path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    q = np.array([[float(r[f"q{j}"]) for j in (1, 2, 3)] for r in rows])
    steps = np.array([[float(r[f"i{j}"]) for j in (1, 2, 3)] for r in rows])
    return t - t[0], q, steps


def prepare(t, q_motor, steps, joint_map: JointMap, current_delay: float, window: float):
    """Uniform resampling, zero-phase differentiation and alignment of the commanded current with the motion."""
    dt = np.median(np.diff(t))
    tu = np.arange(t[0], t[-1], dt)
    q_model = np.array([joint_map.position_to_model(qk) for qk in q_motor])
    steps_model = np.array([joint_map.velocity_to_model(sk) for sk in steps])
    q = np.stack([np.interp(tu, t, q_model[:, j]) for j in range(3)], axis=1)
    # the current commanded at t_k acts on the motion over roughly [t_k + latency, t_k + dt + latency]
    i_cmd = np.stack([np.interp(tu - current_delay, t, steps_model[:, j]) for j in range(3)], axis=1)
    n_win = max(5, int(round(window / dt)) | 1)
    q_f = savgol_filter(q, n_win, 3, axis=0)
    dq = savgol_filter(q, n_win, 3, deriv=1, delta=dt, axis=0)
    ddq = savgol_filter(q, n_win, 3, deriv=2, delta=dt, axis=0)
    i_f = savgol_filter(i_cmd, n_win, 3, axis=0)
    trim = slice(n_win, len(tu) - n_win)  # drop the filter's edge effects and the delayed start
    return q_f[trim], dq[trim], ddq[trim], i_f[trim], i_cmd[trim]


def rigid_body_torque(dyn: ClosedFormDynamics, q, dq, ddq):
    return np.array([dyn.M(qk) @ ddqk + dyn.C(qk, dqk) @ dqk + dyn.g(qk) for qk, dqk, ddqk in zip(q, dq, ddq)])


def lstsq(A, y):
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    residual = y - A @ theta
    r2 = 1.0 - np.sum(residual**2) / max(np.sum((y - y.mean()) ** 2), 1e-12)
    return theta, residual, r2


def fit_air(data, side, actuator: ActuatorModel):
    q, dq, ddq, i_amp, unsaturated = data
    tau_rb = rigid_body_torque(ClosedFormDynamics(side, "air"), q, dq, ddq)
    v = actuator.coulomb_velocity
    kt = actuator.kt.copy()
    result = {}
    free_kt = []
    for j in range(3):
        ok = unsaturated[:, j]
        corr = abs(np.corrcoef(tau_rb[ok, j], ddq[ok, j])[0, 1])
        if corr < MAX_COLLINEARITY:
            A = np.stack([tau_rb[ok, j], ddq[ok, j], dq[ok, j], np.tanh(dq[ok, j] / v)], axis=1)
            theta, res, r2 = lstsq(A, i_amp[ok, j])
            kt[j] = 1.0 / theta[0]
            result[j] = dict(kt=kt[j], armature=theta[1] * kt[j], viscous=theta[2] * kt[j], coulomb=theta[3] * kt[j],
                             r2=r2, rms_ma=1000 * np.sqrt(np.mean(res**2)))
            free_kt.append(j)
    kt_shared = np.mean(kt[free_kt]) if free_kt else None
    for j in range(3):
        if j in result:
            continue
        ok = unsaturated[:, j]
        kt[j] = kt_shared if kt_shared is not None else actuator.kt[j]
        A = np.stack([ddq[ok, j], dq[ok, j], np.tanh(dq[ok, j] / v)], axis=1)
        theta, res, r2 = lstsq(A, kt[j] * i_amp[ok, j] - tau_rb[ok, j])
        result[j] = dict(kt=kt[j], armature=theta[0], viscous=theta[1], coulomb=theta[2], r2=r2,
                         rms_ma=1000 * np.sqrt(np.mean(res**2)) / kt[j], kt_fixed=True)
    return result


def fit_water(data, side, actuator: ActuatorModel, params):
    q, dq, ddq, i_amp, unsaturated = data
    dyn = ClosedFormDynamics(side, "water", gravity=params.get("gravity", [0, 0, -9.81]),
                             buoyancy_scale=params.get("buoyancy_scale", 1.0), added_mass_scale=0.0)
    tau_known = rigid_body_torque(dyn, q, dq, ddq) + actuator.armature * ddq + actuator.friction(dq)
    tau_add = np.array([dyn._M_add(qk) @ ddqk + dyn._C_add(qk, dqk) @ dqk for qk, dqk, ddqk in zip(q, dq, ddq)])
    result = {}
    for j in range(3):
        ok = unsaturated[:, j]
        A = np.stack([tau_add[ok, j], np.abs(dq[ok, j]) * dq[ok, j]], axis=1)
        theta, res, r2 = lstsq(A, actuator.kt[j] * i_amp[ok, j] - tau_known[ok, j])
        result[j] = dict(added_mass_scale=theta[0], quadratic_drag=max(0.0, theta[1]), r2=r2,
                         rms_ma=1000 * np.sqrt(np.mean(res**2)) / actuator.kt[j],
                         weight=np.sum(tau_add[ok, j] ** 2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", type=Path, nargs="+", help="CSV logs from hardware_adapter/flipper_log.py")
    parser.add_argument("--params", type=Path, default=DEFAULT_PARAMS,
                        help="parameter file (default: flipper_params.json next to this script, if present)")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    parser.add_argument("--medium", choices=("air", "water"), default="air")
    parser.add_argument("--current-delay", type=float, default=0.008,
                        help="s between logging a command and its mid-point of action (latency + half a period)")
    parser.add_argument("--window", type=float, default=0.15, help="Savitzky-Golay window in s")
    parser.add_argument("--write", action="store_true", help="write the result into --params")
    args = parser.parse_args()

    params = json.loads(args.params.read_text())
    flipper = params["flippers"][args.side]
    joint_map = JointMap(offset=flipper["offset_rad"], sign=flipper["sign"])
    actuator = ActuatorModel(**{k: v for k, v in flipper.get("actuator", {}).items() if not k.startswith("_")})

    parts = [prepare(*load_log(p), joint_map, args.current_delay, args.window) for p in args.logs]
    q, dq, ddq, i_f, i_raw = (np.concatenate(x) for x in zip(*parts))
    unsaturated = np.abs(i_raw) * actuator.current_unit_ma < 0.98 * actuator.current_limit_ma
    data = (q, dq, ddq, i_f * actuator.current_unit_ma / 1000.0, unsaturated)
    print(f"{len(q)} samples from {len(args.logs)} log(s), {100 * np.mean(~unsaturated):.1f}% saturated (dropped)")

    if args.medium == "air":
        result = fit_air(data, args.side, actuator)
        print(f"{'joint':>5s} {'kt [Nm/A]':>10s} {'armature':>10s} {'viscous':>9s} {'coulomb':>9s} {'R^2':>6s} {'RMS [mA]':>9s}")
        for j, r in sorted(result.items()):
            note = " (kt fixed)" if r.get("kt_fixed") else ""
            print(f"{j + 1:5d} {r['kt']:10.3f} {r['armature']:10.5f} {r['viscous']:9.4f} {r['coulomb']:9.4f} "
                  f"{r['r2']:6.3f} {r['rms_ma']:9.1f}{note}")
        for key in ("kt", "armature", "viscous", "coulomb"):
            flipper["actuator"][key] = [float(max(result[j][key], 0.0)) for j in range(3)]
    else:
        result = fit_water(data, args.side, actuator, params)
        print(f"{'joint':>5s} {'added mass scale':>17s} {'drag [Nm s^2]':>14s} {'R^2':>6s} {'RMS [mA]':>9s}")
        for j, r in result.items():
            print(f"{j + 1:5d} {r['added_mass_scale']:17.3f} {r['quadratic_drag']:14.4f} {r['r2']:6.3f} {r['rms_ma']:9.1f}")
        weights = np.array([r["weight"] for r in result.values()])
        scale = float(np.dot(weights, [r["added_mass_scale"] for r in result.values()]) / weights.sum())
        print(f"added-mass scale (weighted over joints): {scale:.3f}")
        params["added_mass_scale"] = max(scale, 0.0)
        flipper["quadratic_drag"] = [float(result[j]["quadratic_drag"]) for j in range(3)]

    if args.write:
        args.params.write_text(json.dumps(params, indent=4) + "\n")
        print(f"Updated {args.params}")


if __name__ == "__main__":
    main()
