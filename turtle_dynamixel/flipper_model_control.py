"""
Model-based joint-space control of one front flipper of the turtle (three XW540-T260 Dynamixels in current mode).

    tau = rigid-body / hydrodynamic model terms + actuator model terms + PD feedback

The dynamics come from either backend:
    - ClosedFormDynamics: turtle_ctrl/flipper_dynamics_generated.py, numpy only (for the robot's computer)
    - FlipperDynamics (turtle_ctrl/flipper_model.py): MuJoCo on mujoco_models/turtle.xml, including MuJoCo's
      ellipsoid fluid model in water
Both use the joint convention of turtle.xml; JointMap converts motor angles to it.
"""
__all__ = [
    "ActuatorModel",
    "ClosedFormDynamics",
    "FlipperModelController",
    "JointMap",
    "controller_from_params",
    "make_dynamics",
]
from dataclasses import dataclass, field
import numpy as np
from typing import Dict, Optional, Sequence, Tuple

try:
    import flipper_dynamics_generated as generated
except ImportError:
    from turtle_dynamixel import flipper_dynamics_generated as generated

CONTROL_MODES = ("pd_plus", "computed_torque", "desired_ff")


def _vec3(x) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    return np.repeat(x, 3) if x.size == 1 else x


@dataclass
class JointMap:
    """
    Maps the three motor angles of a flipper to the joints of turtle.xml: q_model = sign * (q_motor - offset).
    Arguments:
        offset: motor angles (rad, as read from the Dynamixels) when the flipper is in the zero pose of turtle.xml
        sign: +1 if the motor and the model joint turn the same way, -1 otherwise
    """

    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    sign: np.ndarray = field(default_factory=lambda: np.ones(3))

    def __post_init__(self):
        self.offset = _vec3(self.offset)
        self.sign = _vec3(self.sign)
        assert np.all(np.abs(self.sign) == 1.0), "sign must be +1 or -1"

    def position_to_model(self, q_motor: np.ndarray) -> np.ndarray:
        return self.sign * (np.asarray(q_motor, dtype=float).reshape(-1) - self.offset)

    def position_to_motor(self, q_model: np.ndarray) -> np.ndarray:
        return self.sign * np.asarray(q_model, dtype=float).reshape(-1) + self.offset

    def velocity_to_model(self, dq_motor: np.ndarray) -> np.ndarray:
        # also used for accelerations and, since sign = 1 / sign, to map torques back to the motors
        return self.sign * np.asarray(dq_motor, dtype=float).reshape(-1)

    def gains_to_model(self, k_motor: np.ndarray) -> np.ndarray:
        S = np.diag(self.sign)
        return S @ np.asarray(k_motor, dtype=float) @ S


@dataclass
class ActuatorModel:
    """
    Torque/current relation and losses of the XW540-T260 joints.
    Arguments:
        kt: torque constant at the output shaft in Nm/A. Default from the datasheet stall point (10.6 Nm at 4.4 A).
        current_unit_ma: mA per Goal Current step (2.69 mA for the XW540-T260)
        current_limit_ma: commanded currents are clipped to +/- this value
        armature: reflected rotor and gearbox inertia in kg m^2 (272.5:1 gearbox). The default is a rough estimate;
            identify it with tools/fit_flipper_params.py.
        viscous: viscous friction in Nm s/rad. Zero until identified.
        coulomb: Coulomb friction in Nm. Zero until identified.
        coulomb_velocity: velocity in rad/s over which the Coulomb term is smoothed (tanh)
    """

    kt: np.ndarray = field(default_factory=lambda: np.full(3, 2.4))
    current_unit_ma: float = 2.69
    current_limit_ma: np.ndarray = field(default_factory=lambda: np.full(3, 300.0))
    armature: np.ndarray = field(default_factory=lambda: np.full(3, 0.005))
    viscous: np.ndarray = field(default_factory=lambda: np.zeros(3))
    coulomb: np.ndarray = field(default_factory=lambda: np.zeros(3))
    coulomb_velocity: float = 0.05

    def __post_init__(self):
        for name in ("kt", "current_limit_ma", "armature", "viscous", "coulomb"):
            setattr(self, name, _vec3(getattr(self, name)))

    def friction(self, dq: np.ndarray) -> np.ndarray:
        """Friction torques for velocities of shape (3,) or (n, 3)."""
        dq = np.asarray(dq, dtype=float)
        return self.viscous * dq + self.coulomb * np.tanh(dq / self.coulomb_velocity)

    def torque_to_steps(self, tau: np.ndarray) -> np.ndarray:
        """Joint torques (Nm, motor direction) to Goal Current register steps, clipped to the current limit."""
        current_ma = np.asarray(tau, dtype=float).reshape(-1) / self.kt * 1000.0
        current_ma = np.clip(current_ma, -self.current_limit_ma, self.current_limit_ma)
        return np.round(current_ma / self.current_unit_ma).astype(int)

    def steps_to_torque(self, steps: np.ndarray) -> np.ndarray:
        return np.asarray(steps, dtype=float).reshape(-1) * self.current_unit_ma / 1000.0 * self.kt


class ClosedFormDynamics:
    """
    Same interface as turtle_ctrl.flipper_model.FlipperDynamics, evaluated with the generated closed-form model.
    In water the fluid forces are a diagonal quadratic drag, -quadratic_drag * |dq| * dq.
    """

    def __init__(
        self,
        side: str = "left",
        medium: str = "air",
        gravity: Sequence[float] = (0.0, 0.0, -9.81),
        buoyancy_scale: float = 1.0,
        added_mass_scale: float = 1.0,
        quadratic_drag: Optional[Sequence[float]] = None,
    ):
        assert side in ("left", "right"), f"Unknown side {side}"
        assert medium in ("air", "water"), f"Unknown medium {medium}"
        self.side = side
        self.medium = medium
        self.gravity = np.asarray(gravity, dtype=float)
        self.buoyancy_scale = buoyancy_scale
        self.added_mass_scale = added_mass_scale
        prefix = side.upper()
        self._M = getattr(generated, f"mass_matrix_{side}")
        self._C = getattr(generated, f"coriolis_matrix_{side}")
        self._G = getattr(generated, f"gravity_torque_{side}")
        self._M_add = getattr(generated, f"added_mass_matrix_{side}")
        self._C_add = getattr(generated, f"added_mass_coriolis_matrix_{side}")
        self.link_mass = getattr(generated, f"{prefix}_LINK_MASS")
        self.link_com = getattr(generated, f"{prefix}_LINK_COM")
        self.buoyant_mass = getattr(generated, f"{prefix}_BUOYANT_MASS")
        self.cob = getattr(generated, f"{prefix}_COB")
        if quadratic_drag is None:
            quadratic_drag = getattr(generated, f"{prefix}_QUADRATIC_DRAG")
        self.quadratic_drag = _vec3(quadratic_drag) if medium == "water" else np.zeros(3)

    def M(self, q: np.ndarray) -> np.ndarray:
        M = self._M(q)
        if self.medium == "water":
            M = M + self.added_mass_scale * self._M_add(q)
        return M

    def C(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        C = self._C(q, dq)
        if self.medium == "water":
            C = C + self.added_mass_scale * self._C_add(q, dq)
        return C

    def g(self, q: np.ndarray) -> np.ndarray:
        g = self._G(q, self.link_mass, self.link_com, self.gravity)
        if self.medium == "water":
            g = g + self._G(q, -self.buoyancy_scale * self.buoyant_mass, self.cob, self.gravity)
        return g

    def fluid_force(self, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
        dq = np.asarray(dq, dtype=float).reshape(-1)
        return -self.quadratic_drag * np.abs(dq) * dq


def make_dynamics(backend: str = "closed_form", side: str = "left", medium: str = "air", **kwargs):
    """
    Arguments:
        backend: "closed_form" (numpy only) or "mujoco"
        side, medium, kwargs: passed to ClosedFormDynamics or FlipperDynamics
    """
    if backend == "closed_form":
        kwargs.pop("model_path", None)
        return ClosedFormDynamics(side=side, medium=medium, **kwargs)
    if backend == "mujoco":
        try:
            from flipper_model import FlipperDynamics
        except ImportError:
            from turtle_ctrl.flipper_model import FlipperDynamics
        kwargs.pop("quadratic_drag", None)  # MuJoCo uses its own fluid model
        return FlipperDynamics(side=side, medium=medium, **kwargs)
    raise ValueError(f"Unknown backend {backend}")


class FlipperModelController:
    """
    Model-based tracking controller for one flipper, in the joint convention of turtle.xml.

    With e = q_d - q and de = dq_d - dq, the control modes are:
        pd_plus:         tau = M(q) ddq_d + C(q, dq) dq_d + g(q) + Kp e + Kd de
        computed_torque: tau = M(q) (ddq_d + Kp e + Kd de) + C(q, dq) dq + g(q)
        desired_ff:      tau = M(q_d) ddq_d + C(q_d, dq_d) dq_d + g(q_d) + Kp e + Kd de
    and all of them add the actuator terms armature * a + friction(dq_d) and cancel the fluid forces, where a is
    the commanded acceleration (ddq_d, or ddq_d + Kp e + Kd de for computed_torque).

    pd_plus (Paden & Panja, 1988) keeps the desired velocity in the Coriolis term, as in the students' change to
    torque_control.py. It is globally stable because C comes from the Christoffel symbols of M. desired_ff evaluates
    the model only along the reference, so noise in the measured velocity does not enter the feedforward.
    Gains are in Nm/rad and Nm s/rad for pd_plus and desired_ff, and in 1/s^2 and 1/s for computed_torque.

    Friction and fluid forces are always compensated at the desired velocity. Cancelling them at the measured
    velocity removes the damping of the flipper using a quantized, delayed signal, and in simulation made
    computed_torque unstable in water.

    Arguments:
        dynamics: ClosedFormDynamics or FlipperDynamics
        actuator: ActuatorModel
        kp, kd: feedback gains as arrays of shape (3,) (diagonal) or (3, 3)
        mode: one of CONTROL_MODES
    """

    def __init__(self, dynamics, actuator: Optional[ActuatorModel] = None, kp=5.0, kd=0.3, mode: str = "pd_plus"):
        assert mode in CONTROL_MODES, f"Unknown control mode {mode}"
        self.dynamics = dynamics
        self.actuator = ActuatorModel() if actuator is None else actuator
        self.kp = self._gain(kp)
        self.kd = self._gain(kd)
        self.mode = mode

    @staticmethod
    def _gain(k) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        return np.diag(_vec3(k)) if k.ndim < 2 else k

    def torque(
        self, q, dq, q_d, dq_d, ddq_d, kp: Optional[np.ndarray] = None, kd: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Joint torques in Nm (turtle.xml convention) for the state (q, dq) and the reference (q_d, dq_d, ddq_d),
        all arrays of shape (3,). kp and kd override the gains of the controller for this call.
        Returns the torques of shape (3,) and a dictionary with the individual terms.
        """
        q, dq, q_d, dq_d, ddq_d = (np.asarray(x, dtype=float).reshape(-1) for x in (q, dq, q_d, dq_d, ddq_d))
        kp = self.kp if kp is None else self._gain(kp)
        kd = self.kd if kd is None else self._gain(kd)
        e = q_d - q
        de = dq_d - dq
        dyn = self.dynamics

        if self.mode == "pd_plus":
            acc = ddq_d
            model = dyn.M(q) @ ddq_d + dyn.C(q, dq) @ dq_d + dyn.g(q) - dyn.fluid_force(q, dq_d)
            feedback = kp @ e + kd @ de
        elif self.mode == "computed_torque":
            acc = ddq_d + kp @ e + kd @ de
            model = dyn.M(q) @ acc + dyn.C(q, dq) @ dq + dyn.g(q) - dyn.fluid_force(q, dq_d)
            feedback = np.zeros(3)
        else:
            acc = ddq_d
            model = dyn.M(q_d) @ ddq_d + dyn.C(q_d, dq_d) @ dq_d + dyn.g(q_d) - dyn.fluid_force(q_d, dq_d)
            feedback = kp @ e + kd @ de
        actuator = self.actuator.armature * acc + self.actuator.friction(dq_d)
        tau = model + actuator + feedback
        return tau, dict(tau=tau, model=model, actuator=actuator, feedback=feedback, e=e, de=de)


def controller_from_params(params: Dict, side: str) -> Tuple[FlipperModelController, JointMap, Dict]:
    """
    Builds the controller and joint map of one flipper from a parameter dictionary in the format of
    hardware_adapter/flipper_params.json.
    Returns the controller, the joint map and the flipper's entry of params["flippers"].
    """
    flipper = params["flippers"][side]
    dynamics = make_dynamics(
        backend=params.get("backend", "closed_form"),
        side=side,
        medium=params.get("medium", "air"),
        gravity=params.get("gravity", [0.0, 0.0, -9.81]),
        buoyancy_scale=params.get("buoyancy_scale", 1.0),
        added_mass_scale=params.get("added_mass_scale", 1.0),
        quadratic_drag=flipper.get("quadratic_drag"),
        model_path=params.get("model_path"),
    )
    actuator = ActuatorModel(**{k: v for k, v in flipper.get("actuator", {}).items() if not k.startswith("_")})
    joint_map = JointMap(offset=flipper.get("offset_rad", np.zeros(3)), sign=flipper.get("sign", np.ones(3)))
    controller = FlipperModelController(
        dynamics,
        actuator,
        kp=params.get("kp", 5.0),
        kd=params.get("kd", 0.3),
        mode=params.get("mode", "pd_plus"),
    )
    return controller, joint_map, flipper
