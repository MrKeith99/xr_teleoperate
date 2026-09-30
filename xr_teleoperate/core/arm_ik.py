"""Dual-arm IK for any G1 body x end effector, built on the sim's composed MJCF.

The model is the composed robot pruned to its arm joints on a fixed pelvis (every other joint welded at
zero, like xr_teleoperate's reduced URDF models), so targets are in the pelvis frame. The tool frame of
each side is the end effector's `{side}_tcp` site: the point that follows the operator's tracked wrist.
The bare wrist (`none`) has no parts file, so its site is placed at the body's end-effector mount.

The cost is upstream's: position, rotation, regularization and smoothness terms solved with IPOPT. The
29dof arm solves 7 DoF per side; the 23dof arm 5 DoF, weighted toward position since it cannot follow
the full wrist orientation.
"""

from __future__ import annotations

import contextlib
import ctypes
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from .embodiment import BODIES, END_EFFECTORS, SIDES, load_compose, sim_root

ARM_JOINTS: dict[str, tuple[str, ...]] = {
    "29dof": ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"),
    "23dof": ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll"),
}
# The bare wrist's tool point in the mount frame, as the parts files' `{side}_tcp` sites.
BARE_WRIST_TCP_POS = (0.05, 0.0, 0.0)
_DROPPED_SECTIONS = ("asset", "actuator", "sensor", "equality", "contact", "tendon", "keyframe", "visual")
_DROPPED_ELEMENTS = ("geom", "site", "camera", "light", "freejoint")


@contextlib.contextmanager
def _openmp_threads(n: int):
    """Limit OpenMP threads on this thread (restored on exit): IPOPT's MUMPS on a many-core machine is
    ~200x slower with the default thread count, which can't be set by env once OpenMP is loaded."""
    try:
        gomp = ctypes.CDLL("libgomp.so.1")
    except OSError:
        yield
        return
    previous = gomp.omp_get_max_threads()
    gomp.omp_set_num_threads(n)
    try:
        yield
    finally:
        gomp.omp_set_num_threads(previous)


@dataclass(frozen=True)
class IKWeights:
    position: float = 50.0
    rotation: float = 1.0
    # Upstream's 0.02 pulls reachable targets 5-7 mm toward q = 0; the smoothness term keeps solutions continuous.
    regularization: float = 0.002
    smoothness: float = 0.1


DEFAULT_WEIGHTS: dict[str, IKWeights] = {"29dof": IKWeights(), "23dof": IKWeights(rotation=0.5)}


@dataclass(frozen=True)
class IKSolution:
    q: np.ndarray
    tau_ff: np.ndarray
    converged: bool
    position_error: np.ndarray  # (2,) m, left and right


def arm_joint_names(body: str) -> tuple[str, ...]:
    """MJCF arm joint names, left arm then right arm."""
    return tuple(f"{side}_{joint}_joint" for side in SIDES for joint in ARM_JOINTS[body])


def _quat_wxyz_to_rot(text: str | None) -> np.ndarray:
    if not text:
        return np.eye(3)
    w, x, y, z = (float(v) for v in text.split())
    return pin.Quaternion(w, x, y, z).normalized().toRotationMatrix()


def _vec(text: str | None) -> np.ndarray:
    return np.array([float(v) for v in (text or "0 0 0").split()])


def arm_mjcf(body: str, end_effector: str, root: str | Path | None = None) -> tuple[ET.Element, dict[str, tuple[str, pin.SE3]]]:
    """The pruned arm MJCF, and each side's tool frame as (body name, placement in that body)."""
    compose = load_compose(sim_root(root))
    mjcf = compose.compose_model(body, end_effector, "fixed", "d435i")
    parents = {child: parent for parent in mjcf.iter() for child in parent}

    tools = {}
    for side in SIDES:
        site = mjcf.find(f'.//site[@name="{side}_tcp"]')
        if site is not None:
            placement = pin.SE3(_quat_wxyz_to_rot(site.get("quat")), _vec(site.get("pos")))
            tools[side] = (parents[site].get("name"), placement)
        elif end_effector == "none":
            wrist, offset = compose.BODY_MOUNTS[body]
            tools[side] = (wrist.format(side=side), pin.SE3(np.eye(3), np.add(offset, BARE_WRIST_TCP_POS)))
        else:
            raise ValueError(f"End effector {end_effector!r} has no {side}_tcp site")

    keep = set(arm_joint_names(body))
    for section in _DROPPED_SECTIONS:
        for element in mjcf.findall(section):
            mjcf.remove(element)
    for parent in list(mjcf.iter()):
        for child in list(parent):
            if child.tag in _DROPPED_ELEMENTS or (child.tag == "joint" and child.get("name") not in keep):
                parent.remove(child)
    pelvis = mjcf.find('.//body[@name="pelvis"]')
    pelvis.set("pos", "0 0 0")
    pelvis.attrib.pop("quat", None)
    return mjcf, tools


def build_arm_model(body: str, end_effector: str, root: str | Path | None = None) -> tuple[pin.Model, dict[str, int]]:
    """The fixed-pelvis arm model with a `{side}_tcp` frame per side; returns (model, frame ids)."""
    if body not in BODIES:
        raise ValueError(f"Unknown body {body!r}; expected one of {BODIES}")
    if end_effector not in END_EFFECTORS:
        raise ValueError(f"Unknown end effector {end_effector!r}; expected one of {END_EFFECTORS}")
    mjcf, tools = arm_mjcf(body, end_effector, root)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "arms.xml"
        path.write_text(ET.tostring(mjcf, encoding="unicode"))
        model = pin.buildModelFromMJCF(str(path))

    names = tuple(model.names[i] for i in range(1, model.njoints))
    if names != arm_joint_names(body):
        raise RuntimeError(f"Unexpected arm joints for {body}: {names}")
    frame_ids = {}
    for side, (body_name, placement) in tools.items():
        body_frame = model.frames[model.getFrameId(body_name, pin.FrameType.BODY)]
        frame = pin.Frame(
            f"{side}_tcp", body_frame.parentJoint, body_frame.placement * placement, pin.FrameType.OP_FRAME
        )
        frame_ids[side] = model.addFrame(frame)
    return model, frame_ids


class WeightedMovingFilter:
    """Weighted mean of the last len(weights) solutions, newest first (xr_teleoperate's IK smoothing)."""

    def __init__(self, weights: tuple[float, ...], size: int):
        self.weights = np.asarray(weights, dtype=float)
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError("Filter weights must sum to 1")
        self.size = size
        self._queue: list[np.ndarray] = []

    def reset(self) -> None:
        self._queue.clear()

    def __call__(self, value: np.ndarray) -> np.ndarray:
        self._queue.insert(0, np.asarray(value, dtype=float))
        del self._queue[len(self.weights) :]
        if len(self._queue) < len(self.weights):
            return self._queue[0]
        return np.einsum("i,ij->j", self.weights, np.stack(self._queue))


class G1ArmIK:
    """Both arms of one G1 body carrying one end effector; targets are 4x4 tool poses in the pelvis frame."""

    def __init__(
        self,
        body: str,
        end_effector: str,
        *,
        sim_root: str | Path | None = None,
        weights: IKWeights | None = None,
        filter_weights: tuple[float, ...] | None = (0.4, 0.3, 0.2, 0.1),
        max_iter: int = 30,
    ):
        self.body = body
        self.end_effector = end_effector
        self.model, self.frame_ids = build_arm_model(body, end_effector, sim_root)
        self.data = self.model.createData()
        self.joint_names = arm_joint_names(body)
        self.nq = self.model.nq
        self.weights = weights or DEFAULT_WEIGHTS[body]
        self._filter = WeightedMovingFilter(filter_weights, self.nq) if filter_weights else None
        self._build_problem(max_iter)
        self.reset()

    def _build_problem(self, max_iter: int) -> None:
        cmodel = cpin.Model(self.model)
        cdata = cmodel.createData()
        cq = casadi.SX.sym("q", self.nq)
        targets = {side: casadi.SX.sym(f"tf_{side}", 4, 4) for side in SIDES}
        cpin.framesForwardKinematics(cmodel, cdata, cq)
        pos_err = casadi.vertcat(
            *(cdata.oMf[self.frame_ids[s]].translation - targets[s][:3, 3] for s in SIDES)
        )
        rot_err = casadi.vertcat(
            *(cpin.log3(cdata.oMf[self.frame_ids[s]].rotation @ targets[s][:3, :3].T) for s in SIDES)
        )
        args = [cq, targets["left"], targets["right"]]
        self._pos_err = casadi.Function("pos_err", args, [pos_err])
        rot_err_fn = casadi.Function("rot_err", args, [rot_err])

        opti = casadi.Opti()
        self._var_q = opti.variable(self.nq)
        self._param_q_last = opti.parameter(self.nq)
        self._param_tf = {side: opti.parameter(4, 4) for side in SIDES}
        fn_args = [self._var_q, self._param_tf["left"], self._param_tf["right"]]
        w = self.weights
        opti.subject_to(opti.bounded(self.model.lowerPositionLimit, self._var_q, self.model.upperPositionLimit))
        opti.minimize(
            w.position * casadi.sumsqr(self._pos_err(*fn_args))
            + w.rotation * casadi.sumsqr(rot_err_fn(*fn_args))
            + w.regularization * casadi.sumsqr(self._var_q)
            + w.smoothness * casadi.sumsqr(self._var_q - self._param_q_last)
        )
        opti.solver(
            "ipopt",
            {"expand": True, "detect_simple_bounds": True, "calc_lam_p": False, "print_time": False},
            {
                "sb": "yes",
                "print_level": 0,
                "max_iter": max_iter,
                "tol": 1e-4,
                "acceptable_tol": 5e-4,
                "acceptable_iter": 5,
                "warm_start_init_point": "yes",
            },
        )
        self._opti = opti

    def reset(self, q: np.ndarray | None = None) -> None:
        """Restart from `q` (default: zeros), e.g. the robot's measured arm pose."""
        self._q_last = np.zeros(self.nq) if q is None else np.asarray(q, dtype=float).copy()
        if self._filter is not None:
            self._filter.reset()

    def forward(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Each side's 4x4 tool pose in the pelvis frame."""
        pin.framesForwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        return {side: self.data.oMf[fid].homogeneous.copy() for side, fid in self.frame_ids.items()}

    def gravity_torques(self, q: np.ndarray) -> np.ndarray:
        return pin.computeGeneralizedGravity(self.model, self.data, np.asarray(q, dtype=float)).copy()

    def solve(
        self, left_target: np.ndarray, right_target: np.ndarray, q_current: np.ndarray | None = None
    ) -> IKSolution:
        """Arm joint targets (left then right) reaching both tool poses as closely as the costs allow."""
        q_init = self._q_last if q_current is None else np.asarray(q_current, dtype=float)
        self._opti.set_initial(self._var_q, q_init)
        self._opti.set_value(self._param_q_last, q_init)
        self._opti.set_value(self._param_tf["left"], left_target)
        self._opti.set_value(self._param_tf["right"], right_target)
        try:
            with _openmp_threads(1):
                q = np.asarray(self._opti.solve().value(self._var_q), dtype=float).reshape(-1)
            converged = True
        except RuntimeError:
            # Out of iterations or infeasible: keep IPOPT's last iterate if it is usable.
            q = np.asarray(self._opti.debug.value(self._var_q), dtype=float).reshape(-1)
            converged = False
            if not np.all(np.isfinite(q)):
                q = q_init.copy()
        q = np.clip(q, self.model.lowerPositionLimit, self.model.upperPositionLimit)
        if self._filter is not None:
            q = self._filter(q)
        self._q_last = q
        pos_err = np.asarray(self._pos_err(q, left_target, right_target)).reshape(2, 3)
        return IKSolution(q, self.gravity_torques(q), converged, np.linalg.norm(pos_err, axis=1))
