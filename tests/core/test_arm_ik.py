import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from xr_teleoperate.core import BODIES, END_EFFECTORS, G1ArmIK, arm_joint_names, build_arm_model
from xr_teleoperate.core.arm_ik import WeightedMovingFilter

UPSTREAM_ASSETS = Path(__file__).resolve().parents[2] / "assets" / "g1"
# Upstream's URDF and its IK end effector: G1_29_ArmIK's L_ee/R_ee (wrist_yaw_joint +0.05 m); the
# 23dof has no wrist pitch/yaw, so our tool point sits 0.084 m further out along wrist_roll.
UPSTREAM = {
    "29dof": ("g1_body29_hand14.urdf", "wrist_yaw_joint", 0.05),
    "23dof": ("g1_body23.urdf", "wrist_roll_joint", 0.084 + 0.05),
}
ALL_EMBODIMENTS = [(body, ee) for body in BODIES for ee in END_EFFECTORS]


def upstream_tool_poses(body: str, q_by_name: dict[str, float]) -> dict[str, np.ndarray]:
    urdf, wrist_joint, offset = UPSTREAM[body]
    model = pin.buildModelFromUrdf(str(UPSTREAM_ASSETS / urdf))
    keep = set(q_by_name)
    lock = [model.getJointId(n) for n in model.names[1:] if n not in keep]
    model = pin.buildReducedModel(model, lock, pin.neutral(model))
    frames = {
        side: model.addFrame(
            pin.Frame(
                f"{side}_ee",
                model.getJointId(f"{side}_{wrist_joint}"),
                pin.SE3(np.eye(3), np.array([offset, 0.0, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        for side in ("left", "right")
    }
    q = np.zeros(model.nq)
    for name, value in q_by_name.items():
        q[model.joints[model.getJointId(name)].idx_q] = value
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    return {side: data.oMf[fid].homogeneous for side, fid in frames.items()}


@pytest.fixture(scope="module", params=BODIES)
def dex3_ik(request):
    return G1ArmIK(request.param, "dex3", filter_weights=None)


@pytest.mark.parametrize(("body", "end_effector"), ALL_EMBODIMENTS)
def test_model_has_the_body_arm_joints_and_tool_frames(body, end_effector):
    model, frame_ids = build_arm_model(body, end_effector)
    assert tuple(model.names[1:]) == arm_joint_names(body)
    assert model.nq == {"29dof": 14, "23dof": 10}[body]
    assert np.all(np.isfinite(model.lowerPositionLimit)) and np.all(np.isfinite(model.upperPositionLimit))
    assert set(frame_ids) == {"left", "right"}


def test_tool_point_is_the_same_on_every_embodiment_at_zero():
    """Every hand's TCP placeholder and the bare wrist sit at the same point, on either body."""
    poses = [G1ArmIK(body, ee, filter_weights=None).forward(np.zeros(14 if body == "29dof" else 10))
             for body, ee in ALL_EMBODIMENTS]
    for pose in poses[1:]:
        for side in ("left", "right"):
            np.testing.assert_allclose(pose[side], poses[0][side], atol=1e-6)


def test_tool_poses_match_upstream_urdf_ik(dex3_ik):
    rng = np.random.default_rng(0)
    lo, hi = dex3_ik.model.lowerPositionLimit, dex3_ik.model.upperPositionLimit
    for _ in range(20):
        q = rng.uniform(lo, hi)
        ours = dex3_ik.forward(q)
        theirs = upstream_tool_poses(dex3_ik.body, dict(zip(dex3_ik.joint_names, q, strict=True)))
        for side in ("left", "right"):
            np.testing.assert_allclose(ours[side][:3, 3], theirs[side][:3, 3], atol=2e-3)
            np.testing.assert_allclose(ours[side][:3, :3], theirs[side][:3, :3], atol=1e-3)


@pytest.mark.parametrize(("body", "end_effector"), ALL_EMBODIMENTS)
def test_ik_reaches_reachable_targets(body, end_effector):
    ik = G1ArmIK(body, end_effector, filter_weights=None)
    rng = np.random.default_rng(1)
    lo, hi = ik.model.lowerPositionLimit, ik.model.upperPositionLimit
    errors = []
    for _ in range(10):
        q_target = rng.uniform(0.5 * lo, 0.5 * hi)
        target = ik.forward(q_target)
        ik.reset(np.clip(q_target + rng.normal(0.0, 0.2, ik.nq), lo, hi))
        solution = ik.solve(target["left"], target["right"])
        assert solution.q.shape == (ik.nq,) and solution.tau_ff.shape == (ik.nq,)
        assert np.all(solution.q >= lo - 1e-9) and np.all(solution.q <= hi + 1e-9)
        reached = ik.forward(solution.q)
        errors += [np.linalg.norm(reached[s][:3, 3] - target[s][:3, 3]) for s in ("left", "right")]
    assert np.median(errors) < 0.005
    assert np.max(errors) < 0.03


def test_ik_matches_upstream_targets_on_29dof():
    """Poses from upstream's URDF IK frames are reached by the composed-MJCF IK (29dof + Dex3)."""
    ik = G1ArmIK("29dof", "dex3", filter_weights=None)
    rng = np.random.default_rng(2)
    lo, hi = ik.model.lowerPositionLimit, ik.model.upperPositionLimit
    for _ in range(10):
        q = rng.uniform(0.5 * lo, 0.5 * hi)
        target = upstream_tool_poses("29dof", dict(zip(ik.joint_names, q, strict=True)))
        ik.reset(q)
        solution = ik.solve(target["left"], target["right"])
        assert solution.position_error.max() < 0.01


def test_solve_is_fast_enough_for_teleop(dex3_ik):
    target = dex3_ik.forward(np.zeros(dex3_ik.nq))
    dex3_ik.solve(target["left"], target["right"])
    times = []
    for _ in range(20):
        start = time.perf_counter()
        dex3_ik.solve(target["left"], target["right"])
        times.append(time.perf_counter() - start)
    assert np.median(times) < 0.02


def test_moving_filter_weights_newest_first():
    f = WeightedMovingFilter((0.5, 0.3, 0.2), 1)
    assert f(np.array([1.0]))[0] == 1.0
    f(np.array([2.0]))
    np.testing.assert_allclose(f(np.array([3.0])), [0.5 * 3 + 0.3 * 2 + 0.2 * 1])
    np.testing.assert_allclose(f(np.array([4.0])), [0.5 * 4 + 0.3 * 3 + 0.2 * 2])
    with pytest.raises(ValueError):
        WeightedMovingFilter((0.5, 0.4), 1)
