"""XR teleoperation core, importable without the DDS loop: embodiments, composed-MJCF arm IK and hand
retargeting. The XR input itself is televuer (`teleop/televuer`)."""

from .arm_ik import G1ArmIK, IKSolution, IKWeights, arm_joint_names, build_arm_model
from .embodiment import BODIES, END_EFFECTORS, HEAD_MOUNTS, HEAD_SENSORS, SIDES, Embodiment, sim_root
from .retargeting import ClosureRetargeter, HandInput, make_hand_retargeter

__all__ = [
    "BODIES",
    "END_EFFECTORS",
    "HEAD_MOUNTS",
    "HEAD_SENSORS",
    "SIDES",
    "ClosureRetargeter",
    "Embodiment",
    "G1ArmIK",
    "HandInput",
    "IKSolution",
    "IKWeights",
    "arm_joint_names",
    "build_arm_model",
    "make_hand_retargeter",
    "sim_root",
]
