"""Embodiment fields and the sim's MJCF composer, the single source of the G1 kinematics.

An embodiment is `body` x `end_effector` x `head_mount` x `head_sensor`, with the same values as LeRobot's
`--robot.*` fields and its `robot_type` (`unitree_g1-<body>-<end_effector>-<head_mount>-<head_sensor>`).
The kinematics come from the sim repo's `sim/mjcf/compose.py` (plain MJCF, no MuJoCo import), loaded from a
local checkout or the Hugging Face Hub.
"""

from __future__ import annotations

import functools
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

BODIES: tuple[str, ...] = ("29dof", "23dof")
END_EFFECTORS: tuple[str, ...] = ("rubber_hand", "none", "dex1", "dex3", "amazing_hand")
HEAD_MOUNTS: tuple[str, ...] = ("fixed", "pan_tilt")
HEAD_SENSORS: tuple[str, ...] = ("d435i", "d455")
SIDES: tuple[str, ...] = ("left", "right")

ROBOT_TYPE_PREFIX = "unitree_g1"
SIM_REPO_ID = "k-valentin/unitree-g1-mujoco"
# A local sim checkout (or snapshot) to compose from instead of the Hub.
SIM_ROOT_ENV = "XR_TELEOP_SIM_ROOT"


@dataclass(frozen=True)
class Embodiment:
    body: str = "23dof"
    end_effector: str = "amazing_hand"
    head_mount: str = "pan_tilt"
    head_sensor: str = "d455"

    def __post_init__(self) -> None:
        for field, value, allowed in (
            ("body", self.body, BODIES),
            ("end_effector", self.end_effector, END_EFFECTORS),
            ("head_mount", self.head_mount, HEAD_MOUNTS),
            ("head_sensor", self.head_sensor, HEAD_SENSORS),
        ):
            if value not in allowed:
                raise ValueError(f"Unknown {field} {value!r}; expected one of {allowed}")

    @property
    def robot_type(self) -> str:
        return "-".join((ROBOT_TYPE_PREFIX, self.body, self.end_effector, self.head_mount, self.head_sensor))

    @classmethod
    def from_robot_type(cls, robot_type: str) -> Embodiment:
        """Parse LeRobot's robot type; a 23dof body may carry its revision (`23dof_rev_1_0`)."""
        parts = robot_type.split("-")
        if len(parts) != 5 or parts[0] != ROBOT_TYPE_PREFIX:
            raise ValueError(
                f"Not a {ROBOT_TYPE_PREFIX}-<body>-<end_effector>-<head_mount>-<head_sensor> robot type: {robot_type!r}"
            )
        body, end_effector, head_mount, head_sensor = parts[1:]
        return cls(body.split("_", 1)[0], end_effector, head_mount, head_sensor)


def sim_root(root: str | Path | None = None, revision: str | None = None) -> Path:
    """The sim repo: `root`, else `$XR_TELEOP_SIM_ROOT`, else its Hub snapshot (MJCF and code only, no meshes)."""
    root = root or os.environ.get(SIM_ROOT_ENV)
    if root:
        return Path(root)
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(SIM_REPO_ID, revision=revision, allow_patterns=["sim/mjcf/*.py", "assets/*.xml"])
    )


@functools.cache
def load_compose(root: Path) -> ModuleType:
    """Import the sim's `sim/mjcf/compose.py` by path, without the `sim` package (which imports MuJoCo)."""
    path = Path(root) / "sim" / "mjcf" / "compose.py"
    spec = importlib.util.spec_from_file_location(f"_g1_sim_compose_{abs(hash(str(root)))}", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"No sim composer at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
