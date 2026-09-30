import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_SIBLING_SIM = REPO.parent / "unitree-g1-mujoco"

# Compose from the sibling sim checkout when there is one (MJCF and code suffice, no meshes).
if "XR_TELEOP_SIM_ROOT" not in os.environ and (_SIBLING_SIM / "sim" / "mjcf" / "compose.py").is_file():
    os.environ["XR_TELEOP_SIM_ROOT"] = str(_SIBLING_SIM)
