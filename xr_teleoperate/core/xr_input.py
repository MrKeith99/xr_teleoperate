"""XR input from televuer as one `XRFrame` per read, in the robot convention (x forward, y left, z up).

Wrist targets are televuer's `head_position` poses (world orientation, origin moved from the headset to
near the pelvis), turned by the headset yaw captured at `recenter()`: the operator's facing direction then
is the robot's +x, and turning the head afterwards moves neither the arm targets nor, with a pan/tilt head,
anything but the head. Head yaw is relative to that direction; pitch is absolute, positive looking down.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .embodiment import SIDES
from .retargeting import HandInput

BUTTONS: tuple[str, ...] = ("a", "b", "x", "y", "left_stick", "right_stick", "left_grip", "right_grip")
# televuer's origin shift from the headset to near the pelvis (tv_wrapper, "head_position" mode).
WAIST_OFFSET = np.array([0.15, 0.0, 0.45])
_GRIP_PRESSED = 0.5


@dataclass(frozen=True)
class XRFrame:
    tracking: bool = False  # motion data received and head/wrist poses valid
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    wrists: dict[str, np.ndarray] = field(default_factory=lambda: {side: np.eye(4) for side in SIDES})
    hands: dict[str, HandInput] = field(default_factory=lambda: {side: HandInput() for side in SIDES})
    sticks: dict[str, tuple[float, float]] = field(default_factory=lambda: dict.fromkeys(SIDES, (0.0, 0.0)))
    buttons: dict[str, bool] = field(default_factory=lambda: dict.fromkeys(BUTTONS, False))


def valid_pose(pose) -> bool:
    pose = np.asarray(pose, dtype=float).reshape(4, 4)
    det = np.linalg.det(pose[:3, :3])
    return bool(np.all(np.isfinite(pose)) and np.isfinite(det) and abs(det) > 1e-6)


def yaw_pitch(rotation: np.ndarray) -> tuple[float, float]:
    """Yaw about +z and pitch about +y (positive tips +x down) of a rotation in the robot convention."""
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    pitch = math.atan2(-rotation[2, 0], math.hypot(rotation[0, 0], rotation[1, 0]))
    return yaw, pitch


def _rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def recenter_wrist(wrist: np.ndarray, yaw0: float) -> np.ndarray:
    """Turn a `head_position` wrist pose by -yaw0 about the headset's vertical axis."""
    turn = _rot_z(-yaw0)
    out = np.array(wrist, dtype=float, copy=True)
    out[:3, :3] = turn @ out[:3, :3]
    out[:3, 3] = turn @ (out[:3, 3] - WAIST_OFFSET) + WAIST_OFFSET
    return out


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class XRInput:
    """televuer (Vuer WebXR on https://<host>:8012) read as `XRFrame`s; the Vuer server runs in its own process."""

    def __init__(
        self,
        input_mode: str = "controller",
        display_mode: str = "pass-through",
        image_shape: tuple[int, int] = (480, 640),
        binocular: bool = False,
        display_fps: float = 30.0,
        cert_file: str | None = None,
        key_file: str | None = None,
    ):
        if input_mode not in ("controller", "hand"):
            raise ValueError(f"Unknown input_mode {input_mode!r}; expected 'controller' or 'hand'")
        from televuer import TeleVuerWrapper

        self.use_hands = input_mode == "hand"
        self.display_mode = display_mode
        self.image_shape = tuple(image_shape)
        self.wrapper = TeleVuerWrapper(
            use_hand_tracking=self.use_hands,
            binocular=binocular,
            img_shape=image_shape,
            display_fps=display_fps,
            display_mode=display_mode,
            zmq=display_mode != "pass-through",
            cert_file=cert_file,
            key_file=key_file,
            arm_reference_mode="head_position",
        )
        self.yaw0 = 0.0

    def recenter(self) -> None:
        """Take the current headset yaw as the robot's forward direction."""
        tele = self.wrapper.get_tele_data()
        self.yaw0 = yaw_pitch(tele.head_pose[:3, :3])[0]

    def render(self, image: np.ndarray, *, rgb: bool = True) -> None:
        """Show an HxWx3 uint8 frame in the headset; no-op in pass-through (televuer warns per call there)."""
        if self.display_mode == "pass-through":
            return
        import cv2

        height, width = self.image_shape[:2]
        if image.shape[:2] != (height, width):
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        if rgb:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        self.wrapper.render_to_xr(np.ascontiguousarray(image, dtype=np.uint8))

    def close(self) -> None:
        self.wrapper.close()
        shm = getattr(getattr(self.wrapper, "tvuer", None), "img2display_shm", None)
        if shm is not None:
            for release in (shm.close, shm.unlink):
                try:
                    release()
                except OSError:
                    pass

    def read(self) -> XRFrame:
        tv = self.wrapper.tvuer
        tele = self.wrapper.get_tele_data()
        tracking = bool(
            tele.motion_data_ready
            and valid_pose(tv.head_pose)
            and valid_pose(tv.left_arm_pose)
            and valid_pose(tv.right_arm_pose)
        )
        yaw, pitch = yaw_pitch(tele.head_pose[:3, :3])
        wrists = {
            "left": recenter_wrist(tele.left_wrist_pose, self.yaw0),
            "right": recenter_wrist(tele.right_wrist_pose, self.yaw0),
        }
        if self.use_hands:
            hands = {
                "left": HandInput(pinch_distance=float(tv.left_hand_pinchValue)),
                "right": HandInput(pinch_distance=float(tv.right_hand_pinchValue)),
            }
            return XRFrame(tracking, _wrap(yaw - self.yaw0), pitch, wrists, hands)
        hands = {
            "left": HandInput(trigger=float(tv.left_ctrl_triggerValue)),
            "right": HandInput(trigger=float(tv.right_ctrl_triggerValue)),
        }
        sticks = {
            "left": tuple(float(v) for v in tv.left_ctrl_thumbstickValue[:2]),
            "right": tuple(float(v) for v in tv.right_ctrl_thumbstickValue[:2]),
        }
        # Vuer's aButton/bButton are A/B on the right controller and X/Y on the left one.
        buttons = {
            "a": bool(tv.right_ctrl_aButton),
            "b": bool(tv.right_ctrl_bButton),
            "x": bool(tv.left_ctrl_aButton),
            "y": bool(tv.left_ctrl_bButton),
            "left_stick": bool(tv.left_ctrl_thumbstick),
            "right_stick": bool(tv.right_ctrl_thumbstick),
            "left_grip": float(tv.left_ctrl_squeezeValue) > _GRIP_PRESSED,
            "right_grip": float(tv.right_ctrl_squeezeValue) > _GRIP_PRESSED,
        }
        return XRFrame(tracking, _wrap(yaw - self.yaw0), pitch, wrists, hands, sticks, buttons)
