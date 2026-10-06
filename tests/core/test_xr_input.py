import math
from multiprocessing import shared_memory
from types import SimpleNamespace

import numpy as np
import pytest

from xr_teleoperate.core.xr_input import WAIST_OFFSET, XRInput, recenter_wrist, valid_pose, yaw_pitch


def rot_z(a):
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def rot_y(a):
    return np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])


@pytest.mark.parametrize(("yaw", "pitch"), [(0.0, 0.0), (0.5, 0.0), (-0.4, 0.3), (1.0, -0.6)])
def test_yaw_pitch_of_a_headset_rotation(yaw, pitch):
    got = yaw_pitch(rot_z(yaw) @ rot_y(pitch))
    assert got == pytest.approx((yaw, pitch))


def test_positive_pitch_looks_down():
    forward = rot_y(0.3) @ np.array([1.0, 0.0, 0.0])
    assert forward[2] < 0 and yaw_pitch(rot_y(0.3))[1] > 0


def test_recenter_turns_the_operator_facing_direction_onto_robot_forward():
    yaw0 = 0.7
    wrist = np.eye(4)
    wrist[:3, :3] = rot_z(yaw0)
    wrist[:3, 3] = WAIST_OFFSET + rot_z(yaw0) @ np.array([0.3, 0.1, -0.2])
    out = recenter_wrist(wrist, yaw0)
    np.testing.assert_allclose(out[:3, 3], WAIST_OFFSET + [0.3, 0.1, -0.2], atol=1e-12)
    np.testing.assert_allclose(out[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(recenter_wrist(wrist, 0.0), wrist)


def test_singular_or_nan_poses_are_invalid():
    assert valid_pose(np.eye(4))
    assert not valid_pose(np.zeros((4, 4)))
    bad = np.eye(4)
    bad[0, 3] = np.nan
    assert not valid_pose(bad)


class FakeWrapper:
    def __init__(self, shm=None):
        self.frames = []
        self.closed = False
        self.tvuer = SimpleNamespace(img2display_shm=shm)

    def render_to_xr(self, image):
        self.frames.append(image)

    def close(self):
        self.closed = True


def make_xr(display_mode="ego", image_shape=(4, 6), shm=None):
    xr = XRInput.__new__(XRInput)
    xr.display_mode = display_mode
    xr.image_shape = image_shape
    xr.wrapper = FakeWrapper(shm)
    return xr


def test_render_is_a_no_op_in_pass_through():
    xr = make_xr("pass-through")
    xr.render(np.zeros((4, 6, 3), np.uint8))
    assert xr.wrapper.frames == []


def test_render_converts_rgb_to_bgr_for_televuer():
    xr = make_xr()
    image = np.zeros((4, 6, 3), np.uint8)
    image[..., 0] = 200
    xr.render(image)
    sent = xr.wrapper.frames[0]
    assert sent.dtype == np.uint8 and sent.flags["C_CONTIGUOUS"]
    assert (sent[..., 2] == 200).all() and (sent[..., 0] == 0).all()
    xr.render(image, rgb=False)
    assert (xr.wrapper.frames[1][..., 0] == 200).all()


def test_render_resizes_to_the_display_shape():
    xr = make_xr(image_shape=(4, 6))
    xr.render(np.full((8, 12, 3), 90, np.uint8))
    assert xr.wrapper.frames[0].shape == (4, 6, 3)


def test_close_releases_the_display_shared_memory():
    shm = shared_memory.SharedMemory(create=True, size=16)
    xr = make_xr(shm=shm)
    xr.close()
    assert xr.wrapper.closed
    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=shm.name)
    make_xr(shm=None).close()
