import pytest

from xr_teleoperate.core import END_EFFECTORS, Embodiment, HandInput, make_hand_retargeter


def test_robot_type_round_trip():
    embodiment = Embodiment("23dof", "amazing_hand", "pan_tilt", "d455")
    assert embodiment.robot_type == "unitree_g1-23dof-amazing_hand-pan_tilt-d455"
    assert Embodiment.from_robot_type(embodiment.robot_type) == embodiment


@pytest.mark.parametrize(
    "robot_type", ["unitree_g1-23dof-amazing_hand-pan_tilt", "so100-23dof-dex3-fixed-d435i", "unitree_g1-23dof-dex9-fixed-d435i"]
)
def test_bad_robot_types_are_refused(robot_type):
    with pytest.raises(ValueError):
        Embodiment.from_robot_type(robot_type)


@pytest.mark.parametrize("end_effector", ["rubber_hand", "none"])
def test_fingerless_hands_emit_nothing(end_effector):
    assert make_hand_retargeter(end_effector)(HandInput(trigger=1.0)) is None


@pytest.mark.parametrize("end_effector", [ee for ee in END_EFFECTORS if ee not in ("rubber_hand", "none")])
def test_trigger_and_pinch_map_to_closure(end_effector):
    retarget = make_hand_retargeter(end_effector)
    assert retarget(HandInput(trigger=0.0)) == 0.0
    assert retarget(HandInput(trigger=1.0)) == 1.0
    assert 0.4 < retarget(HandInput(trigger=0.5)) < 0.6
    assert retarget(HandInput(pinch_distance=0.10)) == 0.0
    assert retarget(HandInput(pinch_distance=0.01)) == 1.0
    assert retarget(HandInput(trigger=1.0, pinch_distance=0.10)) == 0.0


def test_unknown_hand_or_mode_is_refused():
    with pytest.raises(ValueError):
        make_hand_retargeter("dex9")
    with pytest.raises(ValueError):
        make_hand_retargeter("dex3", mode="per_motor")
