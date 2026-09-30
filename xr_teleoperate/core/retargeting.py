"""Hand retargeting per end effector: XR hand input -> a closure per side (0 = open, 1 = closed).

One registry keyed by the end-effector names shared with LeRobot and the sim:

- `rubber_hand`, `none`: no fingers; always `None` (no hand keys are emitted).
- `dex1`, `dex3`, `amazing_hand`: controller trigger, or hand-tracking pinch distance, -> closure.

LeRobot turns a closure into joint targets with its `HandSpec` (`closure_to_q`), so every hand is covered
in both `closure` and `per_motor` representations. Per-joint retargeting from hand tracking (Dex3 via
dex-retargeting, AmazingHand per servo) will be further registry modes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .embodiment import END_EFFECTORS

HAND_MODES: tuple[str, ...] = ("closure",)
FINGERLESS: tuple[str, ...] = ("rubber_hand", "none")


@dataclass(frozen=True)
class HandInput:
    """One side's raw XR hand input (televuer's own units, not TeleData's Dex1-scaled ones)."""

    trigger: float = 0.0  # controller trigger, 0 released .. 1 pressed
    pinch_distance: float | None = None  # hand tracking thumb-index distance, None with controllers


def _ramp(value: float, start: float, end: float) -> float:
    return min(max((value - start) / (end - start), 0.0), 1.0)


@dataclass(frozen=True)
class ClosureRetargeter:
    """Trigger or pinch -> closure, linear between a released and a fully pressed/pinched value."""

    trigger_open: float = 0.05
    trigger_closed: float = 0.95
    # Upstream's Dex1 pinch range (5..7 in TeleData's pinchValue * 100).
    pinch_open: float = 0.07
    pinch_closed: float = 0.05

    def __call__(self, hand: HandInput) -> float:
        if hand.pinch_distance is not None:
            return _ramp(hand.pinch_distance, self.pinch_open, self.pinch_closed)
        return _ramp(hand.trigger, self.trigger_open, self.trigger_closed)


HandRetargeter = Callable[[HandInput], float | None]


def _no_fingers(hand: HandInput) -> None:
    return None


def make_hand_retargeter(end_effector: str, mode: str = "closure") -> HandRetargeter:
    """The retargeter of one end effector (the same for both sides)."""
    if end_effector not in END_EFFECTORS:
        raise ValueError(f"Unknown end effector {end_effector!r}; expected one of {END_EFFECTORS}")
    if mode not in HAND_MODES:
        raise ValueError(f"Unknown hand retargeting mode {mode!r}; expected one of {HAND_MODES}")
    if end_effector in FINGERLESS:
        return _no_fingers
    return ClosureRetargeter()
