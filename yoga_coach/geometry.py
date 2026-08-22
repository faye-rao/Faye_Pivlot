"""Pure-geometry helpers used by the pose checks.

Everything here works on normalised image coordinates (x to the right, y
*downwards*, both roughly in ``[0, 1]``) because that is what MediaPipe Pose
returns.  The ``y`` axis pointing down is the single most common source of
sign errors, so every function that cares about it says so explicitly.

The module deliberately has no dependency on MediaPipe or OpenCV: the checks
in :mod:`yoga_coach.poses` are therefore testable with hand-written stick
figures (see ``tests/``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# The direction of "up" in image space.  y grows downwards, so up is -y.
UP = (0.0, -1.0)
RIGHT = (1.0, 0.0)


@dataclass(frozen=True)
class Point:
    """A single body landmark."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


def midpoint(a: Point, b: Point) -> Point:
    return Point(
        (a.x + b.x) / 2.0,
        (a.y + b.y) / 2.0,
        (a.z + b.z) / 2.0,
        min(a.visibility, b.visibility),
    )


def distance(a: Point, b: Point) -> float:
    return math.hypot(b.x - a.x, b.y - a.y)


def _norm(vx: float, vy: float) -> tuple[float, float] | None:
    length = math.hypot(vx, vy)
    if length < 1e-9:
        return None
    return (vx / length, vy / length)


def angle_deg(a: Point, b: Point, c: Point) -> float | None:
    """Interior angle at ``b`` of the path a -> b -> c, in ``[0, 180]``.

    A straight limb gives 180 degrees, a fully folded one gives 0.
    """
    ba = _norm(a.x - b.x, a.y - b.y)
    bc = _norm(c.x - b.x, c.y - b.y)
    if ba is None or bc is None:
        return None
    dot = max(-1.0, min(1.0, ba[0] * bc[0] + ba[1] * bc[1]))
    return math.degrees(math.acos(dot))


def _angle_between(v: tuple[float, float], ref: tuple[float, float]) -> float:
    dot = max(-1.0, min(1.0, v[0] * ref[0] + v[1] * ref[1]))
    return math.degrees(math.acos(dot))


def angle_from_vertical(a: Point, b: Point) -> float | None:
    """Angle in ``[0, 180]`` between the segment a->b and the *up* direction.

    0 means b is straight above a, 90 means the segment is horizontal, 180
    means b is straight below a.
    """
    v = _norm(b.x - a.x, b.y - a.y)
    if v is None:
        return None
    return _angle_between(v, UP)


def angle_from_horizontal(a: Point, b: Point) -> float | None:
    """Angle in ``[0, 90]`` between the segment a->b and the horizon.

    Direction-agnostic: a segment tilted 10 degrees up and one tilted 10
    degrees down both return 10.
    """
    v = _norm(b.x - a.x, b.y - a.y)
    if v is None:
        return None
    a_h = _angle_between(v, RIGHT)
    return min(a_h, 180.0 - a_h)


def signed_tilt(left: Point, right: Point) -> float | None:
    """Signed tilt of the left->right line, in degrees.

    Positive means the *right* landmark sits lower in the image than the left
    one.  Used for shoulder / hip levelling checks.
    """
    dx = right.x - left.x
    dy = right.y - left.y
    if math.hypot(dx, dy) < 1e-9:
        return None
    return math.degrees(math.atan2(dy, abs(dx) if abs(dx) > 1e-9 else 1e-9))


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class Ema:
    """Exponential moving average with a fixed smoothing factor.

    Webcam landmarks jitter by a few degrees frame to frame; feeding raw
    angles into the checks makes the on-screen advice flicker.  ``alpha``
    close to 1 follows the input quickly, close to 0 smooths harder.
    """

    def __init__(self, alpha: float = 0.35) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value: float | None = None

    def update(self, sample: float | None) -> float | None:
        if sample is None:
            return self.value
        if self.value is None:
            self.value = sample
        else:
            self.value += self.alpha * (sample - self.value)
        return self.value

    def reset(self) -> None:
        self.value = None
