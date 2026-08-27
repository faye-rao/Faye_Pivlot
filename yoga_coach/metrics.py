"""Measurements a pose check can make on a skeleton.

Each factory returns a callable ``(skeleton, side) -> float | None``.  ``side``
is ``"left"`` or ``"right"`` and names the *working* side of an asymmetric
pose (the front leg in Warrior II, the standing leg in Tree, ...).  Landmark
names may contain two placeholders:

``{s}``
    the working side
``{o}``
    the other side

so ``"{s}_knee"`` resolves to ``left_knee`` when evaluating the left-side
variant.  :func:`yoga_coach.evaluator.evaluate` tries both variants and keeps
whichever scores higher, which is how the coach works out which leg is in
front without asking.

Angles are in degrees.  Distances are expressed as multiples of the torso
length so they do not change when the practitioner moves towards the camera.
"""

from __future__ import annotations

from typing import Callable

from .geometry import (
    Point,
    angle_deg,
    angle_from_horizontal,
    angle_from_vertical,
    distance,
    signed_tilt,
)
from .landmarks import Skeleton

Metric = Callable[[Skeleton, str], float | None]

_OTHER = {"left": "right", "right": "left"}


def resolve(name: str, side: str) -> str:
    """Substitute the ``{s}`` / ``{o}`` placeholders in a landmark name."""
    return name.format(s=side, o=_OTHER[side])


def _fetch(skeleton: Skeleton, side: str, names: tuple[str, ...]) -> list[Point] | None:
    return skeleton.require(*(resolve(n, side) for n in names))


def joint_angle(a: str, b: str, c: str) -> Metric:
    """Interior angle at joint ``b``.  180 is a straight limb, 0 fully folded."""

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (a, b, c))
        if pts is None:
            return None
        return angle_deg(*pts)

    return metric


def from_vertical(a: str, b: str) -> Metric:
    """Angle of the segment a->b away from straight-up.

    0 means ``b`` is directly above ``a``; 90 means the segment is level.
    """

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (a, b))
        if pts is None:
            return None
        return angle_from_vertical(pts[0], pts[1])

    return metric


def from_horizontal(a: str, b: str) -> Metric:
    """How far the segment a->b departs from level, in ``[0, 90]``.

    Sign-free: tilted up and tilted down by the same amount read the same.
    Use :func:`tilt` when the direction matters.
    """

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (a, b))
        if pts is None:
            return None
        return angle_from_horizontal(pts[0], pts[1])

    return metric


def tilt(left: str, right: str) -> Metric:
    """Signed tilt of the left->right line; positive means the right end is lower."""

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (left, right))
        if pts is None:
            return None
        return signed_tilt(pts[0], pts[1])

    return metric


def _scaled(skeleton: Skeleton, value: float) -> float | None:
    torso = skeleton.torso_length()
    if torso is None:
        return None
    return value / torso


def horizontal_gap(a: str, b: str) -> Metric:
    """``|a.x - b.x|`` in torso lengths -- e.g. knee stacked over ankle."""

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (a, b))
        if pts is None:
            return None
        return _scaled(skeleton, abs(pts[0].x - pts[1].x))

    return metric


def vertical_gap(a: str, b: str) -> Metric:
    """Height of ``a`` above ``b`` in torso lengths (negative when below).

    Remember y grows downwards, so "above" is ``b.y - a.y``.
    """

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (a, b))
        if pts is None:
            return None
        return _scaled(skeleton, pts[1].y - pts[0].y)

    return metric


def span(a: str, b: str) -> Metric:
    """Straight-line distance between two landmarks, in torso lengths."""

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (a, b))
        if pts is None:
            return None
        return _scaled(skeleton, distance(pts[0], pts[1]))

    return metric


def line_offset(a: str, b: str, c: str) -> Metric:
    """How far ``b`` sits off the straight line a->c, in torso lengths.

    Positive means ``b`` is *above* the line (towards the top of the image),
    negative below it.  This is what separates "hips too high" from "hips
    sagging" in Plank, where a single joint angle cannot tell them apart.
    """

    def metric(skeleton: Skeleton, side: str) -> float | None:
        pts = _fetch(skeleton, side, (a, b, c))
        if pts is None:
            return None
        start, point, end = pts
        dx = end.x - start.x
        dy = end.y - start.y
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-9:
            return None

        # Project onto whichever unit normal points *up the image*, rather
        # than onto "left of the direction of travel".  Turning round in front
        # of the camera reverses the segment, and a sign tied to its direction
        # reverses with it -- which had Plank telling someone facing the other
        # way to lift their hips when they needed to lower them.
        nx, ny = -dy / length, dx / length
        if ny > 0:  # y grows downwards, so flip to make the normal point up
            nx, ny = -nx, -ny
        if abs(ny) < 1e-9:
            # A vertical reference line: "above" it is not a thing.
            return None
        return _scaled(skeleton, (point.x - start.x) * nx + (point.y - start.y) * ny)

    return metric


def difference(first: Metric, second: Metric) -> Metric:
    """``first - second``.  Handy for left/right symmetry checks."""

    def metric(skeleton: Skeleton, side: str) -> float | None:
        a = first(skeleton, side)
        b = second(skeleton, side)
        if a is None or b is None:
            return None
        return a - b

    return metric


def absolute(inner: Metric) -> Metric:
    """Magnitude of another metric."""

    def metric(skeleton: Skeleton, side: str) -> float | None:
        value = inner(skeleton, side)
        return None if value is None else abs(value)

    return metric
