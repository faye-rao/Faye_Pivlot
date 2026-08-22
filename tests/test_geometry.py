import math

import pytest

from yoga_coach.geometry import (
    Ema,
    Point,
    angle_deg,
    angle_from_horizontal,
    angle_from_vertical,
    distance,
    midpoint,
    signed_tilt,
)


def p(x, y, visibility=1.0):
    return Point(x, y, 0.0, visibility)


def test_angle_deg_right_angle():
    assert angle_deg(p(0, 1), p(0, 0), p(1, 0)) == pytest.approx(90.0)


def test_angle_deg_straight_limb():
    assert angle_deg(p(0, 0), p(0, 0.5), p(0, 1)) == pytest.approx(180.0)


def test_angle_deg_folded_limb():
    assert angle_deg(p(0, 0), p(0, 1), p(0, 0.1)) == pytest.approx(0.0, abs=1e-6)


def test_angle_deg_none_on_coincident_points():
    assert angle_deg(p(0, 0), p(0, 0), p(1, 1)) is None


def test_angle_from_vertical_uses_screen_up():
    # y grows downwards, so a point with the smaller y is the higher one.
    assert angle_from_vertical(p(0.5, 0.9), p(0.5, 0.1)) == pytest.approx(0.0)
    assert angle_from_vertical(p(0.5, 0.1), p(0.5, 0.9)) == pytest.approx(180.0)
    assert angle_from_vertical(p(0.1, 0.5), p(0.9, 0.5)) == pytest.approx(90.0)


def test_angle_from_horizontal_is_direction_agnostic():
    up = angle_from_horizontal(p(0.0, 0.5), p(1.0, 0.5 - math.tan(math.radians(10))))
    down = angle_from_horizontal(p(0.0, 0.5), p(1.0, 0.5 + math.tan(math.radians(10))))
    assert up == pytest.approx(10.0)
    assert down == pytest.approx(10.0)


def test_signed_tilt_positive_when_right_side_is_lower():
    assert signed_tilt(p(0.4, 0.5), p(0.6, 0.6)) > 0
    assert signed_tilt(p(0.4, 0.6), p(0.6, 0.5)) < 0
    assert signed_tilt(p(0.4, 0.5), p(0.6, 0.5)) == pytest.approx(0.0)


def test_midpoint_and_distance():
    mid = midpoint(p(0.0, 0.0), p(1.0, 1.0))
    assert (mid.x, mid.y) == pytest.approx((0.5, 0.5))
    assert distance(p(0.0, 0.0), p(3.0, 4.0)) == pytest.approx(5.0)


def test_midpoint_takes_the_lower_visibility():
    assert midpoint(p(0, 0, 0.9), p(1, 1, 0.2)).visibility == pytest.approx(0.2)


class TestEma:
    def test_first_sample_passes_through(self):
        ema = Ema(alpha=0.5)
        assert ema.update(10.0) == pytest.approx(10.0)

    def test_converges_towards_the_input(self):
        ema = Ema(alpha=0.5)
        ema.update(0.0)
        assert ema.update(10.0) == pytest.approx(5.0)
        assert ema.update(10.0) == pytest.approx(7.5)

    def test_missing_sample_holds_the_last_value(self):
        ema = Ema(alpha=0.5)
        ema.update(4.0)
        assert ema.update(None) == pytest.approx(4.0)

    def test_reset_forgets_history(self):
        ema = Ema(alpha=0.5)
        ema.update(4.0)
        ema.reset()
        assert ema.update(9.0) == pytest.approx(9.0)

    def test_rejects_invalid_alpha(self):
        with pytest.raises(ValueError):
            Ema(alpha=0.0)
