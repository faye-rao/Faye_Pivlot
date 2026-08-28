"""Overlay smoke tests.

These only assert that drawing succeeds and actually marks the frame -- what
the panel looks like is a judgement call, not something to pin down in a unit
test.  They are skipped when OpenCV or Pillow is missing so the rule tests can
still run in a bare environment.
"""

import numpy as np
import pytest
from figures import WARRIOR_II_RIGHT, figure

pytest.importorskip("cv2")
pytest.importorskip("PIL")

from yoga_coach import get_pose  # noqa: E402
from yoga_coach.render import Overlay, score_colour  # noqa: E402
from yoga_coach.session import CoachSession  # noqa: E402


@pytest.fixture
def frame():
    return np.full((480, 854, 3), 40, dtype=np.uint8)


@pytest.fixture
def overlay():
    # English avoids depending on a CJK font being installed on the runner.
    return Overlay(lang="en")


def state_for(skeleton, pose_key="warrior2"):
    session = CoachSession(pose=get_pose(pose_key))
    session.update(skeleton, 0.0)
    return session.update(skeleton, 0.1)


def test_draws_something_on_the_frame(overlay, frame):
    skeleton = figure(WARRIOR_II_RIGHT)
    out = overlay.draw(frame.copy(), state_for(skeleton), skeleton)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, frame)


def test_handles_a_frame_with_no_body(overlay, frame):
    session = CoachSession(pose=get_pose("mountain"))
    out = overlay.draw(frame.copy(), session.update(None, 0.0), None)
    assert out.shape == frame.shape


def test_detail_mode_adds_more_ink(overlay, frame):
    bad = figure({**WARRIOR_II_RIGHT, "right_knee": (0.800, 0.660)})
    state = state_for(bad)
    plain = overlay.draw(frame.copy(), state, bad, show_details=False)
    detailed = overlay.draw(frame.copy(), state, bad, show_details=True)
    assert int(detailed.sum()) != int(plain.sum())


def test_no_score_is_shown_when_the_body_is_half_out_of_frame(overlay, frame):
    cropped = figure(
        WARRIOR_II_RIGHT,
        hidden=(
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        ),
    )
    session = CoachSession(pose=get_pose("warrior2"))
    state = session.update(cropped, 0.0)
    assert not state.result.confident
    # Drawing must succeed and must not fall back to the confident layout.
    out = overlay.draw(frame.copy(), state, cropped)
    full = overlay.draw(frame.copy(), state_for(figure(WARRIOR_II_RIGHT)), figure(WARRIOR_II_RIGHT))
    assert not np.array_equal(out, full)


def test_score_colour_bands():
    assert score_colour(95) != score_colour(75)
    assert score_colour(75) != score_colour(40)


def test_wrapping_breaks_chinese_between_characters(overlay):
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    font = overlay.font(20)
    lines = overlay._wrap(draw, "前膝移回脚踝正上方别超过脚尖", font, 80)
    assert len(lines) > 1
    assert "".join(lines) == "前膝移回脚踝正上方别超过脚尖"


def test_wrapping_breaks_english_between_words(overlay):
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines = overlay._wrap(
        draw, "Stack the front knee over the ankle", overlay.font(20), 120
    )
    assert len(lines) > 1
    assert all(not line.startswith(" ") for line in lines)
    assert " ".join(lines).split() == "Stack the front knee over the ankle".split()
