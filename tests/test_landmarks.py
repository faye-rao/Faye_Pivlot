import pytest
from figures import STANDING, as_landmark_list, figure

from yoga_coach.geometry import Point
from yoga_coach.landmarks import LANDMARK_NAMES, Skeleton, mirror_name


def test_landmark_order_matches_mediapipe():
    assert len(LANDMARK_NAMES) == 33
    assert LANDMARK_NAMES[0] == "nose"
    assert LANDMARK_NAMES[11] == "left_shoulder"
    assert LANDMARK_NAMES[32] == "right_foot_index"


def test_mirror_name():
    assert mirror_name("left_knee") == "right_knee"
    assert mirror_name("right_foot_index") == "left_foot_index"
    assert mirror_name("nose") == "nose"


def test_from_list_accepts_objects_dicts_and_points():
    points = as_landmark_list(figure())
    dicts = [{"x": pt.x, "y": pt.y, "visibility": pt.visibility} for pt in points]
    objects = [
        type("LM", (), {"x": pt.x, "y": pt.y, "z": pt.z, "visibility": pt.visibility})()
        for pt in points
    ]
    for source in (points, dicts, objects):
        skeleton = Skeleton.from_list(source)
        assert skeleton.get("left_knee").x == pytest.approx(STANDING["left_knee"][0])


def test_low_visibility_landmarks_read_as_missing():
    skeleton = figure(hidden=("left_knee",))
    assert skeleton.get("left_knee") is None
    assert skeleton.require("left_hip", "left_knee") is None
    assert skeleton.require("left_hip", "left_ankle") is not None


def test_mid_landmarks_are_synthesised():
    skeleton = figure()
    mid = skeleton.get("mid_shoulder")
    assert mid.x == pytest.approx(0.5)
    assert mid.y == pytest.approx(STANDING["left_shoulder"][1])


def test_mid_landmark_is_missing_when_one_side_is():
    assert figure(hidden=("left_hip",)).get("mid_hip") is None


def test_torso_length_scales_with_the_figure():
    normal = figure().torso_length()
    # Move the shoulders twice as far from the hips: torso doubles.
    stretched = figure(
        left_shoulder=(0.44, 0.25 - 0.27), right_shoulder=(0.56, 0.25 - 0.27)
    ).torso_length()
    assert stretched == pytest.approx(normal * 2, rel=0.01)


def test_torso_length_none_without_hips():
    assert figure(hidden=("left_hip", "right_hip")).torso_length() is None


def test_coverage_counts_body_landmarks_only():
    assert figure().coverage() == pytest.approx(1.0)
    partial = figure(hidden=("left_ankle", "right_ankle", "left_heel", "right_heel"))
    assert 0.7 < partial.coverage() < 1.0
    # Face landmarks are excluded, so hiding them changes nothing.
    assert figure(hidden=("nose", "left_ear")).coverage() == pytest.approx(1.0)


def test_skeleton_ignores_extra_landmarks_beyond_the_known_names():
    short = Skeleton.from_list([Point(0.5, 0.5)] * 12)
    assert short.get("left_shoulder") is not None
    assert short.get("right_hip") is None
