import math

import pytest
from figures import STANDING, figure

from yoga_coach import metrics as m
from yoga_coach.checks import Check, Text


def constant(value):
    return lambda skeleton, side: value


def make_check(**kwargs):
    defaults = dict(
        key="demo",
        label=Text("演示", "Demo"),
        metric=constant(0.0),
        low=10.0,
        high=20.0,
        falloff=10.0,
        when_low=Text("太小", "too small"),
        when_high=Text("太大", "too big"),
    )
    defaults.update(kwargs)
    return Check(**defaults)


class TestScoring:
    def test_inside_the_band_scores_full_marks(self):
        result = make_check(metric=constant(15.0)).evaluate(figure(), "left")
        assert result.score == pytest.approx(1.0)
        assert result.ok
        assert result.advice() is None

    def test_score_decays_linearly_outside_the_band(self):
        result = make_check(metric=constant(25.0)).evaluate(figure(), "left")
        assert result.score == pytest.approx(0.5)

    def test_score_floors_at_zero(self):
        result = make_check(metric=constant(200.0)).evaluate(figure(), "left")
        assert result.score == pytest.approx(0.0)

    def test_unmeasurable_check_is_skipped_not_failed(self):
        result = make_check(metric=constant(None)).evaluate(figure(), "left")
        assert result.score is None
        assert not result.measured
        assert not result.ok
        assert result.advice() is None
        assert result.severity == 0.0

    def test_slightly_out_of_band_does_not_nag(self):
        # Within the pass threshold: worth a lower score, not worth a cue.
        result = make_check(metric=constant(20.5)).evaluate(figure(), "left")
        assert result.score < 1.0
        assert result.ok
        assert result.advice() is None


class TestAdvice:
    def test_picks_the_message_for_the_violated_side(self):
        low = make_check(metric=constant(0.0)).evaluate(figure(), "left")
        high = make_check(metric=constant(40.0)).evaluate(figure(), "left")
        assert low.advice().zh == "太小"
        assert high.advice().zh == "太大"

    def test_severity_is_weighted(self):
        light = make_check(metric=constant(30.0), weight=0.5).evaluate(figure(), "left")
        heavy = make_check(metric=constant(30.0), weight=2.0).evaluate(figure(), "left")
        assert heavy.severity == pytest.approx(light.severity * 4)

    def test_focus_landmarks_resolve_the_working_side(self):
        check = make_check(focus=("{s}_knee", "{o}_hip"))
        assert check.evaluate(figure(), "right").focus_landmarks() == (
            "right_knee",
            "left_hip",
        )

    def test_target_text_reads_naturally_for_one_sided_bands(self):
        both = make_check().evaluate(figure(), "left")
        only_low = make_check(high=math.inf, when_high=None).evaluate(figure(), "left")
        only_high = make_check(low=-math.inf, when_low=None).evaluate(figure(), "left")
        assert both.target_text() == "10~20°"
        assert only_low.target_text() == "≥10°"
        assert only_high.target_text() == "≤20°"


class TestValidation:
    def test_rejects_inverted_band(self):
        with pytest.raises(ValueError):
            make_check(low=30.0, high=10.0)

    def test_rejects_non_positive_falloff(self):
        with pytest.raises(ValueError):
            make_check(falloff=0.0)

    def test_rejects_a_message_that_can_never_be_shown(self):
        with pytest.raises(ValueError, match="when_low"):
            make_check(low=-math.inf)


class TestMetrics:
    def test_side_placeholders_select_the_working_side(self):
        metric = m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle")
        bent = figure(left_knee=(0.60, 0.72))
        assert metric(bent, "left") < 175.0
        assert metric(bent, "right") == pytest.approx(180.0)

    def test_distances_are_reported_in_torso_lengths(self):
        skeleton = figure()
        torso = skeleton.torso_length()
        expected = (STANDING["right_hip"][0] - STANDING["left_hip"][0]) / torso
        assert m.span("left_hip", "right_hip")(skeleton, "left") == pytest.approx(expected)

    def test_vertical_gap_is_positive_when_the_first_point_is_higher(self):
        skeleton = figure()
        assert m.vertical_gap("left_shoulder", "left_hip")(skeleton, "left") > 0
        assert m.vertical_gap("left_hip", "left_shoulder")(skeleton, "left") < 0

    def test_line_offset_signs_hips_above_and_below_the_line(self):
        metric = m.line_offset("left_shoulder", "left_hip", "left_ankle")

        def plank(hip_y):
            return figure(
                left_shoulder=(0.30, 0.50),
                left_hip=(0.55, hip_y),
                left_ankle=(0.85, 0.61),
            )

        # 0.55 puts the hip exactly on the shoulder-to-ankle line.
        assert metric(plank(0.55), "left") == pytest.approx(0.0, abs=1e-6)
        assert metric(plank(0.45), "left") > 0  # piked: hip above the line
        assert metric(plank(0.68), "left") < 0  # sagging: hip below the line

    def test_metrics_return_none_when_a_landmark_is_hidden(self):
        hidden = figure(hidden=("left_knee",))
        assert m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle")(hidden, "left") is None
        assert m.span("{s}_knee", "{s}_hip")(hidden, "left") is None

    def test_absolute_and_difference_compose(self):
        left = m.joint_angle("left_hip", "left_knee", "left_ankle")
        right = m.joint_angle("right_hip", "right_knee", "right_ankle")
        bent = figure(left_knee=(0.60, 0.72))
        gap = m.absolute(m.difference(left, right))(bent, "left")
        assert gap > 0
