"""The pose library judged against hand-built figures.

Each pose gets three kinds of test: a well-aligned figure scores near 100, a
specific mistake produces the specific cue that names it, and the pose is
distinguishable from the others so ``--pose auto`` picks it.
"""

import pytest
from figures import (
    DOWN_DOG_SIDE,
    FAR_SIDE_OCCLUDED,
    PLANK_SIDE,
    STANDING,
    TREE_LEFT,
    WARRIOR_II_RIGHT,
    figure,
)

from yoga_coach import POSES, evaluate, get_pose, rank_poses

GOOD_FIGURES = {
    "mountain": STANDING,
    "warrior2": WARRIOR_II_RIGHT,
    "plank": PLANK_SIDE,
    "tree": TREE_LEFT,
    "downdog": DOWN_DOG_SIDE,
}

#: The three side-on poses, which is where the far half of the body is hidden.
SIDE_ON = {"plank": PLANK_SIDE, "downdog": DOWN_DOG_SIDE}


def cue_keys(result):
    """Keys of the checks that produced advice, worst first."""
    return [c.check.key for c in result.corrections(limit=99)]


class TestLibraryHygiene:
    def test_pose_keys_are_unique(self):
        keys = [pose.key for pose in POSES]
        assert len(keys) == len(set(keys))

    def test_check_keys_are_unique_within_a_pose(self):
        for pose in POSES:
            keys = [check.key for check in pose.checks]
            assert len(keys) == len(set(keys)), pose.key

    def test_every_bound_has_a_message(self):
        import math

        for pose in POSES:
            for check in pose.checks:
                if not math.isinf(check.low):
                    assert check.when_low is not None, f"{pose.key}.{check.key}"
                if not math.isinf(check.high):
                    assert check.when_high is not None, f"{pose.key}.{check.key}"

    def test_every_pose_is_reachable_by_key(self):
        for pose in POSES:
            assert get_pose(pose.key) is pose

    def test_unknown_pose_key_lists_the_alternatives(self):
        with pytest.raises(KeyError, match="warrior2"):
            get_pose("headstand")


class TestWellAlignedFigures:
    @pytest.mark.parametrize("key", sorted(GOOD_FIGURES))
    def test_scores_near_perfect(self, key):
        result = evaluate(figure(GOOD_FIGURES[key]), get_pose(key))
        assert result.score > 95
        assert result.confident
        assert result.corrections() == []

    @pytest.mark.parametrize("key", sorted(GOOD_FIGURES))
    def test_auto_detection_picks_it(self, key):
        ranked = rank_poses(figure(GOOD_FIGURES[key]))
        assert ranked[0].pose.key == key
        # A clear win, not a coin toss between two poses.
        assert ranked[0].score - ranked[1].score > 10


class TestWarriorII:
    def test_detects_which_leg_is_in_front(self):
        result = evaluate(figure(WARRIOR_II_RIGHT), get_pose("warrior2"))
        assert result.side == "right"

    def test_mirrored_figure_detects_the_other_side(self):
        mirrored = {
            name: (1.0 - x, y) for name, (x, y) in WARRIOR_II_RIGHT.items()
        }
        # Mirroring the image swaps which anatomical side is which, so swap
        # the labels too -- that is exactly what the detector would report.
        swapped = {}
        for name, position in mirrored.items():
            if name.startswith("left_"):
                swapped["right_" + name[5:]] = position
            elif name.startswith("right_"):
                swapped["left_" + name[6:]] = position
            else:
                swapped[name] = position
        result = evaluate(figure(swapped), get_pose("warrior2"))
        assert result.side == "left"
        assert result.score > 95

    def test_shallow_front_knee_asks_for_a_deeper_bend(self):
        shallow = {**WARRIOR_II_RIGHT, "right_knee": (0.610, 0.680)}
        result = evaluate(figure(shallow), get_pose("warrior2"))
        assert "front_knee_bend" in cue_keys(result)
        advice = next(
            c.advice() for c in result.corrections(99) if c.check.key == "front_knee_bend"
        )
        assert "深" in advice.zh

    def test_knee_past_the_toes_is_flagged(self):
        collapsed = {**WARRIOR_II_RIGHT, "right_knee": (0.760, 0.580)}
        result = evaluate(figure(collapsed), get_pose("warrior2"))
        assert "front_knee_over_ankle" in cue_keys(result)

    def test_bent_back_leg_is_flagged(self):
        bent = {**WARRIOR_II_RIGHT, "left_knee": (0.300, 0.760)}
        result = evaluate(figure(bent), get_pose("warrior2"))
        assert "back_leg_straight" in cue_keys(result)

    def test_drooping_arms_are_flagged(self):
        drooping = {
            **WARRIOR_II_RIGHT,
            "right_wrist": (0.750, 0.620),
            "right_elbow": (0.615, 0.500),
        }
        result = evaluate(figure(drooping), get_pose("warrior2"))
        assert "arms_level" in cue_keys(result)

    def test_only_three_cues_are_offered_at_once(self):
        wreck = {
            **WARRIOR_II_RIGHT,
            "right_knee": (0.780, 0.640),
            "left_knee": (0.300, 0.770),
            "right_wrist": (0.700, 0.560),
            "left_wrist": (0.220, 0.300),
        }
        result = evaluate(figure(wreck), get_pose("warrior2"))
        assert len(result.corrections()) == 3
        assert len(result.corrections(limit=99)) > 3

    def test_cues_are_ordered_by_severity(self):
        result = evaluate(
            figure({**WARRIOR_II_RIGHT, "right_knee": (0.800, 0.660), "left_wrist": (0.150, 0.420)}),
            get_pose("warrior2"),
        )
        severities = [c.severity for c in result.corrections(limit=99)]
        assert severities == sorted(severities, reverse=True)


class TestTree:
    def test_foot_resting_on_the_standing_knee_is_flagged(self):
        risky = {**TREE_LEFT, "right_ankle": (0.480, 0.720), "right_knee": (0.640, 0.760)}
        result = evaluate(figure(risky), get_pose("tree"))
        assert "foot_off_knee" in cue_keys(result)

    def test_hip_hike_is_flagged(self):
        hiked = {**TREE_LEFT, "right_hip": (0.540, 0.470)}
        result = evaluate(figure(hiked), get_pose("tree"))
        assert "hips_level" in cue_keys(result)

    def test_standing_figure_is_told_to_lift_the_foot(self):
        result = evaluate(figure(), get_pose("tree"))
        assert "foot_lifted" in cue_keys(result)


class TestPlank:
    def test_sagging_hips_and_piked_hips_get_opposite_advice(self):
        sagging = evaluate(
            figure({**PLANK_SIDE, "left_hip": (0.550, 0.640), "right_hip": (0.553, 0.644)}),
            get_pose("plank"),
        )
        piked = evaluate(
            figure({**PLANK_SIDE, "left_hip": (0.550, 0.430), "right_hip": (0.553, 0.434)}),
            get_pose("plank"),
        )
        assert "body_line" in cue_keys(sagging)
        assert "body_line" in cue_keys(piked)
        sag_advice = next(c.advice() for c in sagging.corrections(99) if c.check.key == "body_line")
        pike_advice = next(c.advice() for c in piked.corrections(99) if c.check.key == "body_line")
        assert sag_advice.zh != pike_advice.zh
        assert "下沉" in sag_advice.zh
        assert "太高" in pike_advice.zh

    def test_hands_too_far_forward_is_flagged(self):
        creeping = {
            **PLANK_SIDE,
            "left_wrist": (0.180, 0.700),
            "right_wrist": (0.183, 0.704),
            "left_elbow": (0.240, 0.600),
            "right_elbow": (0.243, 0.604),
        }
        result = evaluate(figure(creeping), get_pose("plank"))
        assert "shoulder_over_wrist" in cue_keys(result)

    def test_a_standing_body_is_not_a_perfect_plank(self):
        # Standing is geometrically a straight shoulder-hip-ankle line; only
        # the orientation check tells the two apart.
        result = evaluate(figure(), get_pose("plank"))
        assert "body_horizontal" in cue_keys(result)


class TestConfidence:
    def test_a_body_half_out_of_frame_is_not_confident(self):
        cropped = figure(
            hidden=(
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_heel",
                "right_heel",
                "left_foot_index",
                "right_foot_index",
            )
        )
        result = evaluate(cropped, get_pose("mountain"))
        assert not result.confident

    def test_auto_detection_prefers_a_pose_it_can_actually_measure(self):
        # Only the upper body is visible.  Tree keeps a couple of torso checks
        # that a standing figure passes outright; without the measured-share
        # discount that near-empty result would outrank Mountain.
        upper_body_only = figure(
            hidden=(
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
                "left_heel",
                "right_heel",
                "left_foot_index",
                "right_foot_index",
            )
        )
        ranked = rank_poses(upper_body_only)
        assert ranked[0].pose.key == "mountain"
        assert ranked[0].measured_share > ranked[-1].measured_share

    def test_hidden_landmarks_do_not_drag_the_score_down(self):
        full = evaluate(figure(), get_pose("mountain"))
        partial = evaluate(figure(hidden=("left_knee",)), get_pose("mountain"))
        assert partial.score == pytest.approx(full.score)
        assert any(not r.measured for r in partial.results)


class TestSideOnCamera:
    """A pose shot from its own recommended angle must still be scorable.

    Reported from practice: Downward Dog said "body not fully in frame" while
    the whole body plainly was.  Seen from the side, MediaPipe marks the far
    arm and leg as low-visibility -- they are behind the near ones -- and
    three separate places treated that occlusion as missing data.
    """

    @pytest.mark.parametrize("key", sorted(SIDE_ON))
    def test_the_far_side_being_hidden_is_not_bad_framing(self, key):
        skeleton = figure(SIDE_ON[key], hidden=FAR_SIDE_OCCLUDED)
        assert skeleton.coverage() == pytest.approx(1.0)
        assert skeleton.missing_parts() == []

    @pytest.mark.parametrize("key", sorted(SIDE_ON))
    def test_still_confident_with_the_far_side_hidden(self, key):
        result = evaluate(figure(SIDE_ON[key], hidden=FAR_SIDE_OCCLUDED), get_pose(key))
        assert result.confident, f"{key} 在侧面机位下应该能评分"
        assert result.score > 95

    @pytest.mark.parametrize("key", sorted(SIDE_ON))
    def test_the_visible_side_is_the_one_evaluated(self, key):
        """Picking the side on raw score alone hands the frame to the hidden
        half: almost nothing is measurable there, and the two checks that
        survive score 100."""
        result = evaluate(figure(SIDE_ON[key], hidden=FAR_SIDE_OCCLUDED), get_pose(key))
        assert result.side == "right"  # left_* is the occluded set
        assert result.measured_share > 0.6

    @pytest.mark.parametrize("key", sorted(SIDE_ON))
    def test_auto_detection_still_names_it(self, key):
        ranked = rank_poses(figure(SIDE_ON[key], hidden=FAR_SIDE_OCCLUDED))
        assert ranked[0].pose.key == key

    def test_torso_length_survives_one_hidden_side(self):
        """Every distance is measured in torso lengths, and the torso needs a
        shoulder and a hip -- previously both sides of each."""
        skeleton = figure(DOWN_DOG_SIDE, hidden=FAR_SIDE_OCCLUDED)
        assert skeleton.torso_length() is not None
        assert skeleton.get("mid_shoulder") is not None
        assert skeleton.get("mid_hip") is not None

    def test_a_genuinely_cropped_body_is_still_caught(self):
        both_legs_gone = (
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_heel",
            "right_heel",
            "left_foot_index",
            "right_foot_index",
        )
        skeleton = figure(DOWN_DOG_SIDE, hidden=both_legs_gone)
        assert skeleton.coverage() < 0.6
        assert set(skeleton.missing_parts()) == {"knee", "ankle", "heel", "foot_index"}
        assert not evaluate(skeleton, get_pose("downdog")).confident

    def test_levelling_checks_still_need_both_sides(self):
        """mid_* falls back to one side, but a check comparing left against
        right must not silently compare a landmark with itself."""
        from yoga_coach import metrics as m

        skeleton = figure(STANDING, hidden=("left_hip",))
        assert m.tilt("left_hip", "right_hip")(skeleton, "left") is None
