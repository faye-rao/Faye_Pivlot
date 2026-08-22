"""The pose library judged against hand-built figures.

Each pose gets three kinds of test: a well-aligned figure scores near 100, a
specific mistake produces the specific cue that names it, and the pose is
distinguishable from the others so ``--pose auto`` picks it.
"""

import pytest
from figures import (
    DOWN_DOG_SIDE,
    PLANK_SIDE,
    STANDING,
    TREE_LEFT,
    UP_DOG_SIDE,
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
    "updog": UP_DOG_SIDE,
}


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


class TestDownDog:
    def test_low_hips_are_told_to_push_up_and_back(self):
        # Hips dropped from 0.420 to 0.600 with the knee moved onto the new
        # hip -> ankle midpoint, so the legs stay straight and the fold at the
        # hip opens from 69 to 121 degrees -- past the 100 degree band.
        low = {
            **DOWN_DOG_SIDE,
            "left_hip": (0.560, 0.600),
            "right_hip": (0.563, 0.604),
            "left_knee": (0.680, 0.730),
            "right_knee": (0.683, 0.734),
        }
        result = evaluate(figure(low), get_pose("downdog"))
        assert cue_keys(result)[0] == "hip_angle"
        advice = next(
            c.advice() for c in result.corrections(99) if c.check.key == "hip_angle"
        )
        assert "推高" in advice.zh

    def test_a_rounded_back_is_flagged(self):
        # Shoulders sunk towards the floor (0.640 -> 0.790) breaks the
        # wrist-shoulder-hip line down to 143 degrees.  The stance is
        # lengthened at the same time so the hip fold stays inside its band
        # and only the back check fires.
        rounded = {
            **DOWN_DOG_SIDE,
            "left_shoulder": (0.372, 0.790),
            "right_shoulder": (0.375, 0.794),
            "left_elbow": (0.261, 0.845),
            "right_elbow": (0.264, 0.849),
            "left_knee": (0.715, 0.645),
            "right_knee": (0.718, 0.649),
            "left_ankle": (0.870, 0.870),
            "right_ankle": (0.873, 0.874),
            "left_heel": (0.910, 0.898),
            "right_heel": (0.913, 0.902),
            "left_foot_index": (0.815, 0.900),
            "right_foot_index": (0.818, 0.904),
        }
        result = evaluate(figure(rounded), get_pose("downdog"))
        assert cue_keys(result) == ["back_long"]

    def test_an_upward_dog_is_not_a_downward_dog(self):
        result = evaluate(figure(UP_DOG_SIDE), get_pose("downdog"))
        assert "hip_angle" in cue_keys(result)
        assert result.score < 70


class TestUpDog:
    def test_hips_resting_on_the_mat_are_flagged(self):
        # The Cobra a beginner drifts into: pelvis lowered to y=0.890, which
        # puts it on the wrist -> ankle floor line instead of 0.31 torso
        # lengths above it.  Everything else still reads as a good Up Dog.
        cobra = {
            **UP_DOG_SIDE,
            "left_hip": (0.346, 0.890),
            "right_hip": (0.349, 0.894),
            "left_knee": (0.595, 0.870),
            "right_knee": (0.598, 0.874),
        }
        result = evaluate(figure(cobra), get_pose("updog"))
        assert cue_keys(result) == ["hips_off_floor"]

    def test_a_plank_is_told_to_lift_the_chest(self):
        # Plank shares the straight arms, the stacked shoulders and the
        # straight legs; only the torso angle separates the two, so it had
        # better be the check that speaks up.
        result = evaluate(figure(PLANK_SIDE), get_pose("updog"))
        assert cue_keys(result) == ["torso_incline"]
        advice = next(
            c.advice() for c in result.corrections(99) if c.check.key == "torso_incline"
        )
        assert "平板" in advice.zh

    def test_a_bent_knee_is_flagged_on_that_side_only(self):
        bent = {**UP_DOG_SIDE, "left_knee": (0.595, 0.760)}
        result = evaluate(figure(bent), get_pose("updog"))
        assert cue_keys(result) == ["leg_straight_left"]

    def test_too_upright_and_too_flat_get_opposite_advice(self):
        flat = evaluate(figure(PLANK_SIDE), get_pose("updog"))
        # Hips dragged forward under the shoulders: the torso stands up at 5
        # degrees off vertical instead of 35.  The knee follows onto the new
        # hip -> ankle midpoint so the legs stay straight.
        upright = evaluate(
            figure(
                {
                    **UP_DOG_SIDE,
                    "left_hip": (0.205, 0.850),
                    "right_hip": (0.208, 0.854),
                    "left_knee": (0.524, 0.850),
                    "right_knee": (0.527, 0.854),
                }
            ),
            get_pose("updog"),
        )
        flat_advice = next(
            c.advice() for c in flat.corrections(99) if c.check.key == "torso_incline"
        )
        upright_advice = next(
            c.advice() for c in upright.corrections(99) if c.check.key == "torso_incline"
        )
        assert flat_advice.zh != upright_advice.zh
        assert "向后" in upright_advice.zh
        assert "平板" in flat_advice.zh


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
