"""The pose library judged against hand-built figures.

Each pose gets three kinds of test: a well-aligned figure scores near 100, a
specific mistake produces the specific cue that names it, and the pose is
distinguishable from the others so ``--pose auto`` picks it.
"""

import pytest
from figures import (
    DOWN_DOG_HEELS_UP,
    DOWN_DOG_SIDE,
    FAR_SIDE_OCCLUDED,
    PLANK_SIDE,
    STANDING,
    TREE_LEFT,
    UP_DOG_SIDE,
    WARRIOR_II_RIGHT,
    figure,
    mirrored,
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

#: The three side-on poses, which is where the far half of the body is hidden.
SIDE_ON = {"plank": PLANK_SIDE, "downdog": DOWN_DOG_SIDE, "updog": UP_DOG_SIDE}


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

    def head(self, dy):
        return figure(
            {
                **PLANK_SIDE,
                "left_ear": (0.265, 0.500 + dy),
                "right_ear": (0.262, 0.504 + dy),
            }
        )

    def test_a_neutral_neck_passes(self):
        assert "neck_neutral" not in cue_keys(evaluate(self.head(0.0), get_pose("plank")))

    def test_head_lifted_and_chin_tucked_get_opposite_advice(self):
        """Two distinct faults -- "gaze down" is useless to someone whose chin
        is already on their chest."""
        lifted = evaluate(self.head(-0.09), get_pose("plank"))
        tucked = evaluate(self.head(0.09), get_pose("plank"))
        assert "neck_neutral" in cue_keys(lifted)
        assert "neck_neutral" in cue_keys(tucked)
        up = next(c.advice() for c in lifted.corrections(99) if c.check.key == "neck_neutral")
        down = next(c.advice() for c in tucked.corrections(99) if c.check.key == "neck_neutral")
        assert "抬头" in up.zh
        assert "下巴" in down.zh
        assert up.zh != down.zh

    def test_the_neck_cue_survives_the_mirror(self):
        lifted = {**PLANK_SIDE, "left_ear": (0.265, 0.410), "right_ear": (0.262, 0.414)}
        straight = evaluate(figure(lifted), get_pose("plank"))
        flipped = evaluate(figure(mirrored(lifted)), get_pose("plank"))
        assert {c.advice().zh for c in straight.corrections(99)} == {
            c.advice().zh for c in flipped.corrections(99)
        }

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


class TestDownDog:
    """Folding deeper is the pose, not a fault.

    Reported from practice: a correctly held Downward Dog was repeatedly told
    "ease out of the fold".  Pushing the hips higher and further back closes
    the hip angle, so the band's lower bound punished the better version of
    the pose -- while the spine measured *longer*, not rounder.
    """

    def deeper(self, hip_x, hip_y):
        return figure(
            {
                **DOWN_DOG_SIDE,
                "left_hip": (hip_x, hip_y),
                "right_hip": (hip_x + 0.003, hip_y + 0.004),
            }
        )

    @pytest.mark.parametrize(
        "hip_x,hip_y", [(0.52, 0.32), (0.55, 0.26), (0.58, 0.20), (0.60, 0.15)]
    )
    def test_a_deep_fold_is_never_corrected(self, hip_x, hip_y):
        result = evaluate(self.deeper(hip_x, hip_y), get_pose("downdog"))
        assert result.score > 95
        assert result.corrections() == [], f"{[c.check.key for c in result.corrections()]}"

    def test_deeper_folds_lengthen_the_spine_rather_than_round_it(self):
        """The reason there is no lower bound: the thing over-folding could
        damage is the back, and it improves as the fold deepens."""
        from yoga_coach import metrics as m

        back = m.joint_angle("{s}_wrist", "{s}_shoulder", "{s}_hip")
        shallow = back(self.deeper(0.46, 0.44), "left")
        deep = back(self.deeper(0.60, 0.15), "left")
        assert deep > shallow

    @pytest.mark.parametrize("hip_x,hip_y", [(0.44, 0.50), (0.42, 0.56)])
    def test_hips_dropping_towards_a_plank_is_still_caught(self, hip_x, hip_y):
        result = evaluate(self.deeper(hip_x, hip_y), get_pose("downdog"))
        assert "hip_angle" in cue_keys(result)
        advice = next(
            c.advice() for c in result.corrections(99) if c.check.key == "hip_angle"
        )
        assert "推高" in advice.zh

    def test_a_deep_fold_is_still_recognised_as_downward_dog(self):
        ranked = rank_poses(self.deeper(0.60, 0.15))
        assert ranked[0].pose.key == "downdog"

    def test_a_rounded_back_is_what_actually_gets_caught(self):
        """The other half of dropping the lower bound on the hip angle.

        Removing it only holds up if ``back_long`` really does fire when the
        spine rounds, so this is the test that proves the claim rather than
        asserting it in a comment.

        The shoulder already sits 0.099 above the wrist -> hip line in the
        reference figure, so rounding the back means pushing it *further* from
        that line, not dropping it: lowering it towards the line straightens
        the reading to 175 degrees first.  At (0.270, 0.520) -- arm kept
        straight by moving the elbow to the new midpoint -- the angle breaks
        to 139, while the hip fold stays at 81 and inside its band.
        """
        rounded = {
            **DOWN_DOG_SIDE,
            "left_shoulder": (0.270, 0.520),
            "right_shoulder": (0.273, 0.524),
            "left_elbow": (0.235, 0.710),
            "right_elbow": (0.238, 0.714),
        }
        result = evaluate(figure(rounded), get_pose("downdog"))
        assert cue_keys(result) == ["back_long"]

    def test_an_upward_dog_is_not_a_downward_dog(self):
        """Both are side-on, both have straight arms and straight legs.  The
        hip fold is what separates them: 66 degrees here, 131 in Up Dog."""
        result = evaluate(figure(UP_DOG_SIDE), get_pose("downdog"))
        assert "hip_angle" in cue_keys(result)
        assert result.score < 70


class TestDownDogHeels:
    """Heels reaching towards the floor, measured against the toes.

    The ball of the foot is on the ground in this pose, so the toes are the
    floor -- no ground plane has to be inferred.  Deliberately forgiving:
    heels touching down is not the standard in a general class, and forcing it
    rounds the back.
    """

    def with_lift(self, lift):
        return figure(
            {
                **DOWN_DOG_SIDE,
                "left_heel": (0.860, 0.900 - lift),
                "right_heel": (0.863, 0.904 - lift),
            }
        )

    def test_heels_on_the_floor_pass(self):
        result = evaluate(self.with_lift(0.0), get_pose("downdog"))
        assert "heel_down" not in cue_keys(result)

    @pytest.mark.parametrize("lift", [0.02, 0.04, 0.06])
    def test_a_modest_lift_is_tolerated(self, lift):
        """Roughly 2 to 8cm on an adult -- normal for most calves, and not
        worth nagging about."""
        result = evaluate(self.with_lift(lift), get_pose("downdog"))
        assert "heel_down" not in cue_keys(result)

    def test_heels_clearly_up_are_flagged(self):
        result = evaluate(figure(DOWN_DOG_HEELS_UP), get_pose("downdog"))
        assert "heel_down" in cue_keys(result)
        advice = next(
            c.advice() for c in result.corrections(99) if c.check.key == "heel_down"
        )
        assert "脚跟" in advice.zh
        # The cue must not order the impossible.
        assert "正常" in advice.zh

    def test_it_never_dominates_the_score(self):
        """Light on purpose: a body that cannot get its heels down should
        still score well on an otherwise good pose."""
        result = evaluate(figure(DOWN_DOG_HEELS_UP), get_pose("downdog"))
        assert result.score > 88

    def test_heels_up_is_still_downward_dog(self):
        assert rank_poses(figure(DOWN_DOG_HEELS_UP))[0].pose.key == "downdog"

    def test_the_check_survives_a_side_on_camera(self):
        result = evaluate(
            figure(DOWN_DOG_HEELS_UP, hidden=FAR_SIDE_OCCLUDED), get_pose("downdog")
        )
        assert "heel_down" in cue_keys(result)


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


class TestMirrorInvariance:
    """Turning round in front of the camera must not change the advice.

    A signed measurement taken relative to the direction a body segment
    points reverses when the body does.  Plank told someone facing the other
    way to lift their hips when they needed to lower them -- advice that is
    not merely useless but backwards.  This runs every figure both ways.
    """

    ALL = {
        "mountain": STANDING,
        "warrior2": WARRIOR_II_RIGHT,
        "plank": PLANK_SIDE,
        "tree": TREE_LEFT,
        "downdog": DOWN_DOG_SIDE,
        "downdog_heels_up": DOWN_DOG_HEELS_UP,
        "updog": UP_DOG_SIDE,
    }

    #: Deliberately broken versions, because a correct pose has no advice to
    #: get backwards.
    BROKEN = {
        "plank": [
            {"left_hip": (0.550, 0.640), "right_hip": (0.553, 0.644)},
            {"left_hip": (0.550, 0.430), "right_hip": (0.553, 0.434)},
        ],
        "downdog": [
            {"left_hip": (0.44, 0.50), "right_hip": (0.443, 0.504)},
        ],
        "warrior2": [
            {"right_knee": (0.760, 0.580)},
            {"left_knee": (0.300, 0.760)},
        ],
        # Up Dog measures the pelvis against the wrist->ankle floor line, the
        # same signed `line_offset` that had Plank's advice backwards, so it
        # belongs here: hips on the mat, and hips dragged forward under the
        # shoulders.
        "updog": [
            {
                "left_hip": (0.346, 0.890),
                "right_hip": (0.349, 0.894),
                "left_knee": (0.595, 0.870),
                "right_knee": (0.598, 0.874),
            },
            {
                "left_hip": (0.205, 0.850),
                "right_hip": (0.208, 0.854),
                "left_knee": (0.524, 0.850),
                "right_knee": (0.527, 0.854),
            },
        ],
    }

    @pytest.mark.parametrize("name", sorted(ALL))
    def test_scores_the_same_either_way(self, name):
        key = name.split("_")[0] if name.startswith("downdog") else name
        base = self.ALL[name]
        straight = evaluate(figure(base), get_pose(key))
        flipped = evaluate(figure(mirrored(base)), get_pose(key))
        assert flipped.score == pytest.approx(straight.score, abs=0.5)

    @pytest.mark.parametrize("key", sorted(BROKEN))
    def test_the_same_fault_gets_the_same_words_either_way(self, key):
        base = {
            "plank": PLANK_SIDE,
            "downdog": DOWN_DOG_SIDE,
            "warrior2": WARRIOR_II_RIGHT,
            "updog": UP_DOG_SIDE,
        }[key]
        for fault in self.BROKEN[key]:
            broken = {**base, **fault}
            straight = evaluate(figure(broken), get_pose(key))
            flipped = evaluate(figure(mirrored(broken)), get_pose(key))
            said = {c.advice().zh for c in straight.corrections(99)}
            said_flipped = {c.advice().zh for c in flipped.corrections(99)}
            assert said == said_flipped, f"{key} {fault}: {said} vs {said_flipped}"
            assert said, "这个骨架本来就该有建议，否则测不到反向问题"

    def test_auto_detection_is_unaffected(self):
        for name, base in self.ALL.items():
            key = name.split("_")[0] if name.startswith("downdog") else name
            assert rank_poses(figure(mirrored(base)))[0].pose.key == key, name
