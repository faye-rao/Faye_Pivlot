import pytest
from figures import PLANK_SIDE, STANDING, WARRIOR_II_RIGHT, figure

from yoga_coach import get_pose
from yoga_coach.session import CoachSession, SkeletonSmoother


def run(session, skeleton, frames, start=0.0, step=0.1):
    """Feed the same skeleton for ``frames`` frames and return the last state."""
    state = None
    for i in range(frames):
        state = session.update(skeleton, start + i * step)
    return state


class TestSkeletonSmoother:
    def test_first_frame_passes_through(self):
        skeleton = figure()
        smoothed = SkeletonSmoother(alpha=0.5).update(skeleton)
        assert smoothed.get("nose").x == pytest.approx(STANDING["nose"][0])

    def test_jitter_is_damped(self):
        smoother = SkeletonSmoother(alpha=0.5)
        smoother.update(figure())
        jumped = smoother.update(figure(nose=(0.7, 0.1)))
        assert jumped.get("nose").x == pytest.approx(0.6)

    def test_visibility_is_not_smoothed(self):
        smoother = SkeletonSmoother(alpha=0.2)
        smoother.update(figure())
        vanished = smoother.update(figure(hidden=("left_knee",)))
        assert vanished.get("left_knee") is None

    def test_reset_drops_the_history(self):
        smoother = SkeletonSmoother(alpha=0.5)
        smoother.update(figure())
        smoother.reset()
        fresh = smoother.update(figure(nose=(0.7, 0.1)))
        assert fresh.get("nose").x == pytest.approx(0.7)


class TestNoBody:
    def test_missing_skeleton_produces_a_notice(self):
        session = CoachSession(pose=get_pose("mountain"))
        state = session.update(None, 0.0)
        assert state.result is None
        assert state.notice is not None
        assert state.corrections == []

    def test_body_out_of_frame_produces_a_notice(self):
        session = CoachSession(pose=get_pose("mountain"))
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
        state = session.update(cropped, 0.0)
        assert state.notice is not None
        assert not state.in_pose


class TestHoldTimer:
    def test_time_accumulates_while_the_pose_is_good(self):
        session = CoachSession(pose=get_pose("mountain"))
        state = run(session, figure(), frames=21, step=0.1)
        assert state.score > 90
        assert state.hold_seconds == pytest.approx(2.0, abs=0.01)

    def test_timer_resets_when_the_pose_breaks(self):
        session = CoachSession(pose=get_pose("mountain"))
        run(session, figure(), frames=21, step=0.1)
        collapsed = figure(left_knee=(0.60, 0.72), right_knee=(0.40, 0.72))
        state = run(session, collapsed, frames=30, start=2.1, step=0.1)
        assert state.hold_seconds == 0.0
        # The smoothed score takes a few frames to fall through the threshold,
        # so the recorded hold is a little longer than the clean two seconds.
        assert 2.0 <= state.best_hold < 3.0

    def test_reaching_the_target_counts_a_round(self):
        session = CoachSession(pose=get_pose("mountain"), hold_target=1.0)
        state = run(session, figure(), frames=21, step=0.1)
        assert state.holds_completed == 1
        # Still the same unbroken hold: it must not count twice.
        state = run(session, figure(), frames=20, start=2.1, step=0.1)
        assert state.holds_completed == 1

    def test_reset_clears_the_history(self):
        session = CoachSession(pose=get_pose("mountain"), hold_target=1.0)
        run(session, figure(), frames=21, step=0.1)
        session.reset()
        assert session.best_hold == 0.0
        assert session.holds_completed == 0


class TestAdviceStability:
    def test_advice_is_held_long_enough_to_read(self):
        session = CoachSession(pose=get_pose("warrior2"), advice_dwell=1.0)
        bent_back_leg = {**WARRIOR_II_RIGHT, "left_knee": (0.300, 0.780)}
        session.update(figure(bent_back_leg), 0.0)
        first = [c.check.key for c in session.update(figure(bent_back_leg), 0.1).corrections]
        assert "back_leg_straight" in first

        # A different fault appears, but the dwell time has not elapsed.
        also_arms = {
            **bent_back_leg,
            "right_wrist": (0.750, 0.620),
            "right_elbow": (0.615, 0.500),
        }
        state = run(session, figure(also_arms), frames=3, start=0.2, step=0.05)
        assert [c.check.key for c in state.corrections] == first

        # After the dwell time the new fault is allowed through.
        state = run(session, figure(also_arms), frames=10, start=1.5, step=0.05)
        assert "arms_level" in [c.check.key for c in state.corrections]

    def test_fixed_cues_disappear_immediately(self):
        session = CoachSession(pose=get_pose("warrior2"), advice_dwell=5.0)
        bent_back_leg = {**WARRIOR_II_RIGHT, "left_knee": (0.300, 0.780)}
        session.update(figure(bent_back_leg), 0.0)
        assert session.update(figure(bent_back_leg), 0.1).corrections
        # Correcting it should not wait out the five-second dwell: ten frames
        # spanning a fifth of a second are enough.
        state = run(session, figure(WARRIOR_II_RIGHT), frames=10, start=0.12, step=0.02)
        assert state.corrections == []

    def test_numbers_track_the_body_while_the_wording_is_held(self):
        session = CoachSession(pose=get_pose("warrior2"), advice_dwell=10.0)
        session.update(figure({**WARRIOR_II_RIGHT, "left_knee": (0.300, 0.780)}), 0.0)
        first = session.update(
            figure({**WARRIOR_II_RIGHT, "left_knee": (0.300, 0.780)}), 0.1
        ).corrections[0]
        later = session.update(
            figure({**WARRIOR_II_RIGHT, "left_knee": (0.300, 0.760)}), 0.2
        ).corrections[0]
        assert first.check.key == later.check.key
        assert later.value != first.value


class TestAutoPoseTracking:
    def test_picks_a_pose_on_the_first_frame(self):
        session = CoachSession()
        state = session.update(figure(PLANK_SIDE), 0.0)
        assert state.result.pose.key == "plank"

    def test_holds_the_current_pose_through_a_brief_wobble(self):
        session = CoachSession(switch_frames=8)
        run(session, figure(PLANK_SIDE), frames=10, step=0.05)
        # Three frames of something else is not a pose change.
        state = run(session, figure(), frames=3, start=0.5, step=0.05)
        assert state.result.pose.key == "plank"

    def test_switches_once_the_new_pose_is_sustained(self):
        session = CoachSession(switch_frames=5)
        run(session, figure(PLANK_SIDE), frames=10, step=0.05)
        state = run(session, figure(), frames=12, start=0.5, step=0.05)
        assert state.result.pose.key == "mountain"

    def test_switching_restarts_the_hold_timer(self):
        session = CoachSession(switch_frames=3, hold_target=1.0)
        run(session, figure(), frames=40, step=0.1)
        assert session.best_hold > 1.0
        state = run(session, figure(PLANK_SIDE), frames=10, start=4.0, step=0.1)
        assert state.result.pose.key == "plank"
        assert state.hold_seconds < 1.0

    def test_a_fixed_pose_never_switches(self):
        session = CoachSession(pose=get_pose("tree"))
        state = run(session, figure(PLANK_SIDE), frames=30, step=0.05)
        assert state.result.pose.key == "tree"
        assert state.score < 60
