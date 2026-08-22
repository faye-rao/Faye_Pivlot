"""What the coach says out loud, tested without an audio device.

The announcer is pure policy over session states, so every rule here -- what
gets said, what stays silent, what preempts what -- is checked against
synthetic postures rather than by listening to a laptop.
"""

import pytest
from figures import PLANK_SIDE, WARRIOR_II_RIGHT, figure

from yoga_coach import get_pose
from yoga_coach.announce import Announcer
from yoga_coach.checks import Text
from yoga_coach.session import CoachSession

BENT_BACK_LEG = {**WARRIOR_II_RIGHT, "left_knee": (0.300, 0.780)}
CROPPED = (
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


@pytest.fixture
def coach():
    return CoachSession(pose=get_pose("warrior2"), hold_target=1.0)


def drive(session, announcer, skeleton, frames, start=0.0, step=0.1):
    """Feed frames and collect everything the announcer decided to say."""
    said = []
    for i in range(frames):
        now = start + i * step
        cue = announcer.update(session.update(skeleton, now), now)
        if cue is not None:
            said.append(cue.text.zh)
    return said


class TestPoseAnnouncement:
    def test_names_the_pose_and_side_first(self, coach):
        announcer = Announcer()
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=2)
        assert said[0] == "战士二式，右侧"

    def test_symmetric_pose_has_no_side(self):
        session = CoachSession(pose=get_pose("mountain"))
        said = drive(session, Announcer(), figure(), frames=2)
        assert said[0] == "山式"

    def test_pose_is_announced_immediately_not_throttled(self, coach):
        # Even with a long correction gap, the pose name must not wait.
        announcer = Announcer(correction_gap=60.0)
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=2)
        assert said[0].startswith("战士二式")

    def test_switching_pose_reannounces(self):
        session = CoachSession()  # auto mode
        announcer = Announcer()
        said = drive(session, announcer, figure(PLANK_SIDE), frames=10, step=0.05)
        said += drive(session, announcer, figure(), frames=20, start=0.5, step=0.05)
        assert any("平板支撑" in s for s in said)
        assert any("山式" in s for s in said)


class TestReachingAndHolding:
    def test_says_hold_it_when_alignment_lands(self, coach):
        # align_dwell is 1.5s, so this needs more than a moment of good form.
        announcer = Announcer()
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=25)
        assert "到位了，保持住" in said

    def test_a_momentary_gap_in_corrections_does_not_trigger_it(self, coach):
        announcer = Announcer(align_dwell=1.5)
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=10)
        assert "到位了，保持住" not in said

    def test_hold_confirmation_is_said_once(self, coach):
        announcer = Announcer()
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=40)
        assert said.count("到位了，保持住") == 1

    def test_announces_a_completed_round(self, coach):
        announcer = Announcer()
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=30)
        assert any(s.startswith("完成一组") for s in said)

    def test_a_good_pose_does_not_produce_corrections(self, coach):
        announcer = Announcer()
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=60)
        # Only the transitions -- nothing nagging while the pose is correct.
        assert all(
            s.startswith(("战士二式", "到位", "完成一组")) for s in said
        ), said


    def test_never_says_hold_it_while_a_correction_stands(self, coach):
        """The hold timer starts below a clean pose, so timing alone would
        announce "that's it" and then contradict itself a frame later."""
        announcer = Announcer()
        said = []
        for i in range(60):
            now = i * 0.1
            cue = announcer.update(coach.update(figure(BENT_BACK_LEG), now), now)
            if cue is not None:
                said.append(cue.text.zh)
        assert any("后腿" in s for s in said), "应该报出后腿的问题"
        assert "到位了，保持住" not in said

    def test_says_hold_it_once_the_fault_is_fixed(self, coach):
        announcer = Announcer()
        drive(coach, announcer, figure(BENT_BACK_LEG), frames=30)
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=30, start=3.0)
        assert "到位了，保持住" in said


class TestForcing:
    """Transitions must reach the speaker even when it is rate-limiting."""

    def test_transitions_are_forced(self, coach):
        announcer = Announcer()
        forced = []
        for i in range(40):
            now = i * 0.1
            cue = announcer.update(coach.update(figure(WARRIOR_II_RIGHT), now), now)
            if cue is not None:
                forced.append((cue.text.zh, cue.force))
        assert forced, "什么都没说"
        for text, force in forced:
            assert force, f"{text} 应该强制播报"

    def test_corrections_are_not_forced(self, coach):
        announcer = Announcer()
        for i in range(30):
            now = i * 0.1
            cue = announcer.update(coach.update(figure(BENT_BACK_LEG), now), now)
            if cue is not None and "后腿" in cue.text.zh:
                assert cue.force is False
                return
        pytest.fail("没有播报后腿的建议")

    def test_losing_the_body_is_forced(self, coach):
        cue = Announcer().update(coach.update(None, 0.0), 0.0)
        assert cue is not None and cue.force


class TestUndo:
    """A cue the speaker could not deliver must be offered again."""

    def test_a_dropped_correction_is_retried_next_frame(self):
        # A long hold target keeps "round complete" from preempting the retry.
        coach = CoachSession(pose=get_pose("warrior2"), hold_target=600.0)
        announcer = Announcer(correction_gap=30.0)
        first = None
        for i in range(20):
            now = i * 0.1
            cue = announcer.update(coach.update(figure(BENT_BACK_LEG), now), now)
            if cue is not None and not cue.force:
                first = cue
                announcer.undo()  # pretend the speaker was busy
                break
        assert first is not None
        # Without undo the 30s gap would silence it; with undo it comes back.
        again = None
        for i in range(20, 30):
            now = i * 0.1
            cue = announcer.update(coach.update(figure(BENT_BACK_LEG), now), now)
            if cue is not None:
                again = cue
                break
        assert again is not None and again.text.zh == first.text.zh

    def test_undo_without_a_cue_is_harmless(self):
        Announcer().undo()

    def test_undo_does_not_repeat_the_pose_name(self, coach):
        announcer = Announcer()
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=3)
        announcer.undo()
        said += drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=5, start=0.3)
        assert said.count("战士二式，右侧") == 1


class TestCorrections:
    def test_speaks_the_top_correction(self, coach):
        announcer = Announcer()
        said = drive(coach, announcer, figure(BENT_BACK_LEG), frames=10)
        assert any("后腿" in s for s in said)

    def test_same_correction_is_not_repeated_constantly(self, coach):
        announcer = Announcer(correction_gap=5.0, repeat_gap=14.0)
        said = drive(coach, announcer, figure(BENT_BACK_LEG), frames=100, step=0.1)
        # 10 seconds of the same fault: said once on entry, not every 5s.
        assert said.count("后腿蹬直，后脚外缘压实地面") == 1

    def test_repeats_eventually(self, coach):
        announcer = Announcer(correction_gap=1.0, repeat_gap=2.0)
        said = drive(coach, announcer, figure(BENT_BACK_LEG), frames=100, step=0.1)
        assert said.count("后腿蹬直，后脚外缘压实地面") > 1


class TestLosingTheBody:
    def test_says_when_it_cannot_see_you(self, coach):
        announcer = Announcer()
        said = drive(coach, announcer, None, frames=3)
        assert said and "没有检测到人" in said[0]

    def test_notice_is_throttled(self, coach):
        announcer = Announcer(notice_gap=8.0)
        said = drive(coach, announcer, None, frames=50, step=0.1)
        assert len(said) == 1

    def test_returning_to_frame_reannounces_the_pose(self, coach):
        announcer = Announcer()
        drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=5)
        drive(coach, announcer, figure(WARRIOR_II_RIGHT, hidden=CROPPED), frames=5, start=0.5)
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=5, start=1.0)
        assert said and said[0].startswith("战士二式")

    def test_out_of_frame_is_spoken_before_any_correction(self, coach):
        announcer = Announcer()
        said = drive(
            coach, announcer, figure(BENT_BACK_LEG, hidden=CROPPED), frames=10
        )
        assert all("后腿" not in s for s in said)


class TestReset:
    def test_reset_forgets_the_pose(self, coach):
        announcer = Announcer()
        drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=5)
        announcer.reset()
        said = drive(coach, announcer, figure(WARRIOR_II_RIGHT), frames=2, start=1.0)
        assert said and said[0].startswith("战士二式")


class TestBilingual:
    def test_every_cue_has_both_languages(self, coach):
        announcer = Announcer(correction_gap=0.0, repeat_gap=0.0, notice_gap=0.0)
        cues: list[Text] = []
        for i in range(40):
            now = i * 0.1
            cue = announcer.update(coach.update(figure(BENT_BACK_LEG), now), now)
            if cue is not None:
                cues.append(cue.text)
        cue = announcer.update(coach.update(None, 5.0), 5.0)
        if cue is not None:
            cues.append(cue.text)
        assert cues
        for text in cues:
            assert text.zh and text.en, text
