"""Deciding what the coach should say out loud, and when.

Kept separate from :mod:`yoga_coach.voice` on purpose: choosing the words is
policy and can be unit-tested against synthetic session states, while actually
speaking them needs an audio device and a TTS engine.

The hard part is not the wording, it is the silence.  A coach that only reads
out corrections leaves you unable to tell "you are doing it right" apart from
"the camera lost you" -- and when your eyes are on the floor in Downward Dog,
those two sound identical.  So the announcer also speaks the transitions:
which pose it thinks you are in, when you reach alignment, when a hold
completes, and when it cannot see you.

Corrections are throttled; transitions are not.  Hearing "hold it" two
seconds late is useless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checks import Text
from .session import SessionState


@dataclass(frozen=True)
class Announcement:
    """One thing to say, and how badly it needs to get through.

    ``force`` marks the transitions -- the pose you just entered, reaching
    alignment, finishing a hold.  Those bypass the speaker's rate limit,
    because "hold it" delivered three seconds late is worse than useless.
    Corrections are not forced: if the speaker is mid-sentence, dropping one
    is better than stacking up advice about a posture you have already left.
    """

    text: Text
    force: bool = False


@dataclass
class Announcer:
    """Turns a stream of :class:`SessionState` into things worth saying.

    Returns at most one :class:`Text` per frame, or ``None`` for silence.
    Time is passed in rather than read from the clock, so the throttling is
    deterministic in tests.
    """

    #: Seconds between two different corrections.
    correction_gap: float = 5.0
    #: Seconds before repeating the *same* correction -- longer, because
    #: hearing the identical sentence every five seconds is maddening.
    repeat_gap: float = 14.0
    #: Seconds between repeats of "I cannot see you".
    notice_gap: float = 8.0
    #: How long alignment must hold before it is announced.  Must exceed the
    #: session's advice dwell: when one fault is fixed and another is about to
    #: surface, the correction list is briefly empty, and announcing on that
    #: gap says "that's it" a tenth of a second before the next correction.
    align_dwell: float = 1.5

    _pose: tuple[str, str] | None = field(default=None, init=False)
    _aligned: bool = field(default=False, init=False)
    _aligned_since: float | None = field(default=None, init=False)
    _undo: tuple[str | None, float, float] | None = field(default=None, init=False)
    _rounds: int = field(default=0, init=False)
    _last_key: str | None = field(default=None, init=False)
    _last_time: float = field(default=-1e9, init=False)
    _notice_time: float = field(default=-1e9, init=False)

    def update(self, state: SessionState, now: float) -> Announcement | None:
        if state.result is None or state.notice is not None:
            return self._say_notice(state, now)

        result = state.result
        identity = (result.pose.key, result.side)

        # A new pose is the most important thing to say: everything after it
        # is advice about a posture the listener may not think they are in.
        if identity != self._pose:
            self._pose = identity
            self._aligned = False
            self._aligned_since = None
            self._last_key = None
            self._last_time = -1e9
            return Announcement(_pose_name(result), force=True)

        if state.holds_completed > self._rounds:
            self._rounds = state.holds_completed
            return Announcement(
                Text(
                    f"完成一组，保持了 {state.hold_seconds:.0f} 秒",
                    f"Round complete, held {state.hold_seconds:.0f} seconds",
                ),
                force=True,
            )

        # "Hold it" only once nothing is left to correct.  The hold timer
        # starts at a lower bar than a clean pose, so announcing on the timer
        # alone would say "that's it" and then immediately contradict itself
        # with a correction.  This matches what the on-screen panel shows.
        aligned = state.in_pose and not state.corrections
        if not aligned:
            self._aligned = False
            self._aligned_since = None
        else:
            if self._aligned_since is None:
                self._aligned_since = now
            if not self._aligned and now - self._aligned_since >= self.align_dwell:
                self._aligned = True
                return Announcement(
                    Text("到位了，保持住", "That's it, hold there"), force=True
                )

        return self._say_correction(state, now)

    def undo(self) -> None:
        """Take back the last cue, because the speaker could not deliver it.

        A correction handed over while the engine is mid-sentence is dropped,
        and without this the announcer would consider it said and stay quiet
        for the whole throttle window.  Only the correction bookkeeping is
        rolled back: re-announcing a pose the listener already heard named
        would be worse than losing one cue.
        """
        if self._undo is None:
            return
        self._last_key, self._last_time, self._notice_time = self._undo
        self._undo = None

    def reset(self) -> None:
        self._pose = None
        self._aligned = False
        self._aligned_since = None
        self._undo = None
        self._rounds = 0
        self._last_key = None
        self._last_time = -1e9
        self._notice_time = -1e9

    # -- internals ----------------------------------------------------------

    def _say_notice(self, state: SessionState, now: float) -> Announcement | None:
        # Forget the pose so stepping back into frame re-announces it.
        self._pose = None
        self._aligned = False
        self._aligned_since = None
        if state.notice is None or now - self._notice_time < self.notice_gap:
            return None
        self._notice_time = now
        return Announcement(state.notice, force=True)

    def _say_correction(self, state: SessionState, now: float) -> Announcement | None:
        if not state.corrections:
            return None
        top = state.corrections[0]
        advice = top.advice()
        if advice is None:
            return None
        gap = self.repeat_gap if top.check.key == self._last_key else self.correction_gap
        if now - self._last_time < gap:
            return None
        self._undo = (self._last_key, self._last_time, self._notice_time)
        self._last_key = top.check.key
        self._last_time = now
        return Announcement(advice, force=False)


def _pose_name(result) -> Text:
    """Spoken form of the pose, naming the working side when there is one."""
    side = result.side_label
    if not side.zh:
        return Text(result.pose.name.zh, result.pose.name.en)
    return Text(
        f"{result.pose.name.zh}，{side.zh}",
        f"{result.pose.name.en}, {side.en} side",
    )
