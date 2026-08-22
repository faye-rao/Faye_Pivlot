"""Frame-to-frame state: smoothing, pose tracking, hold timing.

A single frame is a noisy sample.  Landmarks jitter by a few pixels, one bad
frame can drop the score by ten points, and advice that changes every 33 ms is
unreadable.  Everything in this module exists to turn a stream of per-frame
:class:`~yoga_coach.evaluator.PoseResult` objects into something a human can
actually follow while holding a posture.

Time is passed in by the caller rather than read from the clock, so the
behaviour is deterministic in tests and correct when replaying a video file
faster than real time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .checks import CheckResult, Text
from .evaluator import PoseResult, evaluate, rank_poses
from .geometry import Ema, Point
from .landmarks import PART_NAMES_ZH, Skeleton
from .poses import POSES, PoseSpec


def _framing_notice(skeleton: Skeleton) -> Text:
    """Say *what* cannot be seen, not just that something cannot.

    "Get your whole body in frame" is useless advice to someone who believes
    their whole body is in frame -- and in a side-on pose they are usually
    right.  Naming the parts turns it into something actionable, and when the
    parts named look wrong it points at the detector rather than the camera.
    """
    missing = skeleton.missing_parts()
    if not missing:
        return Text(
            "关键点不够可靠，评分暂停——检查光线和背景",
            "Landmarks too unreliable to score -- check the lighting and background",
        )
    zh = "、".join(PART_NAMES_ZH.get(part, part) for part in missing)
    en = ", ".join(part.replace("_", " ") for part in missing)
    return Text(
        f"看不到{zh}，请后退或调整摄像头角度",
        f"Cannot see: {en}. Step back or move the camera",
    )


class SkeletonSmoother:
    """Per-landmark exponential smoothing of the raw detector output."""

    def __init__(self, alpha: float = 0.5) -> None:
        self.alpha = alpha
        self._previous: dict[str, Point] = {}

    def update(self, skeleton: Skeleton) -> Skeleton:
        smoothed: dict[str, Point] = {}
        for name, point in skeleton.points.items():
            previous = self._previous.get(name)
            if previous is None:
                smoothed[name] = point
            else:
                a = self.alpha
                smoothed[name] = Point(
                    previous.x + a * (point.x - previous.x),
                    previous.y + a * (point.y - previous.y),
                    previous.z + a * (point.z - previous.z),
                    # Visibility follows the new frame directly: smoothing it
                    # would keep a landmark "visible" after it left the frame.
                    point.visibility,
                )
        self._previous = smoothed
        return Skeleton(points=smoothed, min_visibility=skeleton.min_visibility)

    def reset(self) -> None:
        self._previous = {}


@dataclass
class SessionState:
    """What the renderer and the console reporter draw for one frame."""

    result: PoseResult | None
    score: float
    corrections: list[CheckResult]
    hold_seconds: float
    best_hold: float
    holds_completed: int
    #: Set when there is nothing useful to judge (no body, half out of frame).
    notice: Text | None = None

    @property
    def in_pose(self) -> bool:
        return self.hold_seconds > 0.0


@dataclass
class CoachSession:
    """Stateful coach for one practice session.

    ``pose`` fixes the posture being practised; leave it ``None`` to let the
    session pick the best match each frame (``--pose auto``).
    """

    pose: PoseSpec | None = None
    library: tuple[PoseSpec, ...] = POSES
    #: Score at or above which the posture counts as "held".
    hold_threshold: float = 78.0
    #: A hold this long counts as one completed round.
    hold_target: float = 20.0
    #: Minimum seconds a set of corrections stays on screen before it is
    #: replaced, so the advice can be read.
    advice_dwell: float = 1.2
    #: In auto mode, another pose must beat the current one by this many
    #: points for this many consecutive frames before the session switches.
    switch_margin: float = 6.0
    switch_frames: int = 8

    smoother: SkeletonSmoother = field(default_factory=SkeletonSmoother)
    score_filter: Ema = field(default_factory=lambda: Ema(alpha=0.25))

    _current: PoseSpec | None = field(default=None, init=False)
    _challenger: str | None = field(default=None, init=False)
    _challenger_frames: int = field(default=0, init=False)
    _hold: float = field(default=0.0, init=False)
    _best_hold: float = field(default=0.0, init=False)
    _holds_completed: int = field(default=0, init=False)
    _hold_counted: bool = field(default=False, init=False)
    _last_time: float | None = field(default=None, init=False)
    _advice: list[CheckResult] = field(default_factory=list, init=False)
    _advice_time: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._current = self.pose

    @property
    def best_hold(self) -> float:
        """Longest single hold so far, in seconds."""
        return self._best_hold

    @property
    def holds_completed(self) -> int:
        """Number of holds that reached :attr:`hold_target`."""
        return self._holds_completed

    @property
    def current_pose(self) -> PoseSpec | None:
        """The pose being coached right now (auto mode changes this)."""
        return self._current

    # -- main entry point ---------------------------------------------------

    def update(self, skeleton: Skeleton | None, now: float) -> SessionState:
        dt = 0.0 if self._last_time is None else max(0.0, now - self._last_time)
        self._last_time = now

        if skeleton is None:
            self.smoother.reset()
            self.score_filter.reset()
            self._reset_hold()
            self._advice = []
            return self._state(
                None,
                0.0,
                Text("没有检测到人，请退到摄像头能拍到全身的位置", "No body detected -- step back into frame"),
            )

        skeleton = self.smoother.update(skeleton)
        result = self._select_pose(skeleton)

        if not result.confident:
            self.score_filter.reset()
            self._reset_hold()
            self._advice = []
            return self._state(result, result.score, _framing_notice(skeleton))

        score = self.score_filter.update(result.score) or result.score
        self._update_hold(score, dt)
        self._update_advice(result, now)
        return self._state(result, score, None)

    def reset(self) -> None:
        """Forget the practice history but keep the configuration."""
        self.smoother.reset()
        self.score_filter.reset()
        self._reset_hold()
        self._best_hold = 0.0
        self._holds_completed = 0
        self._advice = []
        self._current = self.pose
        self._challenger = None
        self._challenger_frames = 0

    # -- internals ----------------------------------------------------------

    def _select_pose(self, skeleton: Skeleton) -> PoseResult:
        if self.pose is not None:
            return evaluate(skeleton, self.pose)

        ranked = rank_poses(skeleton, self.library)
        best = ranked[0]
        if self._current is None:
            self._commit(best.pose)
            return best

        current = next(r for r in ranked if r.pose.key == self._current.key)
        if best.pose.key == current.pose.key:
            self._challenger = None
            self._challenger_frames = 0
            return current

        # Require a sustained, clear win before switching: adjacent poses
        # (Warrior I vs Warrior II) trade places constantly on single frames.
        if best.score < current.score + self.switch_margin:
            self._challenger = None
            self._challenger_frames = 0
            return current

        if self._challenger == best.pose.key:
            self._challenger_frames += 1
        else:
            self._challenger = best.pose.key
            self._challenger_frames = 1

        if self._challenger_frames >= self.switch_frames:
            self._commit(best.pose)
            return best
        return current

    def _commit(self, pose: PoseSpec) -> None:
        self._current = pose
        self._challenger = None
        self._challenger_frames = 0
        self._reset_hold()
        self._advice = []
        self.score_filter.reset()

    def _update_hold(self, score: float, dt: float) -> None:
        if score >= self.hold_threshold:
            self._hold += dt
            self._best_hold = max(self._best_hold, self._hold)
            if not self._hold_counted and self._hold >= self.hold_target:
                self._holds_completed += 1
                self._hold_counted = True
        else:
            self._reset_hold()

    def _reset_hold(self) -> None:
        self._hold = 0.0
        self._hold_counted = False

    def _update_advice(self, result: PoseResult, now: float) -> None:
        fresh = result.corrections()
        if not self._advice:
            self._advice = fresh
            self._advice_time = now
            return
        # Drop cues the moment they are fixed, but wait out the dwell time
        # before promoting new ones -- otherwise the list churns every frame.
        still_wrong = {r.check.key for r in fresh}
        kept = [r for r in self._advice if r.check.key in still_wrong]
        if len(kept) != len(self._advice):
            self._advice = kept
            self._advice_time = now
        if now - self._advice_time >= self.advice_dwell:
            self._advice = fresh
            self._advice_time = now
        else:
            # Refresh the measured values of the cues already on screen so the
            # numbers track the body even while the wording is held steady.
            by_key = {r.check.key: r for r in result.results}
            self._advice = [by_key.get(r.check.key, r) for r in self._advice]

    def _state(
        self, result: PoseResult | None, score: float, notice: Text | None
    ) -> SessionState:
        return SessionState(
            result=result,
            score=score,
            corrections=list(self._advice),
            hold_seconds=self._hold,
            best_hold=self._best_hold,
            holds_completed=self._holds_completed,
            notice=notice,
        )
