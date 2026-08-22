"""Scoring a skeleton against a pose, and guessing which pose it is."""

from __future__ import annotations

from dataclasses import dataclass

from .checks import CheckResult, Text
from .landmarks import Skeleton
from .poses import POSES, PoseSpec

#: Below this share of the pose's total check weight actually measured, the
#: result is reported as low confidence -- usually a body that is partly out
#: of frame or turned so far that landmarks are guessed rather than seen.
MIN_MEASURED_SHARE = 0.6


@dataclass
class PoseResult:
    """How well one frame matches one pose."""

    pose: PoseSpec
    side: str
    score: float  # 0-100
    results: list[CheckResult]
    coverage: float
    measured_share: float

    @property
    def confident(self) -> bool:
        return (
            self.coverage >= self.pose.min_coverage
            and self.measured_share >= MIN_MEASURED_SHARE
        )

    @property
    def ranking_score(self) -> float:
        """Score discounted by how much of the pose could actually be checked.

        Used only to pick between poses in auto mode.  A body cropped at the
        waist leaves most of Tree unmeasurable, and the two or three checks
        that survive can easily score 100 -- without this discount that empty
        result would beat a pose the camera can actually see.
        """
        return self.score * self.measured_share

    @property
    def side_label(self) -> Text:
        if self.pose.symmetric:
            return Text("", "")
        return Text("左侧", "left") if self.side == "left" else Text("右侧", "right")

    def corrections(self, limit: int = 3) -> list[CheckResult]:
        """The worst offenders first, capped so the overlay stays readable.

        Three cues is about what anyone can act on while holding a pose; the
        rest resurface once these are fixed.
        """
        failing = [r for r in self.results if r.advice() is not None]
        failing.sort(key=lambda r: r.severity, reverse=True)
        return failing[:limit]

    def passing(self) -> list[CheckResult]:
        return [r for r in self.results if r.ok]


def evaluate_side(skeleton: Skeleton, pose: PoseSpec, side: str) -> PoseResult:
    """Score ``skeleton`` against ``pose``, reading it as the given side."""
    results = [check.evaluate(skeleton, side) for check in pose.checks]

    total_weight = sum(r.check.weight for r in results)
    measured = [r for r in results if r.measured]
    measured_weight = sum(r.check.weight for r in measured)

    if measured_weight <= 0:
        score = 0.0
    else:
        score = (
            sum(r.score * r.check.weight for r in measured) / measured_weight * 100.0
        )

    return PoseResult(
        pose=pose,
        side=side,
        score=score,
        results=results,
        coverage=skeleton.coverage(),
        measured_share=measured_weight / total_weight if total_weight else 0.0,
    )


def evaluate(skeleton: Skeleton, pose: PoseSpec) -> PoseResult:
    """Score a pose, working out which side is the working side.

    For an asymmetric pose both readings are scored and the better one wins:
    if you are in Warrior II with the right leg forward, the right-side
    reading fits and the left-side one does not, so the coach follows you
    without being told which way round you are standing.
    """
    return max(
        (evaluate_side(skeleton, pose, side) for side in pose.sides()),
        key=lambda r: r.score,
    )


def rank_poses(
    skeleton: Skeleton, poses: tuple[PoseSpec, ...] = POSES
) -> list[PoseResult]:
    """Every pose scored against this frame, best match first.

    Used by ``--pose auto``.  The best score is a reasonable proxy for "which
    asana is this": a body in Plank scores badly on Tree's checks and vice
    versa.  It is a heuristic, not a classifier -- see
    :class:`yoga_coach.session.PoseTracker` for the hysteresis that keeps it
    from flickering between similar poses.
    """
    ranked = [evaluate(skeleton, pose) for pose in poses]
    ranked.sort(key=lambda r: r.ranking_score, reverse=True)
    return ranked
