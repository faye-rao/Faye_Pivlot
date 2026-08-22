"""The rule that turns one measurement into one piece of advice."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import clamp
from .landmarks import Skeleton
from .metrics import Metric, resolve


@dataclass(frozen=True)
class Text:
    """A short bilingual string.  Chinese is the default; English is a fallback
    for terminals or fonts that cannot render CJK."""

    zh: str
    en: str

    def get(self, lang: str = "zh") -> str:
        return self.zh if lang == "zh" else self.en


@dataclass(frozen=True)
class Check:
    """One alignment rule: measure something, compare it to a target band.

    ``low``/``high`` bound the band the pose should sit in.  Use ``-inf`` or
    ``inf`` for one-sided rules ("the knee must not pass the ankle" has no
    lower bound).  ``falloff`` says how far outside the band counts as
    completely wrong -- it sets how quickly the score decays, and is in the
    same unit as the metric.
    """

    key: str
    label: Text
    metric: Metric
    low: float = -math.inf
    high: float = math.inf
    falloff: float = 20.0
    weight: float = 1.0
    when_low: Text | None = None
    when_high: Text | None = None
    unit: str = "°"
    #: Landmark name templates (``{s}``/``{o}`` allowed) the renderer should
    #: highlight when this check fails.
    focus: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"check {self.key}: low > high")
        if self.falloff <= 0:
            raise ValueError(f"check {self.key}: falloff must be positive")
        if math.isinf(self.low) and self.when_low is not None:
            raise ValueError(f"check {self.key}: when_low set but no lower bound")
        if math.isinf(self.high) and self.when_high is not None:
            raise ValueError(f"check {self.key}: when_high set but no upper bound")

    def target_text(self) -> str:
        """The target band as a human-readable string, e.g. ``80~110°``.

        Lives on the check rather than the result so the documentation
        generator can print the band without evaluating anything.
        """
        low_inf = math.isinf(self.low)
        high_inf = math.isinf(self.high)
        if low_inf and high_inf:
            return "-"
        if low_inf:
            return f"≤{self.high:g}{self.unit}"
        if high_inf:
            return f"≥{self.low:g}{self.unit}"
        return f"{self.low:g}~{self.high:g}{self.unit}"

    def evaluate(self, skeleton: Skeleton, side: str) -> "CheckResult":
        value = self.metric(skeleton, side)
        if value is None:
            return CheckResult(check=self, side=side, value=None, score=None)
        if value < self.low:
            score = clamp(1.0 - (self.low - value) / self.falloff, 0.0, 1.0)
        elif value > self.high:
            score = clamp(1.0 - (value - self.high) / self.falloff, 0.0, 1.0)
        else:
            score = 1.0
        return CheckResult(check=self, side=side, value=value, score=score)


@dataclass
class CheckResult:
    """Outcome of one :class:`Check` on one frame."""

    check: Check
    side: str
    value: float | None
    score: float | None
    #: A check passes if it is within the band, or close enough that nagging
    #: about it would be noise.
    pass_threshold: float = field(default=0.9, repr=False)

    @property
    def measured(self) -> bool:
        return self.score is not None

    @property
    def ok(self) -> bool:
        return self.score is not None and self.score >= self.pass_threshold

    @property
    def severity(self) -> float:
        """0 when perfect, 1 when hopeless.  Used to rank the advice."""
        if self.score is None:
            return 0.0
        return (1.0 - self.score) * self.check.weight

    def advice(self) -> Text | None:
        """The correction to show, or ``None`` when nothing is wrong."""
        if self.value is None or self.ok:
            return None
        if self.value < self.check.low:
            return self.check.when_low
        if self.value > self.check.high:
            return self.check.when_high
        return None

    def focus_landmarks(self) -> tuple[str, ...]:
        return tuple(resolve(name, self.side) for name in self.check.focus)

    def target_text(self) -> str:
        return self.check.target_text()

    def value_text(self) -> str:
        if self.value is None:
            return "--"
        # Angles read fine as whole degrees; torso-length ratios need decimals.
        digits = 0 if self.check.unit == "°" else 2
        return f"{self.value:.{digits}f}{self.check.unit}"
