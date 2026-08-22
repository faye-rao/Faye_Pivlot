"""Recording a practice session so the target bands can be tuned afterwards.

Telling someone mid-pose to "write down the pose, the check, the measured
value and how it felt" does not work: their hands are on the mat and their
eyes are on the floor.  The program already knows the first three, so it
should keep them itself and hand back a summary at the end.  All the
practitioner has to remember is how it felt.

The summary is deliberately shaped around the one question that matters for
tuning: *which bands are wrong for this body?*  A check whose median sits
outside its band for a whole session is a band worth revisiting -- or a real
alignment habit worth working on, which is exactly the conversation the
numbers should start.
"""

from __future__ import annotations

import csv
import statistics
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .checks import Check
from .session import SessionState


def _display_width(text: str) -> int:
    """Terminal columns a string occupies; CJK glyphs take two."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


@dataclass
class CheckRecord:
    """Every measurement of one check within one pose, across the session."""

    check: Check
    values: list[float] = field(default_factory=list)
    passes: int = 0

    @property
    def samples(self) -> int:
        return len(self.values)

    @property
    def pass_rate(self) -> float:
        return self.passes / self.samples if self.samples else 0.0

    @property
    def median(self) -> float:
        return statistics.median(self.values)

    def verdict(self) -> str:
        """Where the middle of the session sat relative to the band."""
        middle = self.median
        if middle < self.check.low:
            return "低"
        if middle > self.check.high:
            return "高"
        return ""


@dataclass
class Logbook:
    """Per-check statistics for a practice session, plus an optional CSV.

    ``interval`` throttles sampling: at 30 fps every frame would be thousands
    of near-identical rows, and the median does not get better for it.
    """

    interval: float = 0.5
    #: Ignore the first moments in a pose -- you are still arranging yourself,
    #: and those frames would drag every median towards "wrong".
    settle: float = 2.0

    poses: dict[str, dict[str, CheckRecord]] = field(default_factory=dict)
    pose_names: dict[str, str] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)

    _last_sample: float = field(default=-1e9, init=False)
    _pose_since: float | None = field(default=None, init=False)
    _current: str | None = field(default=None, init=False)

    def record(self, state: SessionState, now: float, *, collect_rows: bool = False) -> bool:
        """Take a sample if one is due.  Returns whether anything was stored."""
        result = state.result
        if result is None or not result.confident:
            self._pose_since = None
            self._current = None
            return False

        key = result.pose.key
        if key != self._current:
            self._current = key
            self._pose_since = now
        if self._pose_since is None or now - self._pose_since < self.settle:
            return False
        if now - self._last_sample < self.interval:
            return False
        self._last_sample = now

        self.pose_names.setdefault(key, result.pose.name.zh)
        self.seconds[key] = self.seconds.get(key, 0.0) + self.interval
        records = self.poses.setdefault(key, {})

        for item in result.results:
            if not item.measured:
                continue
            record = records.get(item.check.key)
            if record is None:
                record = CheckRecord(check=item.check)
                records[item.check.key] = record
            record.values.append(item.value)
            if item.ok:
                record.passes += 1
            if collect_rows:
                self.rows.append(
                    {
                        "秒": f"{now:.1f}",
                        "体式": result.pose.name.zh,
                        "侧": result.side_label.zh,
                        "总分": f"{result.score:.0f}",
                        "检查项": item.check.label.zh,
                        "实测": item.value_text(),
                        "目标": item.target_text(),
                        "通过": "是" if item.ok else "否",
                    }
                )
        return True

    @property
    def empty(self) -> bool:
        return not self.poses

    def write_csv(self, path: str | Path) -> None:
        """Raw samples, for looking at the whole session rather than a median."""
        target = Path(path)
        if not self.rows:
            return
        # utf-8-sig so Excel on a Chinese Windows opens it without mojibake.
        with target.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def summary_lines(self) -> list[str]:
        """The end-of-session table, worst-behaved check first."""
        if self.empty:
            return ["这次没有采集到有效数据（身体可能一直没有完整入镜）。"]

        out: list[str] = []
        for key, records in self.poses.items():
            held = self.seconds.get(key, 0.0)
            out.append("")
            out.append(f"{self.pose_names.get(key, key)}  采样 {held:.0f} 秒")

            ranked = sorted(records.values(), key=lambda r: r.pass_rate)
            labels = [r.check.label.zh for r in ranked]
            width = max([_display_width(s) for s in labels] + [_display_width("检查项")])

            out.append(
                f"  {_pad('检查项', width)}  {'通过率':>6}  {'中位数':>9}  {'目标区间':>12}"
            )
            for record in ranked:
                digits = 0 if record.check.unit == "°" else 2
                middle = f"{record.median:.{digits}f}{record.check.unit}"
                verdict = record.verdict()
                flag = f"  ← 长期偏{verdict}" if verdict else ""
                out.append(
                    f"  {_pad(record.check.label.zh, width)}  "
                    f"{record.pass_rate * 100:5.0f}%  {middle:>9}  "
                    f"{record.check.target_text():>12}{flag}"
                )

        out.append("")
        out.append(
            "「长期偏高/偏低」表示整段练习的中位数都在区间外——要么是这条区间不适合你，"
            "要么是一个值得练的习惯。"
        )
        return out
