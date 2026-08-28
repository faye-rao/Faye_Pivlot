"""Text output for headless runs and for the still-image mode."""

from __future__ import annotations

import sys

from .evaluator import PoseResult
from .session import SessionState


def _t(lang: str, zh: str, en: str) -> str:
    return zh if lang == "zh" else en


class ConsoleReporter:
    """Prints the coach's state without a window.

    Rewriting the terminal every frame is unreadable, so a line is only
    printed when the advice actually changes or after ``interval`` seconds.
    """

    def __init__(self, lang: str = "zh", interval: float = 2.0) -> None:
        self.lang = lang
        self.interval = interval
        self._last_key: tuple | None = None
        self._last_time = -1e9

    def update(self, state: SessionState, now: float) -> None:
        key = (
            state.result.pose.key if state.result else None,
            state.result.side if state.result else None,
            tuple(c.check.key for c in state.corrections),
            state.notice.en if state.notice else None,
        )
        if key == self._last_key and now - self._last_time < self.interval:
            return
        self._last_key = key
        self._last_time = now
        print(self.line(state), flush=True)

    def line(self, state: SessionState) -> str:
        if state.result is None:
            notice = state.notice.get(self.lang) if state.notice else ""
            return f"[--] {notice}"

        result = state.result
        name = result.pose.name.get(self.lang)
        side = result.side_label.get(self.lang)
        if side:
            name = f"{name}({side})"
        head = f"[{state.score:3.0f}] {name}"
        if state.in_pose:
            head += _t(self.lang, f" 保持{state.hold_seconds:.0f}s", f" held {state.hold_seconds:.0f}s")
        if state.notice is not None:
            return f"{head} · {state.notice.get(self.lang)}"
        if not state.corrections:
            return f"{head} · " + _t(self.lang, "到位，保持住", "good, hold it")
        cues = " | ".join(
            c.advice().get(self.lang) for c in state.corrections if c.advice() is not None
        )
        return f"{head} · {cues}"


def report(result: PoseResult, lang: str = "zh", stream=None) -> None:
    """Full breakdown of one evaluation -- used by the still-image mode."""
    # Resolved here rather than as a default argument: binding sys.stdout at
    # import time would ignore any later redirection of the stream.
    stream = sys.stdout if stream is None else stream
    name = result.pose.name.get(lang)
    side = result.side_label.get(lang)
    if side:
        name = f"{name} ({side})"

    print(f"{name}  {result.pose.sanskrit}", file=stream)
    # The warning comes before the number: a score computed from three of a
    # pose's ten checks looks authoritative until you know that.
    if not result.confident:
        print(
            _t(
                lang,
                "⚠ 身体没有完整入镜，体式判断和下面的分数都不可靠。",
                "! The body is only partly in frame -- the pose guess and the score below are both unreliable.",
            ),
            file=stream,
        )
    print(_t(lang, f"总分：{result.score:.0f} / 100", f"Score: {result.score:.0f} / 100"), file=stream)
    print(file=stream)

    skipped = [r for r in result.results if not r.measured]
    corrections = result.corrections(limit=99)
    if corrections:
        print(_t(lang, "需要调整：", "Corrections:"), file=stream)
        for i, item in enumerate(corrections, start=1):
            advice = item.advice()
            assert advice is not None
            print(
                f"  {i}. {advice.get(lang)}"
                f"  [{item.check.label.get(lang)} {item.value_text()} → {item.target_text()}]",
                file=stream,
            )
    elif skipped:
        print(
            _t(
                lang,
                f"看得到的部分没有问题，但有 {len(skipped)} 项没能测量。",
                f"Nothing wrong in what is visible, but {len(skipped)} checks could not be measured.",
            ),
            file=stream,
        )
    else:
        print(_t(lang, "所有检查项都通过了，很棒！", "Every check passed -- nicely done."), file=stream)

    if skipped:
        print(file=stream)
        print(
            _t(lang, "未能测量（关键点被遮挡）：", "Not measured (landmarks hidden):"),
            file=stream,
        )
        print("  " + ", ".join(r.check.label.get(lang) for r in skipped), file=stream)
