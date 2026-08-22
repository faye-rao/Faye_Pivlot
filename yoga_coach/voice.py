"""Optional spoken cues.

Half the point of a camera coach is that you can be in Downward Dog, where
the screen is upside down and behind you.  Speech makes the corrections
usable there.  ``pyttsx3`` is an optional dependency: without it the coach
runs exactly as before, silently.
"""

from __future__ import annotations

import queue
import sys
import threading


class Speaker:
    """Speaks short cues on a background thread, at most one at a time.

    Speech is slow compared with the frame rate, so cues are dropped rather
    than queued: hearing a correction from six seconds ago is worse than
    hearing nothing.
    """

    def __init__(self, *, rate: int = 165, min_gap: float = 4.0) -> None:
        self.min_gap = min_gap
        self._last_spoken: str | None = None
        self._last_time = -1e9
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=1)
        self._engine = None
        self._thread: threading.Thread | None = None

        try:
            import pyttsx3
        except ImportError:
            print(
                "未安装 pyttsx3，语音提示已关闭（pip install pyttsx3 可开启）。",
                file=sys.stderr,
            )
            return

        try:
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", rate)
        except Exception as exc:  # pragma: no cover - depends on host audio
            print(f"语音引擎初始化失败，已关闭语音提示：{exc}", file=sys.stderr)
            self._engine = None
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def say(self, text: str, now: float) -> None:
        """Speak ``text`` unless it repeats the last cue or comes too soon."""
        if not self.enabled or not text:
            return
        if text == self._last_spoken and now - self._last_time < self.min_gap * 2:
            return
        if now - self._last_time < self.min_gap:
            return
        try:
            self._queue.put_nowait(text)
        except queue.Full:
            return  # still speaking the previous cue
        self._last_spoken = text
        self._last_time = now

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    def _run(self) -> None:  # pragma: no cover - needs a real audio device
        while True:
            text = self._queue.get()
            if text is None:
                return
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception:
                return
