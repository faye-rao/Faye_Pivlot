"""Speaking the coach's cues out loud.

Half the point of a camera coach is that you can be in Downward Dog, where
the screen is upside down and behind you.  Speech is what makes the program
usable with your eyes on the floor.

``pyttsx3`` is an optional dependency.  Without it -- or without a working
audio device -- the coach runs exactly as before, silently, and says so once
on stderr rather than failing.

The subtle failure this module exists to avoid: handing Chinese text to a
voice that only speaks English.  SAPI5 on Windows will happily accept it and
produce noise or nothing at all, with no error anywhere.  So the speaker picks
a voice matching the requested language, and if it cannot find one it speaks
the English half of the cue instead of pretending.
"""

from __future__ import annotations

import queue
import sys
import threading

from .checks import Text

#: Substrings that mark a system voice as Mandarin-capable.  Different
#: platforms describe the same voice very differently: SAPI5 gives
#: "Microsoft Huihui Desktop - Chinese (Simplified)", macOS gives "Tingting"
#: with a "zh_CN" language tag, espeak gives "zh".
_ZH_MARKERS = (
    "zh",
    "chinese",
    "mandarin",
    "huihui",
    "yaoyao",
    "kangkang",
    "tingting",
    "sinji",
    "meijia",
)


def _voice_matches_chinese(voice) -> bool:
    haystack = [str(getattr(voice, "id", "")), str(getattr(voice, "name", ""))]
    for language in getattr(voice, "languages", None) or []:
        # pyttsx3 hands these back as bytes on some platforms.
        if isinstance(language, bytes):
            language = language.decode("utf-8", "ignore")
        haystack.append(str(language))
    blob = " ".join(haystack).lower()
    return any(marker in blob for marker in _ZH_MARKERS)


class Speaker:
    """Speaks short cues on a background thread, at most one at a time.

    Speech is slow compared with the frame rate, so cues are dropped rather
    than queued: hearing a correction from six seconds ago is worse than
    hearing nothing.
    """

    def __init__(
        self,
        lang: str = "zh",
        *,
        rate: int = 165,
        min_gap: float = 3.0,
    ) -> None:
        self.min_gap = min_gap
        #: The language actually spoken, which may differ from the one asked
        #: for when the machine has no matching voice installed.
        self.lang = lang
        self._last_spoken: str | None = None
        self._last_time = -1e9
        # Two slots, so a forced cue can follow another forced one -- entering
        # a pose and reaching alignment happen a fraction of a second apart,
        # and both need to be heard.  Corrections never queue behind anything
        # (see `say`), so nothing stale can accumulate here.
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self._engine = None
        self._thread: threading.Thread | None = None

        try:
            import pyttsx3
        except ImportError:
            print(
                "未安装 pyttsx3，语音提示已关闭。开启方法：pip install pyttsx3",
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

        if lang == "zh":
            self.lang = self._select_chinese_voice()

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def say(self, text: Text | str, now: float, *, force: bool = False) -> bool:
        """Speak a cue.  Returns whether it was actually handed to the engine.

        Repeats and cues arriving inside ``min_gap`` are dropped, unless
        ``force`` is set -- used for things the listener must not miss, like
        the name of the pose they just moved into.
        """
        if not self.enabled:
            return False
        spoken = text.get(self.lang) if isinstance(text, Text) else text
        if not spoken:
            return False
        if not force:
            # An unforced cue waits its turn and never stacks: if the engine
            # is still speaking, advice about the posture you were in a
            # sentence ago is not worth hearing.
            if not self._queue.empty():
                return False
            if now - self._last_time < self.min_gap:
                return False
            if spoken == self._last_spoken and now - self._last_time < self.min_gap * 3:
                return False
        try:
            self._queue.put_nowait(spoken)
        except queue.Full:
            return False  # still speaking the previous cue
        self._last_spoken = spoken
        self._last_time = now
        return True

    def close(self) -> None:
        if self._thread is None:
            return
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=2.0)

    # -- internals ----------------------------------------------------------

    def _select_chinese_voice(self) -> str:
        """Switch the engine to a Mandarin voice, or fall back to English.

        Returns the language that will actually be spoken.
        """
        try:
            voices = self._engine.getProperty("voices") or []
        except Exception:  # pragma: no cover - driver dependent
            voices = []

        for voice in voices:
            if _voice_matches_chinese(voice):
                try:
                    self._engine.setProperty("voice", voice.id)
                    return "zh"
                except Exception:  # pragma: no cover - driver dependent
                    continue

        print(
            "系统里没有中文语音包，语音提示改用英文（界面仍是中文）。\n"
            "  Windows：设置 → 时间和语言 → 语言和区域 → 添加“中文(简体)”并勾选语音\n"
            "  macOS：系统设置 → 辅助功能 → 朗读内容 → 系统嗓音 → 管理嗓音 → 中文\n"
            "  Linux：安装 espeak-ng（含 zh 音色）",
            file=sys.stderr,
        )
        return "en"

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
