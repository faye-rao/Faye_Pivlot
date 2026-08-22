"""Speaking the coach's cues out loud.

Half the point of a camera coach is that you can be in Downward Dog, where
the screen is upside down and behind you.  Speech is what makes the program
usable with your eyes on the floor.

``pyttsx3`` is an optional dependency.  Without it -- or without a working
audio device -- the coach runs exactly as before, silently, and says so once
on stderr rather than failing.

Two failures here are silent by nature, and both have bitten this program:

* **Chinese through an English voice.**  SAPI5 accepts it and produces noise
  or nothing, raising nothing anywhere.  So the speaker picks a voice matching
  the requested language, and if it cannot find one it speaks the English half
  of the cue instead of pretending.
* **A dead speech thread.**  SAPI5 is COM, and COM objects belong to the
  thread that created them; an engine built on the main thread and driven from
  a worker will typically manage one utterance and then start throwing.  The
  engine is therefore created *on* the speech thread, the thread survives
  errors instead of exiting, and a jam is reported rather than swallowed --
  an earlier version lost every cue after the first with no message at all.
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

#: Consecutive engine errors before speech gives up for the session.
_MAX_FAILURES = 3


def _voice_matches_chinese(voice) -> bool:
    haystack = [str(getattr(voice, "id", "")), str(getattr(voice, "name", ""))]
    for language in getattr(voice, "languages", None) or []:
        # pyttsx3 hands these back as bytes on some platforms.
        if isinstance(language, bytes):
            language = language.decode("utf-8", "ignore")
        haystack.append(str(language))
    blob = " ".join(haystack).lower()
    return any(marker in blob for marker in _ZH_MARKERS)


def describe_voices() -> list[str]:
    """Human-readable list of the system voices, for ``--speak-test``."""
    try:
        import pyttsx3
    except ImportError:
        return []
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty("voices") or []
    except Exception as exc:  # pragma: no cover - driver dependent
        return [f"（无法列出音色：{exc}）"]
    out = []
    for v in voices:
        mark = "中文" if _voice_matches_chinese(v) else "    "
        out.append(f"  [{mark}] {getattr(v, 'name', '?')}  ({getattr(v, 'id', '?')})")
    return out


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
        ready_timeout: float = 15.0,
    ) -> None:
        self.min_gap = min_gap
        #: The language actually spoken, which may differ from the one asked
        #: for when the machine has no matching voice installed.
        self.lang = lang
        self._requested_lang = lang
        self._rate = rate
        self._last_spoken: str | None = None
        self._last_time = -1e9
        # Two slots, so a forced cue can follow another forced one -- entering
        # a pose and reaching alignment happen a fraction of a second apart,
        # and both need to be heard.  Corrections never queue behind anything
        # (see `say`), so nothing stale can accumulate here.
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=2)
        self._ready = threading.Event()
        self._alive = False
        self._reported_stall = False
        self._thread: threading.Thread | None = None

        # Probed here only to report a missing dependency at construction
        # time; the engine itself is built on the speech thread.
        try:
            import importlib.util

            if importlib.util.find_spec("pyttsx3") is None:
                raise ImportError
        except ImportError:
            print(
                "未安装 pyttsx3，语音提示已关闭。开启方法：pip install pyttsx3",
                file=sys.stderr,
            )
            self._ready.set()
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        # Wait for the engine to come up so `lang` is settled before the first
        # cue, and so a broken engine is reported at startup rather than
        # halfway through a practice.
        self._ready.wait(timeout=ready_timeout)

    @property
    def enabled(self) -> bool:
        return self._alive and self._thread is not None and self._thread.is_alive()

    def say(self, text: Text | str, now: float, *, force: bool = False) -> bool:
        """Speak a cue.  Returns whether it was actually handed to the engine.

        Repeats and cues arriving inside ``min_gap`` are dropped, unless
        ``force`` is set -- used for things the listener must not miss, like
        the name of the pose they just moved into.
        """
        if not self.enabled:
            self._report_stall()
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

    def _report_stall(self) -> None:
        """Say once, loudly, that speech has stopped working.

        Losing every cue after the first with no message is what made the
        original bug so hard to spot from the mat.
        """
        if self._reported_stall or self._thread is None:
            return
        self._reported_stall = True
        print(
            "语音播报已停止（引擎出错或未能启动），练习继续，屏幕提示不受影响。\n"
            "  用 python -m yoga_coach --speak-test 可以单独诊断语音。",
            file=sys.stderr,
        )

    def _select_chinese_voice(self, engine) -> str:
        """Switch ``engine`` to a Mandarin voice, or fall back to English.

        Returns the language that will actually be spoken.
        """
        try:
            voices = engine.getProperty("voices") or []
        except Exception:  # pragma: no cover - driver dependent
            voices = []

        for voice in voices:
            if _voice_matches_chinese(voice):
                try:
                    engine.setProperty("voice", voice.id)
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
        engine = self._start_engine()
        self._ready.set()
        if engine is None:
            self._drain()
            return

        failures = 0
        while True:
            text = self._queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
                failures = 0
            except Exception as exc:
                failures += 1
                print(
                    f"语音播报出错（{failures}/{_MAX_FAILURES}）：{exc}",
                    file=sys.stderr,
                )
                # Never `return` here: an earlier version did, and every
                # later cue was then silently dropped into a queue nobody
                # was reading.
                try:
                    engine.stop()
                except Exception:
                    pass
                if failures >= _MAX_FAILURES:
                    print("语音连续出错，已关闭语音提示。", file=sys.stderr)
                    break

        self._alive = False
        self._drain()

    def _start_engine(self):  # pragma: no cover - needs a real audio device
        """Build the engine *on this thread*.

        SAPI5 is COM, and a COM object belongs to the thread that created it.
        Creating the engine on the main thread and driving it from here is
        what killed the speech thread after a single utterance on Windows.
        """
        try:
            # Present only on Windows, and only when pywin32 is installed --
            # pyttsx3 pulls it in for the SAPI5 driver.
            import pythoncom

            pythoncom.CoInitialize()
        except Exception:
            pass

        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", self._rate)
        except Exception as exc:
            print(f"语音引擎初始化失败，已关闭语音提示：{exc}", file=sys.stderr)
            return None

        if self._requested_lang == "zh":
            self.lang = self._select_chinese_voice(engine)
        self._alive = True
        return engine

    def _drain(self) -> None:  # pragma: no cover - needs a real audio device
        """Empty the queue so `say` reports the stall instead of jamming."""
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
