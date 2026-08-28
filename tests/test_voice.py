"""Voice selection and graceful degradation, without an audio device.

The failure this guards against is silent: hand Chinese text to an
English-only SAPI5 voice and Windows produces noise or nothing, raising
nothing anywhere. So the matcher is tested against how each platform actually
describes its voices.
"""

from types import SimpleNamespace

import pytest

from yoga_coach.checks import Text
from yoga_coach.voice import (
    Speaker,
    _voice_matches_chinese,
    default_engine_mode,
    pick_chinese_voice,
)


def voice(id="", name="", languages=None):
    return SimpleNamespace(id=id, name=name, languages=languages or [])


class TestChineseVoiceMatching:
    @pytest.mark.parametrize(
        "candidate",
        [
            # Windows SAPI5
            voice(
                id=r"HKEY_LOCAL_MACHINE\...\TTS_MS_ZH-CN_HUIHUI_11.0",
                name="Microsoft Huihui Desktop - Chinese (Simplified)",
            ),
            voice(id="TTS_MS_ZH-TW_HANHAN_11.0", name="Microsoft Hanhan Desktop"),
            # macOS NSSpeechSynthesizer
            voice(id="com.apple.voice.compact.zh-CN.Tingting", name="Tingting"),
            voice(id="com.apple.speech.synthesis.voice.sinji", name="Sinji"),
            # espeak-ng on Linux, which reports bytes
            voice(id="mb-zh1", name="Mandarin", languages=[b"\x05zh"]),
            voice(id="chinese", name="chinese"),
        ],
    )
    def test_recognises_mandarin_voices(self, candidate):
        assert _voice_matches_chinese(candidate)

    @pytest.mark.parametrize(
        "candidate",
        [
            voice(
                id=r"HKEY_LOCAL_MACHINE\...\TTS_MS_EN-US_DAVID_11.0",
                name="Microsoft David Desktop - English (United States)",
            ),
            voice(id="com.apple.voice.compact.en-GB.Daniel", name="Daniel"),
            voice(id="mb-fr1", name="French", languages=[b"\x05fr"]),
            voice(id="", name="", languages=[]),
        ],
    )
    def test_rejects_other_languages(self, candidate):
        assert not _voice_matches_chinese(candidate)

    def test_a_japanese_voice_is_not_mistaken_for_chinese(self):
        # "ja" would match a naive substring search for "a"; make sure the
        # markers are specific enough.
        assert not _voice_matches_chinese(
            voice(id="com.apple.voice.compact.ja-JP.Kyoko", name="Kyoko")
        )


class TestDegradation:
    """pyttsx3 is not installed in the test environment, which is the point:
    the coach must stay usable and say so once, rather than crashing."""

    def test_missing_engine_disables_speech(self, capsys):
        speaker = Speaker(lang="zh")
        assert not speaker.enabled
        assert "pip install pyttsx3" in capsys.readouterr().err

    def test_say_is_a_no_op_when_disabled(self):
        speaker = Speaker(lang="zh")
        assert speaker.say(Text("测试", "test"), 0.0) is False
        assert speaker.say(Text("测试", "test"), 0.0, force=True) is False

    def test_close_is_safe_when_disabled(self):
        Speaker(lang="zh").close()


class FakeEngine:
    """Minimal stand-in for a pyttsx3 engine."""

    def __init__(self, voices):
        self._voices = voices
        self.properties = {}
        self.spoken: list[str] = []

    def setProperty(self, key, value):
        self.properties[key] = value

    def getProperty(self, key):
        return self._voices if key == "voices" else self.properties.get(key)

    def say(self, text):
        self.spoken.append(text)

    def runAndWait(self):
        pass


def make_speaker(monkeypatch, voices, lang="zh"):
    """Build a Speaker around a fake engine, with no background thread."""
    import queue
    import threading

    engine = FakeEngine(voices)
    speaker = Speaker.__new__(Speaker)
    speaker.min_gap = 3.0
    speaker.lang = lang
    speaker._requested_lang = lang
    speaker._rate = 165
    speaker._last_spoken = None
    speaker._last_time = -1e9
    speaker._queue = queue.Queue(maxsize=2)
    speaker._ready = threading.Event()
    speaker._ready.set()
    speaker._alive = True
    speaker._reported_stall = False
    speaker._voice_id = None
    speaker.mode = "persistent"
    speaker._thread = SimpleNamespace(is_alive=lambda: True)
    speaker.lang = speaker._resolve_voice(engine)
    return speaker, engine


class TestVoiceSelection:
    def test_switches_to_a_chinese_voice_when_present(self, monkeypatch):
        chinese = voice(id="zh-CN-Huihui", name="Microsoft Huihui - Chinese")
        speaker, engine = make_speaker(
            monkeypatch, [voice(id="en-US-David", name="David"), chinese]
        )
        assert speaker.lang == "zh"
        assert engine.properties["voice"] == "zh-CN-Huihui"

    def test_falls_back_to_english_text_without_a_chinese_voice(
        self, monkeypatch, capsys
    ):
        speaker, engine = make_speaker(
            monkeypatch, [voice(id="en-US-David", name="David")]
        )
        assert speaker.lang == "en"
        assert "没有中文语音包" in capsys.readouterr().err
        # And the cue spoken is the English half, not Chinese through an
        # English voice.
        speaker.say(Text("后腿蹬直", "Straighten the back leg"), 0.0)
        assert speaker._queue.get_nowait() == "Straighten the back leg"

    def test_english_ui_never_hunts_for_a_chinese_voice(self, monkeypatch):
        speaker, engine = make_speaker(
            monkeypatch, [voice(id="zh-CN-Huihui", name="Huihui")], lang="en"
        )
        assert speaker.lang == "en"
        assert "voice" not in engine.properties


class FlakyEngine(FakeEngine):
    """An engine whose ``runAndWait`` throws, like SAPI5 driven cross-thread."""

    def __init__(self, voices, fail_times):
        super().__init__(voices)
        self.remaining_failures = fail_times

    def runAndWait(self):
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise RuntimeError("run loop already started")

    def stop(self):
        pass


def run_loop_with(monkeypatch, engine, cues):
    """Drive Speaker._run synchronously with a pre-filled queue."""
    import queue
    import threading

    speaker = Speaker.__new__(Speaker)
    speaker.min_gap = 3.0
    speaker.lang = "zh"
    speaker._requested_lang = "en"  # skip voice selection
    speaker._rate = 165
    speaker._last_spoken = None
    speaker._last_time = -1e9
    speaker._queue = queue.Queue(maxsize=10)
    speaker._ready = threading.Event()
    speaker._alive = False
    speaker._reported_stall = False
    speaker._voice_id = None
    speaker.mode = "persistent"
    speaker._thread = SimpleNamespace(is_alive=lambda: True)

    def fake_start(self):
        self._alive = True
        return engine

    monkeypatch.setattr(Speaker, "_start_engine", fake_start)
    for cue in cues:
        speaker._queue.put_nowait(cue)
    speaker._queue.put_nowait(None)  # sentinel: end the loop
    speaker._run()
    return speaker


class TestSpeechThreadSurvival:
    """The bug this guards: the thread used to `return` on any engine error,
    after which every later cue vanished into a queue nobody was reading --
    the user heard the startup line and then nothing, with no message."""

    def test_one_engine_error_does_not_kill_the_thread(self, monkeypatch, capsys):
        engine = FlakyEngine([], fail_times=1)
        run_loop_with(monkeypatch, engine, ["第一句", "第二句", "第三句"])
        # All three were attempted, not just the one before the error.
        assert engine.spoken == ["第一句", "第二句", "第三句"]
        assert "语音播报出错" in capsys.readouterr().err

    def test_gives_up_after_repeated_failures_and_says_so(self, monkeypatch, capsys):
        engine = FlakyEngine([], fail_times=99)
        speaker = run_loop_with(monkeypatch, engine, ["一", "二", "三", "四", "五"])
        assert not speaker.enabled
        err = capsys.readouterr().err
        assert "语音连续出错，已关闭语音提示" in err

    def test_the_queue_is_drained_when_speech_stops(self, monkeypatch):
        engine = FlakyEngine([], fail_times=99)
        speaker = run_loop_with(monkeypatch, engine, ["一", "二", "三", "四", "五"])
        # A jammed queue is what made every later `say` silently fail.
        assert speaker._queue.empty()

    def test_a_stalled_speaker_reports_once(self, monkeypatch, capsys):
        engine = FlakyEngine([], fail_times=99)
        speaker = run_loop_with(monkeypatch, engine, ["一", "二", "三", "四"])
        capsys.readouterr()
        assert speaker.say(Text("测试", "test"), 100.0) is False
        first = capsys.readouterr().err
        assert "语音播报已停止" in first
        assert speaker.say(Text("测试", "test"), 200.0) is False
        assert capsys.readouterr().err == ""  # not once per frame


class TestThrottling:
    def test_drops_cues_inside_the_minimum_gap(self, monkeypatch):
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN", name="Huihui")]
        )
        assert speaker.say(Text("一", "one"), 10.0) is True
        speaker._queue.get_nowait()
        assert speaker.say(Text("二", "two"), 11.0) is False

    def test_force_bypasses_the_gap(self, monkeypatch):
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN", name="Huihui")]
        )
        speaker.say(Text("一", "one"), 10.0)
        speaker._queue.get_nowait()
        assert speaker.say(Text("二", "two"), 10.1, force=True) is True

    def test_an_unforced_cue_never_stacks_behind_another(self, monkeypatch):
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN", name="Huihui")]
        )
        assert speaker.say(Text("一", "one"), 10.0) is True
        # The speech thread has not consumed it yet: advice about a posture
        # you may already have left is dropped rather than queued.
        assert speaker.say(Text("二", "two"), 20.0) is False

    def test_two_forced_cues_fit_then_the_third_drops(self, monkeypatch):
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN", name="Huihui")]
        )
        # Entering a pose and reaching alignment arrive together; both must
        # get through.
        assert speaker.say(Text("一", "one"), 10.0, force=True) is True
        assert speaker.say(Text("二", "two"), 10.1, force=True) is True
        assert speaker.say(Text("三", "three"), 10.2, force=True) is False

    def test_empty_text_says_nothing(self, monkeypatch):
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN", name="Huihui")]
        )
        assert speaker.say(Text("", ""), 10.0) is False


class TestEngineMode:
    """Windows needs a new engine per utterance.

    pyttsx3 on SAPI5 speaks the first cue and then goes quiet -- no
    exception, `runAndWait()` returns cleanly, and nothing comes out. No
    error handling can catch that, so the platform default sidesteps it.
    """

    def test_windows_defaults_to_a_fresh_engine(self, monkeypatch):
        monkeypatch.setattr("sys.platform", "win32")
        assert default_engine_mode() == "fresh"

    @pytest.mark.parametrize("platform", ["linux", "darwin"])
    def test_other_platforms_reuse_the_engine(self, monkeypatch, platform):
        monkeypatch.setattr("sys.platform", platform)
        assert default_engine_mode() == "persistent"

    def test_auto_resolves_to_the_platform_default(self):
        speaker = Speaker(lang="zh", engine_mode="auto")  # pyttsx3 absent here
        assert speaker.mode == default_engine_mode()

    def test_an_explicit_mode_is_kept(self):
        assert Speaker(lang="zh", engine_mode="fresh").mode == "fresh"
        assert Speaker(lang="zh", engine_mode="persistent").mode == "persistent"


class TestVoicePicking:
    def test_picks_the_first_mandarin_voice(self):
        chinese = voice(id="zh-CN-Huihui", name="Microsoft Huihui - Chinese")
        chosen = pick_chinese_voice(
            [voice(id="en-US-David", name="David"), chinese, voice(id="zh-2", name="Yaoyao")]
        )
        assert chosen is chinese

    def test_returns_none_when_there_is_no_mandarin_voice(self):
        assert pick_chinese_voice([voice(id="en-US-David", name="David")]) is None

    def test_remembers_the_voice_id_for_later_engines(self, monkeypatch):
        """In fresh mode every utterance builds a new engine, so the chosen
        voice has to be reapplied rather than rediscovered."""
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN-Huihui", name="Huihui")]
        )
        assert speaker._voice_id == "zh-CN-Huihui"
