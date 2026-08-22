"""Voice selection and graceful degradation, without an audio device.

The failure this guards against is silent: hand Chinese text to an
English-only SAPI5 voice and Windows produces noise or nothing, raising
nothing anywhere. So the matcher is tested against how each platform actually
describes its voices.
"""

from types import SimpleNamespace

import pytest

from yoga_coach.checks import Text
from yoga_coach.voice import Speaker, _voice_matches_chinese


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
    engine = FakeEngine(voices)
    monkeypatch.setattr(Speaker, "_run", lambda self: None)
    speaker = Speaker.__new__(Speaker)
    speaker.min_gap = 3.0
    speaker.lang = lang
    speaker._last_spoken = None
    speaker._last_time = -1e9
    import queue

    speaker._queue = queue.Queue(maxsize=1)
    speaker._engine = engine
    speaker._thread = None
    if lang == "zh":
        speaker.lang = speaker._select_chinese_voice()
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

    def test_a_full_queue_drops_rather_than_blocks(self, monkeypatch):
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN", name="Huihui")]
        )
        assert speaker.say(Text("一", "one"), 10.0) is True
        # Queue still full: previous cue not consumed by the speech thread.
        assert speaker.say(Text("二", "two"), 20.0) is False

    def test_empty_text_says_nothing(self, monkeypatch):
        speaker, _ = make_speaker(
            monkeypatch, [voice(id="zh-CN", name="Huihui")]
        )
        assert speaker.say(Text("", ""), 10.0) is False
