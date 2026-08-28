"""Drawing the coach overlay on a video frame.

Two things make this more than a handful of ``cv2`` calls:

* ``cv2.putText`` cannot render Chinese.  Text therefore goes through Pillow
  with a CJK font; if no such font can be found on the machine the renderer
  falls back to the English wording rather than drawing tofu boxes.
* The skeleton is coloured by *what is wrong*: joints named in a failing
  check are marked, so the advice in the panel has something to point at.
"""

from __future__ import annotations

import glob
import os
import sys
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .landmarks import SKELETON_EDGES, Skeleton
from .session import SessionState

# Palette in RGB.  cv2 wants BGR, hence _bgr() below.
GOOD = (86, 199, 132)
OKAY = (240, 189, 79)
BAD = (233, 96, 96)
INK = (245, 245, 245)
MUTED = (176, 182, 194)
PANEL = (18, 20, 26)

FONT_ENV = "YOGA_COACH_FONT"

_FONT_CANDIDATES = (
    # Linux
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
)

_FONT_GLOBS = (
    "/usr/share/fonts/**/NotoSansCJK*.*",
    "/usr/share/fonts/**/NotoSerifCJK*.*",
    "/usr/share/fonts/**/*wqy*.*",
    "/usr/local/share/fonts/**/*CJK*.*",
    str(os.path.expanduser("~/.fonts/**/*CJK*.*")),
)


def find_cjk_font() -> str | None:
    """Locate a font that can draw Chinese, or ``None`` if there is none."""
    override = os.environ.get(FONT_ENV)
    if override:
        return override if os.path.exists(override) else None
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            return path
    for pattern in _FONT_GLOBS:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def score_colour(score: float) -> tuple[int, int, int]:
    if score >= 85:
        return GOOD
    if score >= 70:
        return OKAY
    return BAD


@dataclass
class _Sizes:
    title: int
    body: int
    small: int
    huge: int


class Overlay:
    """Renders :class:`~yoga_coach.session.SessionState` onto BGR frames."""

    def __init__(self, lang: str = "zh", font_path: str | None = None) -> None:
        self.font_path = font_path or find_cjk_font()
        self.lang = lang
        if lang == "zh" and self.font_path is None:
            print(
                "找不到中文字体，界面文字改用英文。"
                f"可用环境变量 {FONT_ENV} 指定字体文件路径。",
                file=sys.stderr,
            )
            self.lang = "en"
        self._font_cache: dict[int, ImageFont.FreeTypeFont] = {}

    # -- fonts --------------------------------------------------------------

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        cached = self._font_cache.get(size)
        if cached is not None:
            return cached
        if self.font_path:
            try:
                font = ImageFont.truetype(self.font_path, size)
            except OSError:
                font = ImageFont.load_default(size)
        else:
            font = ImageFont.load_default(size)
        self._font_cache[size] = font
        return font

    def _sizes(self, width: int) -> _Sizes:
        scale = max(0.75, min(1.6, width / 960.0))
        return _Sizes(
            title=int(26 * scale),
            body=int(20 * scale),
            small=int(16 * scale),
            huge=int(52 * scale),
        )

    # -- public API ---------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        state: SessionState,
        skeleton: Skeleton | None,
        *,
        show_details: bool = False,
    ) -> np.ndarray:
        """Return ``frame`` with the skeleton and the coaching panel drawn on."""
        canvas = frame
        if skeleton is not None:
            canvas = self._draw_skeleton(canvas, skeleton, state)
        return self._draw_panel(canvas, state, show_details=show_details)

    # -- skeleton -----------------------------------------------------------

    def _draw_skeleton(
        self, frame: np.ndarray, skeleton: Skeleton, state: SessionState
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        # Red is reserved for the joints a cue is pointing at, so the rest of
        # the skeleton only turns green once the pose is actually good.
        good = state.result is not None and state.result.confident and state.score >= 85
        base = _bgr(GOOD if good else MUTED)

        flagged: set[str] = set()
        for correction in state.corrections:
            flagged.update(correction.focus_landmarks())

        def pixel(name: str) -> tuple[int, int] | None:
            point = skeleton.get(name)
            if point is None:
                return None
            return (int(point.x * width), int(point.y * height))

        for a, b in SKELETON_EDGES:
            pa, pb = pixel(a), pixel(b)
            if pa is None or pb is None:
                continue
            hot = a in flagged or b in flagged
            cv2.line(frame, pa, pb, _bgr(BAD) if hot else base, 3 if hot else 2, cv2.LINE_AA)

        for name in skeleton.points:
            if name.endswith(("_eye", "_eye_inner", "_eye_outer", "_ear")) or name.startswith("mouth"):
                continue
            position = pixel(name)
            if position is None:
                continue
            if name in flagged:
                cv2.circle(frame, position, 11, _bgr(BAD), 2, cv2.LINE_AA)
                cv2.circle(frame, position, 4, _bgr(BAD), -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, position, 3, base, -1, cv2.LINE_AA)

        # Synthetic mid-body points are worth showing: several cues talk about
        # the torso line rather than a real joint.
        for name in ("mid_shoulder", "mid_hip"):
            if name in flagged:
                position = pixel(name)
                if position is not None:
                    cv2.circle(frame, position, 9, _bgr(BAD), 2, cv2.LINE_AA)
        return frame

    # -- panel --------------------------------------------------------------

    def _draw_panel(
        self, frame: np.ndarray, state: SessionState, *, show_details: bool
    ) -> np.ndarray:
        height, width = frame.shape[:2]
        sizes = self._sizes(width)
        # Narrow enough to leave the middle of the frame -- where the body
        # actually is -- unobstructed.
        panel_width = int(min(max(width * 0.30, 280), 420))

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, height), _bgr(PANEL), -1)
        frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image)
        pad = int(sizes.body * 0.9)
        y = pad
        inner = panel_width - 2 * pad

        if state.result is None:
            draw.text((pad, y), self._t("瑜伽姿势教练", "Yoga Coach"), font=self.font(sizes.title), fill=INK)
            y += int(sizes.title * 1.8)
            if state.notice is not None:
                y = self._paragraph(draw, state.notice.get(self.lang), pad, y, inner, sizes.body, MUTED)
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        result = state.result
        colour = score_colour(state.score)
        # A score computed from the third of the pose that is still in frame
        # is not a score.  Say so instead of printing a confident number.
        reliable = result.confident

        # Pose name and which side is working.
        name = result.pose.name.get(self.lang)
        side = result.side_label.get(self.lang)
        if side:
            name = f"{name} · {side}" if self.lang == "zh" else f"{name} ({side})"
        draw.text((pad, y), name, font=self.font(sizes.title), fill=INK if reliable else MUTED)
        y += int(sizes.title * 1.35)
        draw.text((pad, y), result.pose.sanskrit, font=self.font(sizes.small), fill=MUTED)
        y += int(sizes.small * 2.0)

        if reliable:
            draw.text((pad, y), f"{state.score:.0f}", font=self.font(sizes.huge), fill=colour)
            score_width = draw.textlength(f"{state.score:.0f}", font=self.font(sizes.huge))
            draw.text(
                (pad + score_width + 8, y + sizes.huge - sizes.small * 2),
                self._t("分", "/100"),
                font=self.font(sizes.small),
                fill=MUTED,
            )
            y += int(sizes.huge * 1.25)
            y = self._bar(
                draw, pad, y, inner, max(6, sizes.small // 2), state.score / 100.0, colour
            )
            y += int(sizes.body * 0.9)

            if state.in_pose:
                hold = self._t(f"已保持 {state.hold_seconds:.1f} 秒", f"held {state.hold_seconds:.1f}s")
                draw.text((pad, y), hold, font=self.font(sizes.body), fill=GOOD)
            else:
                draw.text(
                    (pad, y),
                    self._t("调整中…", "adjusting..."),
                    font=self.font(sizes.body),
                    fill=MUTED,
                )
            y += int(sizes.body * 1.9)
        else:
            y = self._paragraph(
                draw,
                self._t("画面不完整，无法评分", "Not enough of you in frame to score"),
                pad,
                y,
                inner,
                sizes.body,
                OKAY,
            )
            y += int(sizes.body * 0.9)

        # Corrections, worst first.
        if state.corrections:
            draw.text(
                (pad, y),
                self._t("调整建议", "Corrections"),
                font=self.font(sizes.small),
                fill=MUTED,
            )
            y += int(sizes.small * 1.9)
            for index, correction in enumerate(state.corrections, start=1):
                advice = correction.advice()
                if advice is None:
                    continue
                y = self._paragraph(
                    draw,
                    f"{index}. {advice.get(self.lang)}",
                    pad,
                    y,
                    inner,
                    sizes.body,
                    INK,
                )
                if show_details:
                    detail = (
                        f"    {correction.check.label.get(self.lang)}: "
                        f"{correction.value_text()} → {correction.target_text()}"
                    )
                    y = self._paragraph(draw, detail, pad, y, inner, sizes.small, MUTED)
                y += int(sizes.body * 0.45)
        elif reliable:
            draw.text(
                (pad, y),
                self._t("体式到位，保持住！", "Alignment looks good -- hold it!"),
                font=self.font(sizes.body),
                fill=GOOD,
            )
            y += int(sizes.body * 1.9)
            y = self._paragraph(
                draw, result.pose.cue.get(self.lang), pad, y, inner, sizes.small, MUTED
            )

        if state.notice is not None:
            y += int(sizes.body * 0.6)
            self._paragraph(draw, state.notice.get(self.lang), pad, y, inner, sizes.small, OKAY)

        # Footer: session totals and key bindings.
        footer = self._t(
            f"最长保持 {state.best_hold:.0f}s · 完成 {state.holds_completed} 组",
            f"best {state.best_hold:.0f}s · {state.holds_completed} rounds",
        )
        keys = self._t("q 退出 · d 详情 · r 重置", "q quit · d details · r reset")
        draw.text((pad, height - pad - sizes.small * 3), footer, font=self.font(sizes.small), fill=MUTED)
        draw.text((pad, height - pad - sizes.small), keys, font=self.font(sizes.small), fill=MUTED)

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # -- small helpers ------------------------------------------------------

    def _t(self, zh: str, en: str) -> str:
        return zh if self.lang == "zh" else en

    def _bar(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        width: int,
        height: int,
        fraction: float,
        colour: tuple[int, int, int],
    ) -> int:
        fraction = max(0.0, min(1.0, fraction))
        radius = height // 2
        draw.rounded_rectangle([x, y, x + width, y + height], radius=radius, fill=(52, 56, 66))
        if fraction > 0:
            filled = max(int(width * fraction), height)
            draw.rounded_rectangle([x, y, x + filled, y + height], radius=radius, fill=colour)
        return y + height

    def _wrap(self, draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
        """Wrap text to ``max_width``, per word for Latin and per glyph for CJK."""
        if not text:
            return [""]
        tokens: list[str] = []
        buffer = ""
        for char in text:
            if char == " ":
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(" ")
            elif ord(char) > 0x2E80:  # CJK and friends: breakable anywhere
                if buffer:
                    tokens.append(buffer)
                    buffer = ""
                tokens.append(char)
            else:
                buffer += char
        if buffer:
            tokens.append(buffer)

        lines: list[str] = []
        current = ""
        for token in tokens:
            candidate = current + token
            if current and draw.textlength(candidate, font=font) > max_width:
                lines.append(current.rstrip())
                current = "" if token == " " else token
            else:
                current = candidate
        if current.strip():
            lines.append(current.rstrip())
        return lines or [""]

    def _paragraph(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        x: int,
        y: int,
        width: int,
        size: int,
        colour: tuple[int, int, int],
    ) -> int:
        font = self.font(size)
        for line in self._wrap(draw, text, font, width):
            draw.text((x, y), line, font=font, fill=colour)
            y += int(size * 1.4)
        return y
