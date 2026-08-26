"""第二遍：回视频取原始分辨率的帧，裁成正方形，拼成九宫格。"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .score import Candidate

# 常见的中文字体，按平台顺序试。列表里只放确定含中日韩字形的字体，
# 找不到就退回英文标签，不去猜某个拉丁字体能不能渲染汉字。
_CJK_FONTS = (
    "/System/Library/Fonts/PingFang.ttc",                        # macOS
    "/System/Library/Fonts/Hiragino Sans GB.ttc",                # macOS
    "/Library/Fonts/Arial Unicode.ttf",                          # macOS
    "C:/Windows/Fonts/msyh.ttc",                                 # Windows 微软雅黑
    "C:/Windows/Fonts/simhei.ttf",                               # Windows 黑体
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",     # Linux
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",              # 文泉驿正黑
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
)


@dataclass
class GridStyle:
    size: int = 2048          # 输出正方形边长（像素）
    gap: int = 16             # 格子之间的间隔
    margin: int = 24          # 外边距
    background: tuple[int, int, int] = (255, 255, 255)
    pad_factor: float = 1.35  # 裁切框相对人体外框的放大倍数
    pad_mode: str = "blur"    # blur | solid | crop
    labels: bool = True
    font_path: str | None = None


def find_cjk_font() -> str | None:
    for path in _CJK_FONTS:
        if Path(path).is_file():
            return path
    return None


def _load_font(style: GridStyle, px: int) -> tuple[ImageFont.FreeTypeFont, bool]:
    """返回 (字体, 是否支持中文)。"""
    path = style.font_path or find_cjk_font()
    if path and Path(path).is_file():
        try:
            # 走到这里字体要么来自 _CJK_FONTS 列表，要么是用户显式指定的，
            # 两种情况都当作支持中文。
            return ImageFont.truetype(path, px), True
        except OSError:
            pass
    # 退而求其次：任何可用的 TrueType 字体，标签改用英文体式名。
    for fallback in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(fallback).is_file():
            try:
                return ImageFont.truetype(fallback, px), False
            except OSError:
                continue
    return ImageFont.load_default(), False


def _blurred_backdrop(frame: np.ndarray, side: int) -> np.ndarray:
    """整帧拉成正方形后重度模糊，用作裁切留白的底 —— 比纯色更耐看。"""
    small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (side, side), interpolation=cv2.INTER_LINEAR)
    ksize = max(3, (side // 12) | 1)  # 必须是奇数
    blurred = cv2.GaussianBlur(big, (ksize, ksize), 0)
    return cv2.convertScaleAbs(blurred, alpha=0.85, beta=10)


def square_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    style: GridStyle,
) -> np.ndarray:
    """以人体为中心裁一个正方形。

    ``pad_mode="crop"`` 会把裁切框压进画面内，人体可能被切掉一部分；
    ``blur`` / ``solid`` 保持裁切框大小、用背景补足，绝不切身体 ——
    竖屏视频拍站立体式时，人体外框又高又窄，压进画面就会切掉头或脚，
    所以默认不压。
    """
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = bbox
    px0, py0, px1, py1 = x0 * w, y0 * h, x1 * w, y1 * h
    cx, cy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
    side = max(px1 - px0, py1 - py0) * style.pad_factor
    side = max(side, 32.0)

    if style.pad_mode == "crop":
        side = min(side, float(min(w, h)))
        left = float(np.clip(cx - side / 2.0, 0.0, w - side))
        top = float(np.clip(cy - side / 2.0, 0.0, h - side))
        s = int(round(side))
        return frame[int(round(top)) : int(round(top)) + s,
                     int(round(left)) : int(round(left)) + s].copy()

    s = int(round(side))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))

    if style.pad_mode == "solid":
        canvas = np.empty((s, s, 3), dtype=frame.dtype)
        canvas[:] = np.asarray(style.background[::-1], dtype=frame.dtype)  # RGB -> BGR
    else:
        canvas = _blurred_backdrop(frame, s)

    sx0, sy0 = max(0, left), max(0, top)
    sx1, sy1 = min(w, left + s), min(h, top + s)
    if sx1 > sx0 and sy1 > sy0:
        canvas[sy0 - top : sy1 - top, sx0 - left : sx1 - left] = frame[sy0:sy1, sx0:sx1]
    return canvas


def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:d}:{seconds - minutes * 60:04.1f}"


def _draw_label(
    canvas: Image.Image,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
) -> None:
    """在格子底部画一条半透明色带和文字。"""
    left, top, cell, _ = box
    overlay = Image.new("RGBA", (cell, cell), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    pad = max(6, cell // 60)
    try:
        _, _, _, text_h = draw.textbbox((0, 0), text, font=font)
    except (AttributeError, TypeError):
        text_h = font.size if hasattr(font, "size") else 16
    bar_h = text_h + pad * 2

    draw.rectangle([0, cell - bar_h, cell, cell], fill=(0, 0, 0, 130))
    draw.text((pad, cell - bar_h + pad), text, font=font, fill=(255, 255, 255, 235))

    cell_img = canvas.crop((left, top, left + cell, top + cell)).convert("RGBA")
    canvas.paste(Image.alpha_composite(cell_img, overlay).convert("RGB"), (left, top))


def build_grid(
    raw: dict[int, np.ndarray],
    picks: list[Candidate],
    style: GridStyle,
    frames_dir: Path | None = None,
) -> Image.Image:
    """拼出九宫格。

    ``raw`` 是 帧号 -> 原始分辨率 BGR 帧 的映射，由调用方准备好 ——
    这样导出候选缩略图和拼图可以共用同一次视频遍历。
    ``frames_dir`` 非空时顺便把每张裁好的原分辨率单图存下来。
    """
    if not picks:
        raise ValueError("没有可用的候选帧，无法拼图")

    cols = math.ceil(math.sqrt(len(picks)))
    rows = math.ceil(len(picks) / cols)
    cell = (style.size - 2 * style.margin - (cols - 1) * style.gap) // cols
    if cell < 64:
        raise ValueError(f"输出尺寸 {style.size} 太小，放不下 {cols}x{rows} 的格子")

    width = style.margin * 2 + cols * cell + (cols - 1) * style.gap
    height = style.margin * 2 + rows * cell + (rows - 1) * style.gap
    canvas = Image.new("RGB", (width, height), style.background)

    font, cjk = _load_font(style, max(14, cell // 20))

    missing = [c for c in picks if c.frame.frame_no not in raw]
    if missing:
        print(
            f"警告：{len(missing)} 帧无法从视频重新读取，将被跳过",
            file=sys.stderr,
        )

    if frames_dir is not None:
        frames_dir.mkdir(parents=True, exist_ok=True)

    for slot, cand in enumerate(picks):
        frame = raw.get(cand.frame.frame_no)
        if frame is None:
            continue

        crop = square_crop(frame, cand.frame.bbox, style)
        cell_img = cv2.resize(crop, (cell, cell), interpolation=cv2.INTER_AREA)

        if frames_dir is not None:
            # 存原始分辨率的裁切图，而不是缩到格子大小的那张 ——
            # 单张图她可能另有用处，压缩过就找不回来了。
            name = f"{slot + 1:02d}_{cand.pose.key if cand.pose else 'unknown'}_{cand.t:07.2f}s.jpg"
            cv2.imwrite(str(frames_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        left = style.margin + (slot % cols) * (cell + style.gap)
        top = style.margin + (slot // cols) * (cell + style.gap)
        canvas.paste(
            Image.fromarray(cv2.cvtColor(cell_img, cv2.COLOR_BGR2RGB)), (left, top)
        )

        if style.labels:
            name = cand.pose_label if cjk else (cand.pose.en if cand.pose else "unknown")
            _draw_label(
                canvas, (left, top, cell, cell),
                f"{slot + 1}. {name}  {_fmt_time(cand.t)}", font,
            )

    return canvas
