"""标准体式对照图：把入选的每一帧和它的标准线稿并排放，配上要点与偏差。

一行一个体式：[你的画面] [标准线稿] [体式名 / 正位分 / 偏差项 / 发力要点]。

用「一行一体式」而不是塞进 3×3：要点文字需要横向空间，压进方格里字号会小到
读不了，而这张图的用途恰恰是读要点、找差距。
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import reference
from .grid import GridStyle, find_cjk_font, square_crop
from .poses import score_by_key
from .reference import ACCENT, INK, INK_SOFT, PAPER
from .score import Candidate


def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:d}:{seconds - minutes * 60:04.1f}"


def _load_font(font_path: str | None, px: int):
    from pathlib import Path

    from PIL import ImageFont

    path = font_path or find_cjk_font()
    if path and Path(path).is_file():
        try:
            return ImageFont.truetype(path, px)
        except OSError:
            pass
    return ImageFont.load_default()


def build_comparison(
    raw: dict[int, np.ndarray],
    picks: list[Candidate],
    style: GridStyle,
    width: int = 1800,
    font_path: str | None = None,
) -> Image.Image:
    """拼出对照图。``raw`` 是 帧号 -> 原分辨率 BGR 帧（已遮脸）。"""
    if not picks:
        raise ValueError("没有入选帧，无法生成对照图")

    margin = round(width * 0.022)
    gap = round(width * 0.014)
    row_h = round(width * 0.20)
    panel = row_h
    text_x = margin + panel * 2 + gap * 2

    font_row = _load_font(font_path, max(17, round(row_h * 0.105)))
    font_body = _load_font(font_path, max(14, round(row_h * 0.078)))
    font_head = _load_font(font_path, max(20, round(row_h * 0.135)))
    font_foot = _load_font(font_path, max(12, round(row_h * 0.062)))

    header_h = round(row_h * 0.52)
    footer_h = round(row_h * 0.30)
    height = header_h + len(picks) * (row_h + gap) + footer_h

    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)

    draw.text((margin, round(header_h * 0.22)), "标准体式对照", font=font_head, fill=INK)
    tw = draw.textlength("标准体式对照", font=font_head)
    draw.text(
        (margin + tw + round(width * 0.012), round(header_h * 0.34)),
        "左：你的画面　中：标准对位示意　右：偏差与发力要点",
        font=font_foot, fill=INK_SOFT,
    )
    draw.line([(margin, header_h - gap), (width - margin, header_h - gap)],
              fill=(224, 220, 212), width=2)

    for i, cand in enumerate(picks):
        top = header_h + i * (row_h + gap)

        # ---- 左：你的画面 ----
        frame = raw.get(cand.frame.frame_no)
        if frame is not None:
            crop = square_crop(frame, cand.frame.bbox, style)
            cell = cv2.resize(crop, (panel, panel), interpolation=cv2.INTER_AREA)
            image.paste(Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB)),
                        (margin, top))
        else:
            draw.rectangle([margin, top, margin + panel, top + panel],
                           outline=(224, 220, 212), width=2)

        # ---- 中：标准线稿 ----
        ref_x = margin + panel + gap
        key = cand.pose.key if cand.pose else None
        if key and key in reference.CANONICAL:
            art = reference.render_pose(key, panel)
            image.paste(art, (ref_x, top))
            draw.rectangle([ref_x, top, ref_x + panel - 1, top + panel - 1],
                           outline=(224, 220, 212), width=2)
        else:
            draw.rectangle([ref_x, top, ref_x + panel - 1, top + panel - 1],
                           outline=(224, 220, 212), width=2)
            note = "未识别体式\n无标准图对照"
            for j, line in enumerate(note.split("\n")):
                lw = draw.textlength(line, font=font_body)
                draw.text(
                    (ref_x + (panel - lw) / 2, top + panel / 2 - 20 + j * 26),
                    line, font=font_body, fill=INK_SOFT,
                )

        # ---- 右：文字 ----
        y = top + round(row_h * 0.045)
        title = f"{i + 1}. {cand.pose_label}"
        draw.text((text_x, y), title, font=font_row, fill=INK)
        tw = draw.textlength(title, font=font_row)
        meta = f"{_fmt_time(cand.t)}"
        if cand.alignment is not None:
            meta += f"　正位 {cand.alignment:.2f}"
        meta += f"　画质 {cand.quality:.2f}"
        if cand.note:
            meta += f"　{cand.note}"
        draw.text((text_x + tw + round(width * 0.010), y + round(row_h * 0.028)),
                  meta, font=font_foot, fill=INK_SOFT)
        y += round(row_h * 0.155)

        line_h = round(row_h * 0.108)

        # 偏差项：直接从骨架重算，这样从 scores.json 重拼时也有（存的是关键点，
        # 逐项明细没存）。
        weak = []
        if key and cand.frame.norm is not None:
            match = score_by_key(cand.frame.norm, key)
            if match is not None:
                weak = match.weak_checks()[:2]
        if weak:
            draw.text((text_x, y), "偏差较大：", font=font_body, fill=ACCENT)
            y += line_h
            for check in weak:
                if check.value != check.value:  # nan
                    continue
                digits = 0 if check.unit == "°" else 2
                text = (
                    f"· {check.label}　实测 {check.value:.{digits}f}{check.unit}"
                    f"　目标 {check.target:.{digits}f}{check.unit}±{check.tol:.{digits}f}"
                )
                draw.text((text_x, y), text, font=font_body, fill=INK)
                y += line_h
            y += round(line_h * 0.25)

        cue_list = reference.cues(key) if key else ()
        if cue_list:
            draw.text((text_x, y), "发力要点：", font=font_body, fill=INK_SOFT)
            y += line_h
            room = (top + row_h - round(row_h * 0.04) - y) // max(line_h, 1)
            for cue in cue_list[: max(int(room), 0)]:
                draw.text((text_x, y), f"· {cue}", font=font_body, fill=INK)
                y += line_h

    draw.text(
        (margin, height - footer_h + round(footer_h * 0.25)),
        "标准图由打分模板的目标几何渲染，与正位分同源 · 发力要点为通用教学口令，"
        "不针对个人，不能替代老师指导",
        font=font_foot, fill=INK_SOFT,
    )
    return image
