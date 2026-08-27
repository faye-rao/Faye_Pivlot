"""从 scores.json 里存好的关键点重跑识别、聚类和选帧，跳过姿态估计。

为什么值得单独一条路：姿态估计占整条流水线约 **92%** 的耗时，而它的产物
（关键点）已经在 scores.json 里了。改一次体式模板就重跑一遍完整视频，等于把
最贵的一步白做一次 —— 十几分钟的视频要等 2~4 分钟，而重算识别本身是毫秒级。

不重算的部分
-----------
画质分及其分量（置信度、清晰度、构图、静止度、保持时长）、速度、保持段归属
都直接沿用存好的值。它们不依赖体式模板，重算也不会变；而清晰度这类还需要
像素，没必要为此再读一遍视频。

仍然需要视频的地方只有一处：入选那几张要回读原分辨率像素来出图。
"""

from __future__ import annotations

import sys
from pathlib import Path

from .extract import VideoInfo
from .poses import match_pose
from .score import Candidate


def rematch(candidates: list[Candidate], exclude: frozenset[str] = frozenset()) -> int:
    """用当前模板重新识别体式，并清掉上一轮的选帧结果。

    返回识别出体式的候选数。``cluster`` / ``alignment`` / ``selected`` 等都要
    清空 —— 它们是上一轮的结论，留着会让新一轮的聚类和补位逻辑读到脏数据。
    """
    recognized = 0
    for cand in candidates:
        cand.cluster = -1
        cand.alignment = None
        cand.selected = False
        cand.grid_slot = None
        cand.note = ""
        cand.pose = (
            match_pose(cand.frame.norm, exclude) if cand.frame.norm is not None else None
        )
        recognized += cand.pose is not None
    return recognized


def video_info_from_payload(payload: dict, path: Path) -> VideoInfo:
    """从 scores.json 的 video 段重建 VideoInfo，供 report.md 用。"""
    video = payload.get("video", {})
    fps = float(video.get("fps") or 30.0)
    duration = float(video.get("duration") or 0.0)
    frame_count = int(video.get("frame_count") or round(duration * fps))
    return VideoInfo(
        path=path,
        fps=fps if fps > 0 else 30.0,
        frame_count=frame_count,
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
    )


def usable(candidates: list[Candidate]) -> tuple[list[Candidate], int]:
    """筛出带骨架、能参与重算的候选，并返回被丢掉的数量。"""
    keep = [c for c in candidates if c.frame.norm is not None]
    return keep, len(candidates) - len(keep)


def warn_if_no_landmarks(dropped: int, total: int) -> bool:
    """没有骨架就没法重算 —— 提示怎么办，返回是否应当中止。"""
    if dropped == 0:
        return False
    print(
        f"警告：{dropped}/{total} 个候选没有可用骨架，无法参与重算。",
        file=sys.stderr,
    )
    if dropped == total:
        print(
            "这份 scores.json 里没有可用骨架 —— 或是当初用了 --no-landmarks，\n"
            "或是文件里的骨架格式无法识别。重算需要骨架，请重跑一次完整流水线：\n"
            "  python -m yoga_grid <视频>",
            file=sys.stderr,
        )
        return True
    return False
