"""保持段检测：正位几乎只出现在体式稳住的那几秒。

这是整条流水线里最可靠的一层过滤 —— 判断「动没动」比判断「姿势对不对」
稳得多，而且能把上千帧砍到几十段，几乎不误杀。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import landmarks as L
from .extract import FramePose


@dataclass
class Segment:
    """一段身体基本静止的连续采样帧。"""

    start: int          # 采样序列下标（含）
    end: int            # 采样序列下标（含）
    t_start: float
    t_end: float
    duration: float
    mean_velocity: float

    @property
    def indices(self) -> range:
        return range(self.start, self.end + 1)


def velocities(frames: list[FramePose]) -> np.ndarray:
    """逐帧速度，单位「躯干长度/秒」。未检出的帧记为 inf。

    速度由两部分相加：

    * 姿态速度 —— 髋心归一化坐标下主干关键点的平均位移，反映肢体构型的变化；
    * 整体速度 —— 髋中点自身的位移，反映人整体在移动。

    只算前者的话，「保持同一姿势往后退一步」会被误判成静止；只算后者的话，
    原地挥手臂会被误判成静止。两者相加把这两种漏判都堵上。
    """
    n = len(frames)
    vel = np.full(n, np.inf)
    if n < 2:
        return vel

    hip_mid_idx = (L.IDX["left_hip"], L.IDX["right_hip"])

    for i in range(1, n):
        prev, cur = frames[i - 1], frames[i]
        dt = cur.t - prev.t
        if not prev.detected or not cur.detected or dt <= 1e-6:
            continue

        pose_shift = float(
            np.linalg.norm(cur.norm[L.CORE] - prev.norm[L.CORE], axis=1).mean()
        )

        # 髋心平移：在各自的躯干尺度下比较，量纲才和 pose_shift 一致。
        prev_px = L.to_pixels(prev.lm, 1000, 1000)
        cur_px = L.to_pixels(cur.lm, 1000, 1000)
        scale = 0.5 * (L.torso_scale(prev_px) + L.torso_scale(cur_px))
        prev_hip = prev_px[list(hip_mid_idx)].mean(axis=0)
        cur_hip = cur_px[list(hip_mid_idx)].mean(axis=0)
        global_shift = float(np.linalg.norm(cur_hip - prev_hip) / max(scale, 1e-6))

        vel[i] = (pose_shift + global_shift) / dt

    # 首帧没有前一帧可比，借用第二帧的速度，免得整段开头被判为「未知」。
    vel[0] = vel[1] if np.isfinite(vel[1]) else np.inf
    return vel


def auto_threshold(
    vel: np.ndarray, percentile: float = 30.0, lo: float = 0.08, hi: float = 0.60
) -> float:
    """自适应速度阈值。

    不同视频的节奏差异很大（流瑜伽 vs 阴瑜伽），固定阈值不通用。取速度分布
    的一个低分位数当阈值，等于说「最慢的那三成帧算保持」，再夹到合理区间。
    """
    finite = vel[np.isfinite(vel)]
    if finite.size == 0:
        return hi
    return float(np.clip(np.percentile(finite, percentile), lo, hi))


def find_holds(
    frames: list[FramePose],
    vel: np.ndarray,
    threshold: float,
    min_hold: float = 1.0,
    min_visibility: float = 0.5,
) -> list[Segment]:
    """把速度低于阈值的连续帧聚成保持段，短于 ``min_hold`` 秒的丢掉。"""
    segments: list[Segment] = []
    run: list[int] = []

    def flush() -> None:
        if not run:
            return
        start, end = run[0], run[-1]
        t_start, t_end = frames[start].t, frames[end].t
        # 段尾那一帧本身也占了一个采样间隔的时长。
        step = (
            (frames[end].t - frames[start].t) / max(end - start, 1)
            if end > start
            else 0.0
        )
        duration = (t_end - t_start) + step
        if duration + 1e-6 >= min_hold:
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    t_start=t_start,
                    t_end=t_end,
                    duration=duration,
                    mean_velocity=float(np.mean([vel[i] for i in run])),
                )
            )
        run.clear()

    for i, frame in enumerate(frames):
        still = (
            frame.detected
            and np.isfinite(vel[i])
            and vel[i] < threshold
            and frame.visibility >= min_visibility
        )
        if still:
            run.append(i)
        else:
            flush()
    flush()

    return segments
