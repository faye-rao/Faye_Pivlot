"""候选帧的构建与打分。

两套分数，刻意分开，因为它们不可互换：

``quality``
    「这一帧适不适合入选照片」—— 稳、清晰、关键点可信、人完整在画面里。
    和体式无关，因此可以跨体式比较。

``alignment``
    「这个体式做到多正位」—— 由 ``poses.py`` 的角度模板给出。只有识别出
    体式的帧才有，而且不同体式之间**不可比**（战士二的 0.8 和树式的 0.8
    宽严不同）。所以它只用于在同一个体式内部挑最好的那一帧。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .extract import FramePose
from .poses import PoseMatch, match_pose
from .stability import Segment

# quality 各分量的默认权重。
DEFAULT_WEIGHTS = {
    "visibility": 0.25,
    "sharpness": 0.20,
    "framing": 0.20,
    "stillness": 0.20,
    "hold": 0.15,
}


@dataclass
class Candidate:
    frame: FramePose
    segment_id: int
    segment_duration: float
    velocity: float
    components: dict[str, float]
    quality: float
    pose: PoseMatch | None
    cluster: int = -1
    # 聚类之后按簇内主导体式重算的正位分，见 select.py。
    alignment: float | None = None
    selected: bool = False
    grid_slot: int | None = None
    note: str = ""

    @property
    def t(self) -> float:
        return self.frame.t

    @property
    def pose_label(self) -> str:
        return self.pose.zh if self.pose else "未识别体式"

    def rank_score(self) -> float:
        """簇内排序用：正位分可得时占主导，否则退回画质分。"""
        if self.alignment is None:
            return self.quality
        return 0.45 * self.quality + 0.55 * self.alignment


def _framing_score(bbox: tuple[float, float, float, float]) -> float:
    """人体是否完整、够大地落在画面里。"""
    x0, y0, x1, y1 = bbox
    outside = (
        max(0.0, -x0) + max(0.0, -y0) + max(0.0, x1 - 1.0) + max(0.0, y1 - 1.0)
    )
    # 越界总量累计到 0.25（画面的四分之一）就扣到零分。
    inside = 1.0 - float(np.clip(outside / 0.25, 0.0, 1.0))

    longest = max(x1 - x0, y1 - y0)
    # 主体长边占画面不足 15% 太远，超过 50% 给满分。
    size = float(np.clip((longest - 0.15) / 0.35, 0.0, 1.0))
    return 0.7 * inside + 0.3 * size


def _sharpness_ranks(values: list[float]) -> list[float]:
    """把拉普拉斯方差换成 0~1 的分位排名。

    绝对值没有可比性 —— 它随分辨率、光线、裁剪面积大幅变化。同一支视频内
    比排名才有意义。
    """
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    order = np.argsort(np.asarray(values, dtype=np.float64))
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64) / (n - 1)
    return ranks.tolist()


def build_candidates(
    frames: list[FramePose],
    segments: list[Segment],
    vel: np.ndarray,
    vel_threshold: float,
    per_segment: int = 2,
    min_gap: float = 1.0,
    exclude_poses: frozenset[str] = frozenset(),
    weights: dict[str, float] | None = None,
    hold_saturation: float = 3.0,
) -> list[Candidate]:
    """从保持段里挑出候选帧并打分。

    每段最多留 ``per_segment`` 帧，且彼此间隔至少 ``min_gap`` 秒 —— 同一次
    保持里的相邻帧几乎一模一样，多留只是浪费。
    """
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    weight_sum = sum(weights.values()) or 1.0

    # 先把所有段内帧的清晰度收齐，才能算分位排名。
    in_segment: list[tuple[int, int]] = []  # (segment_id, frame idx)
    for seg_id, seg in enumerate(segments):
        for i in seg.indices:
            if frames[i].detected:
                in_segment.append((seg_id, i))

    ranks = _sharpness_ranks([frames[i].sharpness for _, i in in_segment])
    rank_by_idx = {i: r for (_, i), r in zip(in_segment, ranks)}

    scored: dict[int, list[Candidate]] = {}
    for seg_id, i in in_segment:
        frame = frames[i]
        seg = segments[seg_id]
        v = float(vel[i]) if np.isfinite(vel[i]) else vel_threshold

        components = {
            "visibility": frame.visibility,
            "sharpness": rank_by_idx[i],
            "framing": _framing_score(frame.bbox),
            "stillness": 1.0 - float(np.clip(v / max(vel_threshold, 1e-6), 0.0, 1.0)),
            "hold": float(np.clip(seg.duration / hold_saturation, 0.0, 1.0)),
        }
        quality = sum(components[k] * weights[k] for k in weights) / weight_sum

        scored.setdefault(seg_id, []).append(
            Candidate(
                frame=frame,
                segment_id=seg_id,
                segment_duration=seg.duration,
                velocity=v,
                components=components,
                quality=quality,
                pose=match_pose(
                    frame.norm,
                    exclude_poses,
                    frame.lm[:, 2] if frame.lm is not None else None,
                ),
            )
        )

    candidates: list[Candidate] = []
    for seg_id, group in scored.items():
        group.sort(key=lambda c: c.quality, reverse=True)
        kept: list[Candidate] = []
        for cand in group:
            if len(kept) >= per_segment:
                break
            if all(abs(cand.t - k.t) >= min_gap for k in kept):
                kept.append(cand)
        candidates.extend(kept)

    candidates.sort(key=lambda c: c.t)
    return candidates
