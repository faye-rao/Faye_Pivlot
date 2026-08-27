"""从候选帧里挑出九宫格的九张。

两级排序，刻意用不同的依据：

* **簇内**（同一体式里挑哪一帧）—— 按正位分，因为同体式共用一套角度模板，
  分数可比。
* **簇间**（哪九个体式进九宫格）—— 按画质分和保持时长，**不用**正位分。
  不同体式的正位分宽严不一，拿来横向比较会系统性地偏袒模板宽松的体式，
  甚至把没识别出体式的帧一律挤掉。
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .poses import score_by_key
from .score import Candidate


@dataclass
class SelectionReport:
    picks: list[Candidate]
    n_clusters: int
    n_filled: int          # 体式种类不够、用同体式另一帧补的数量
    requested: int


def _assign_alignment(group: list[Candidate]) -> str | None:
    """定出簇内主导体式，并用它给每一帧算可比的正位分。

    返回主导体式的 key（没有任何帧识别出体式时返回 None）。
    """
    votes: dict[str, float] = defaultdict(float)
    for cand in group:
        if cand.pose is not None:
            # 用匹配分加权投票，擦边匹配的话语权小。
            votes[cand.pose.key] += cand.pose.score

    if not votes:
        for cand in group:
            cand.alignment = None
        return None

    dominant = max(votes, key=lambda k: votes[k])
    for cand in group:
        match = score_by_key(cand.frame.norm, dominant)
        cand.alignment = match.score if match else None
        # 让标签跟随簇的结论，避免同一簇里出现两个体式名。
        if match is not None:
            cand.pose = match
    return dominant


def total_hold(group: list[Candidate]) -> float:
    """簇的累计保持时长。

    同一簇的帧可能来自多个保持段（体式重复做了几轮），按段去重再累计，
    否则同一次保持里留下的两帧会把时长算成两倍。
    """
    return sum({c.segment_id: c.segment_duration for c in group}.values())


def cluster_rank_score(
    representative: Candidate, group: list[Candidate], hold_saturation: float = 6.0
) -> float:
    """簇间排序分：决定哪些体式进九宫格。

    刻意只用画质分和保持时长，不用正位分 —— 不同体式的模板宽严不一，
    横向比较会系统性偏袒容差松的体式。

    独立成函数供 ``explain`` 复用，免得诊断输出和真实选帧逻辑各算一套。
    """
    return 0.55 * representative.quality + 0.45 * float(
        np.clip(total_hold(group) / hold_saturation, 0.0, 1.0)
    )


def _merge_by_pose(
    groups: dict[int, list[Candidate]], dominant: dict[int, str | None]
) -> dict[int, list[Candidate]]:
    """把主导体式相同的簇并成一个。

    骨架距离聚类会因为机位角度、深度歧义把同一个体式拆成两簇；既然目标是
    「九个不同体式」，判成同一个体式的簇就该合并，否则九宫格里会出现两个桥式。
    没识别出体式的簇（dominant 为 None）各自独立，不参与合并 —— 它们只是
    「未知」，不代表彼此相同。
    """
    merged: dict[int, list[Candidate]] = {}
    by_key: dict[str, int] = {}

    for cluster_id in sorted(groups):
        key = dominant.get(cluster_id)
        if key is None:
            merged[cluster_id] = groups[cluster_id]
            continue
        target = by_key.setdefault(key, cluster_id)
        merged.setdefault(target, []).extend(groups[cluster_id])

    # 让 scores.json 里的 cluster 反映合并后的分组。
    for cluster_id, group in merged.items():
        for cand in group:
            cand.cluster = cluster_id
    return merged


def select(
    candidates: list[Candidate],
    count: int = 9,
    order: str = "time",
    hold_saturation: float = 6.0,
    fill_min_gap: float = 8.0,
    merge_same_pose: bool = True,
) -> SelectionReport:
    if not candidates:
        return SelectionReport([], 0, 0, count)

    groups: dict[int, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        groups[cand.cluster].append(cand)

    dominant = {cid: _assign_alignment(group) for cid, group in groups.items()}
    if merge_same_pose:
        groups = _merge_by_pose(dict(groups), dominant)

    ranked: list[tuple[float, Candidate]] = []
    for group in groups.values():
        representative = max(group, key=lambda c: c.rank_score())
        ranked.append(
            (cluster_rank_score(representative, group, hold_saturation), representative)
        )

    ranked.sort(key=lambda pair: pair[0], reverse=True)
    picks = [cand for _, cand in ranked[:count]]
    n_filled = 0

    # 体式种类不足时，用已入选体式的另一次保持来补位 —— 要求时间上离得够远，
    # 否则补进来的会是几乎相同的一张。
    if len(picks) < count:
        chosen = set(id(c) for c in picks)
        spare = sorted(
            (c for c in candidates if id(c) not in chosen),
            key=lambda c: c.rank_score(),
            reverse=True,
        )
        for cand in spare:
            if len(picks) >= count:
                break
            same_pose = [p for p in picks if p.cluster == cand.cluster]
            if any(abs(cand.t - p.t) < fill_min_gap for p in same_pose):
                continue
            cand.note = "同体式补位"
            picks.append(cand)
            n_filled += 1

    if order == "score":
        picks.sort(key=lambda c: c.rank_score(), reverse=True)
    else:
        picks.sort(key=lambda c: c.t)

    for slot, cand in enumerate(picks):
        cand.selected = True
        cand.grid_slot = slot

    return SelectionReport(
        picks=picks,
        n_clusters=len(groups),
        n_filled=n_filled,
        requested=count,
    )
