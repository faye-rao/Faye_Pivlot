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
        match = score_by_key(
            cand.frame.norm,
            dominant,
            cand.frame.lm[:, 2] if cand.frame.lm is not None else None,
        )
        cand.alignment = match.score if match else None
        # 让标签跟随簇的结论，避免同一簇里出现两个体式名。
        if match is not None:
            cand.pose = match
    return dominant


def _split_by_pose(groups: dict[int, list[Candidate]]) -> dict[int, list[Candidate]]:
    """把一个骨架簇按**逐帧识别结果**拆开，然后才投票。

    骨架距离聚类只保证「这些帧的姿势彼此接近」，不保证它们是同一个体式。
    俯卧类体式在 ``cluster_dist=0.35`` 下尤其糊：真实视频里 331 个候选聚出
    13 簇，其中**一簇 159 帧横跨 1.7~31.9 分钟、含 6 个体式**（直臂斜板 115、
    四柱支撑式 22、半神猴式 9、上犬式 6、反板式 4、婴儿式 3）。

    不拆的后果有两重，第二重更糟：

    * 这一簇只拿到九宫格里的**一个**格子，另外五个体式连竞争的机会都没有；
    * ``_assign_alignment`` 会把主导体式强加给簇里每一帧，于是四柱支撑式的
      帧被当成直臂斜板重新打分，report.md 写出「直臂斜板，身体成一直线 118°，
      目标 178±10」—— 一条**针对错误体式**的纠正建议。认不出来只是没帮上忙，
      指着四柱式说你的平板不直是在帮倒忙。

    拆的判据是「有没有**确信的**分歧」：按识别出体式的帧分组。识别不出的帧
    在明显是少数时并入最大的那一份 —— 擦边没过门槛的帧和主导体式几何上确实
    接近，跟着它拿一个偏低的正位分正是 ``_assign_alignment`` 想要的效果
    （见那里的注释），单独拆出来只会凭空多一个「未识别」簇去抢格子。

    但「少数」这个前提要当真。一簇 6 帧站姿里 2 帧过了山式门槛、4 帧没过
    （越站越松，较屈的膝从 160° 掉到 146°），无条件并入就是把少数派的标签
    盖到多数派头上 —— 和这个函数要修的那个错误一模一样，只是低了一层。
    真实视频里的后果是九宫格那一格写着「山式 0.63」，而用户的原话是
    「随意站着手臂没有下垂不是山式」。未识别的帧多于最大的那一份时，
    它们自己成一簇，标签就老实写「未识别体式」。
    """
    out: dict[int, list[Candidate]] = {}
    next_id = max(groups, default=-1) + 1

    for cluster_id in sorted(groups):
        group = groups[cluster_id]
        by_key: dict[str | None, list[Candidate]] = {}
        for cand in group:
            by_key.setdefault(cand.pose.key if cand.pose else None, []).append(cand)

        recognized = {k: v for k, v in by_key.items() if k is not None}
        unknown = by_key.get(None, [])
        if not recognized:
            out[cluster_id] = group  # 整簇未识别：只是「未知」，不代表彼此不同
            continue

        largest = max(recognized, key=lambda k: len(recognized[k]))
        if len(unknown) <= len(recognized[largest]):
            recognized[largest].extend(unknown)
            unknown = []

        if len(recognized) == 1 and not unknown:
            out[cluster_id] = group
            continue

        # 最大的那一份留用原簇号，其余各分一个新号。
        out[cluster_id] = recognized.pop(largest)
        for key in sorted(recognized):
            out[next_id] = recognized[key]
            next_id += 1
        if unknown:
            out[next_id] = unknown
            next_id += 1

    # 让 scores.json 里的 cluster 反映拆分后的分组（关掉合并时也要正确）。
    for cluster_id, group in out.items():
        for cand in group:
            cand.cluster = cluster_id
    return out


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
    skip_unrecognized: bool = False,
) -> SelectionReport:
    if not candidates:
        return SelectionReport([], 0, 0, count)

    groups: dict[int, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        groups[cand.cluster].append(cand)

    # 先按逐帧识别拆，再投票，最后合。顺序很要紧：投票必须在拆完之后，
    # 否则一个混着 6 个体式的簇会把多数派的标签盖到所有帧上（见 _split_by_pose）。
    groups = _split_by_pose(dict(groups))
    dominant = {cid: _assign_alignment(group) for cid, group in groups.items()}
    if merge_same_pose:
        groups = _merge_by_pose(dict(groups), dominant)

    # 未识别的簇默认参与竞争：练了模板之外的体式不该被丢掉。但排除某个体式
    # key 只是让模板不再认领这些帧，帧本身照旧靠画质和保持时长抢格子 ——
    # 想让它们彻底不出现，只有这个开关。
    if skip_unrecognized:
        kept = {
            cid: group
            for cid, group in groups.items()
            if any(c.pose is not None for c in group)
        }
        if kept:
            groups = kept
            surviving = {id(c) for group in groups.values() for c in group}
            candidates = [c for c in candidates if id(c) in surviving]

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
    #
    # 间隔要求逐级放宽：守死一个间隔会让九宫格空出格子，而 3×3 里空一格看起来
    # 就是坏的。宁可补一张时间上近一些的，也比留洞好 —— 用了多大间隔会记在
    # note 里，看图的人能判断这一格的成色。
    if len(picks) < count:
        for gap in (fill_min_gap, fill_min_gap / 2.0, fill_min_gap / 4.0, 0.0):
            if len(picks) >= count:
                break
            chosen = {id(c) for c in picks}
            spare = sorted(
                (c for c in candidates if id(c) not in chosen),
                key=lambda c: c.rank_score(),
                reverse=True,
            )
            for cand in spare:
                if len(picks) >= count:
                    break
                same_pose = [p for p in picks if p.cluster == cand.cluster]
                if any(abs(cand.t - p.t) < gap for p in same_pose):
                    continue
                cand.note = f"同体式补位（间隔 {min(abs(cand.t - p.t) for p in same_pose):.0f}s）" if same_pose else "同体式补位"
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
