"""把选帧决策摊开，回答「这一帧为什么没被选上」。

选帧是两级的，一帧落选只可能是这两个原因之一：

* 它所在的**体式簇没进前 N** —— 那一格给了别的体式，簇里谁都进不去；
* 簇进了，但**簇内另一帧分更高** —— 同一体式只占一格。

这里把两级排序都按真实公式重算一遍（直接复用 ``select`` 里的函数，
不另写一套，免得解释和实际行为对不上），并指出每一帧属于哪种情况。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .poses import TEMPLATES, score_by_key
from .report import load_candidates
from .score import Candidate
from .select import cluster_rank_score, total_hold


def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:d}:{seconds - minutes * 60:04.1f}"


def _pose_label(group: list[Candidate]) -> str:
    for cand in group:
        if cand.pose is not None:
            return cand.pose.zh
    return "未识别体式"


def explain(
    scores_path: Path,
    focus: float | None = None,
    max_clusters: int = 0,
    hold_saturation: float = 6.0,
) -> str:
    """生成选帧决策说明。``focus`` 是想追查的那一帧的时间戳（秒）。"""
    _, candidates, payload = load_candidates(scores_path)
    if not candidates:
        return "scores.json 里没有候选帧。"

    summary = payload.get("summary", {})
    params = payload.get("params", {})
    count = int(params.get("count") or summary.get("requested") or 9)

    groups: dict[int, list[Candidate]] = defaultdict(list)
    for cand in candidates:
        groups[cand.cluster].append(cand)

    # 复用真实公式重算两级排序。
    rows = []
    for cluster_id, group in groups.items():
        representative = max(group, key=lambda c: c.rank_score())
        rows.append(
            {
                "cluster": cluster_id,
                "group": group,
                "rep": representative,
                "score": cluster_rank_score(representative, group, hold_saturation),
                "hold": total_hold(group),
                "label": _pose_label(group),
            }
        )
    rows.sort(key=lambda r: r["score"], reverse=True)

    lines: list[str] = []
    add = lines.append

    add(f"候选 {len(candidates)} 张，体式簇 {len(groups)} 个，要挑 {count} 张，"
        f"实际入选 {sum(1 for c in candidates if c.selected)} 张")
    if summary.get("same_pose_fills"):
        add(f"其中 {summary['same_pose_fills']} 格是体式种类不够、用同体式另一次保持补的位")
    add("")

    add("── 簇间排序：决定哪些体式进九宫格（只看画质分和保持时长，不看正位分）──")
    add("")
    add(f"{'排名':>4} {'簇':>5} {'体式':<10} {'最佳画质':>8} {'总保持':>8} {'簇间分':>7}  结果")
    shown = rows if max_clusters <= 0 else rows[: max(max_clusters, count + 3)]
    for rank, row in enumerate(shown, start=1):
        picked = [c for c in row["group"] if c.selected]
        if picked:
            slots = "、".join(str((c.grid_slot or 0) + 1) for c in picked)
            outcome = f"✓ 入选（第 {slots} 格）"
        else:
            outcome = f"✗ 未进前 {count}"
        add(
            f"{rank:>4} {'c%02d' % row['cluster']:>5} {row['label']:<10} "
            f"{row['rep'].quality:>8.2f} {row['hold']:>7.1f}s {row['score']:>7.2f}  {outcome}"
        )
    if len(rows) > len(shown):
        add(f"     …… 另有 {len(rows) - len(shown)} 个簇未列出（用 --all 全部显示）")
    add("")

    # 需要展开簇内排序的簇：多于一帧的，或者用户点名追查的那一帧所在的簇。
    focus_cand: Candidate | None = None
    if focus is not None:
        focus_cand = min(candidates, key=lambda c: abs(c.t - focus))
        if abs(focus_cand.t - focus) > 1.0:
            focus_cand = None

    detail_ids = {r["cluster"] for r in rows if len(r["group"]) > 1}
    if focus_cand is not None:
        detail_ids.add(focus_cand.cluster)

    if detail_ids:
        add("── 簇内排序：决定同一体式用哪一帧（正位分主导）──")
        add("")
        for row in rows:
            if row["cluster"] not in detail_ids:
                continue
            add(f"c{row['cluster']:02d} {row['label']}（簇间排名 "
                f"{rows.index(row) + 1}/{len(rows)}，簇间分 {row['score']:.2f}）")
            add(f"    {'时间':>8} {'画质':>6} {'正位':>6} {'簇内分':>7}  结果")
            for cand in sorted(row["group"], key=lambda c: c.rank_score(), reverse=True):
                alignment = "  —  " if cand.alignment is None else f"{cand.alignment:6.2f}"
                mark = f"✓ 第 {(cand.grid_slot or 0) + 1} 格" if cand.selected else ""
                if cand.note:
                    mark += f"（{cand.note}）"
                here = "  ← 你问的这张" if cand is focus_cand else ""
                add(
                    f"    {_fmt_time(cand.t):>8} {cand.quality:>6.2f} {alignment} "
                    f"{cand.rank_score():>7.2f}  {mark}{here}"
                )
            add("")

    if focus_cand is not None and focus_cand.frame.norm is not None:
        add("── 这一帧在每个体式模板下的得分 ──")
        add("")
        add("按分数排序。`朝向` 是躯干朝向门槛的系数，为 0 说明这一帧的体位方向")
        add("根本不符合该体式（比如倒过来了），此时模板分直接归零。")
        add("")
        scored = []
        for template in TEMPLATES:
            match = score_by_key(focus_cand.frame.norm, template.key)
            if match is not None:
                scored.append((template, match))
        scored.sort(key=lambda pair: pair[1].score, reverse=True)

        add(f"    {'体式':<12} {'得分':>6} {'门槛':>6} {'朝向':>6}  判定")
        for template, match in scored:
            verdict = "✓ 过线" if match.score >= template.min_score else ""
            add(
                f"    {template.zh:<12} {match.score:>6.2f} {template.min_score:>6.2f} "
                f"{match.orientation:>6.2f}  {verdict}"
            )
        add("")

        # 展开前两名的检查明细 —— 误判的原因几乎总在这里。
        for template, match in scored[:2]:
            add(f"    「{template.zh}」逐项：")
            for check in sorted(match.checks, key=lambda c: c.score):
                if check.value != check.value:  # nan
                    continue
                digits = 0 if check.unit == "°" else 2
                add(
                    f"      {check.label:<16} 实测 {check.value:>8.{digits}f}{check.unit}"
                    f"  目标 {check.target:.{digits}f}{check.unit}±{check.tol:.{digits}f}"
                    f"  得分 {check.score:.2f}"
                )
            add("")

    if focus_cand is not None and focus_cand.frame.norm is None:
        add("（这份 scores.json 没有存骨架，无法复算各模板得分。"
            "重跑一次即可 —— 新版默认会存。）")
        add("")

    if focus_cand is not None:
        add("── 结论 ──")
        add("")
        row = next(r for r in rows if r["cluster"] == focus_cand.cluster)
        rank = rows.index(row) + 1
        add(f"你问的是 {_fmt_time(focus_cand.t)} 的「{row['label']}」"
            f"（簇 c{focus_cand.cluster:02d}，画质 {focus_cand.quality:.2f}）。")
        add("")
        if focus_cand.selected:
            add(f"它**已经入选**，在第 {(focus_cand.grid_slot or 0) + 1} 格。")
        elif rank > count:
            add(f"落选原因：**它所在的体式簇没进前 {count}**（簇间排名第 {rank}，"
                f"簇间分 {row['score']:.2f}）。这一格给了排名更高的体式，"
                f"所以这个簇里哪一帧都进不去。")
            add("")
            add(f"这个簇的最佳画质 {row['rep'].quality:.2f}、总保持 {row['hold']:.1f}s；"
                f"簇间分 = 0.55×画质 + 0.45×min(总保持/{hold_saturation:.0f}, 1)。")
            add("要让它进来：`--count 12` 多挑几张，或 `--exclude <挤掉它的体式 key>`。")
        else:
            winner = max(row["group"], key=lambda c: c.rank_score())
            add(f"落选原因：**簇进了前 {count}（第 {rank} 名），但簇内另一帧分更高**。")
            add("")
            add(f"胜出的是 {_fmt_time(winner.t)}（画质 {winner.quality:.2f}，"
                f"正位 {'—' if winner.alignment is None else f'{winner.alignment:.2f}'}，"
                f"簇内分 {winner.rank_score():.2f}）"
                f"，你问的这张簇内分 {focus_cand.rank_score():.2f}。")
            add("")
            add("簇内分 = 0.45×画质 + 0.55×正位分（没识别出体式时直接用画质分）。")
            add("如果你觉得这张明显更好，说明正位模板的判断和你的眼睛不一致 —— "
                "把两张图发我，我看是模板容差要调还是该给这个体式换算法。")
            add("")
            add("想直接换掉：在 scores.json 里把胜出那张的 `selected` 改 false、"
                "这张改 true 并抄走它的 `grid_slot`，然后 `python -m yoga_grid grid <目录>`。")
        add("")

    return "\n".join(lines)
