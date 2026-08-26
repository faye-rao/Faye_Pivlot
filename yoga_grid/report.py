"""落盘产物：scores.json（可回改重拼）与 report.md（正位反馈）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .extract import FramePose, VideoInfo
from .poses import PoseMatch
from .score import Candidate
from .select import SelectionReport


def _fmt_time(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:d}:{seconds - minutes * 60:04.1f}"


def _pose_dict(pose: PoseMatch | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    return {
        "key": pose.key,
        "zh": pose.zh,
        "en": pose.en,
        "side": pose.side,
        "score": round(pose.score, 4),
        "orientation": round(pose.orientation, 4),
        "checks": [
            {
                "label": c.label,
                "value": None if c.value != c.value else round(c.value, 2),  # nan 安全
                "target": c.target,
                "tol": c.tol,
                "unit": c.unit,
                "score": round(c.score, 4),
            }
            for c in pose.checks
        ],
    }


def dump_json(
    path: Path,
    info: VideoInfo,
    candidates: list[Candidate],
    selection: SelectionReport,
    params: dict[str, Any],
    n_detected: int,
    n_segments: int,
) -> None:
    payload = {
        "video": {
            "path": str(info.path),
            "fps": round(info.fps, 3),
            "frame_count": info.frame_count,
            "width": info.width,
            "height": info.height,
            "duration": round(info.duration, 2),
        },
        "params": params,
        "summary": {
            "sampled_frames": params.get("sampled_frames"),
            "detected_frames": n_detected,
            "hold_segments": n_segments,
            "candidates": len(candidates),
            "pose_clusters": selection.n_clusters,
            "requested": selection.requested,
            "selected": len(selection.picks),
            "same_pose_fills": selection.n_filled,
        },
        "candidates": [
            {
                "t": round(c.t, 3),
                "frame_no": c.frame.frame_no,
                "bbox": [round(v, 5) for v in c.frame.bbox],
                "cluster": c.cluster,
                "segment_id": c.segment_id,
                "segment_duration": round(c.segment_duration, 2),
                "velocity": round(c.velocity, 4),
                "quality": round(c.quality, 4),
                "components": {k: round(v, 4) for k, v in c.components.items()},
                "alignment": None if c.alignment is None else round(c.alignment, 4),
                "pose": _pose_dict(c.pose),
                "selected": c.selected,
                "grid_slot": c.grid_slot,
                "note": c.note,
            }
            for c in candidates
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_picks(path: Path) -> tuple[Path, list[Candidate]]:
    """从 scores.json 读回被选中的帧，供 ``grid`` 子命令重拼。

    只重建拼图需要的字段（帧号、外框、时间、体式名）。关键点不需要 ——
    裁图和标签都不用它。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    video_path = Path(payload["video"]["path"])

    picks: list[Candidate] = []
    for entry in payload["candidates"]:
        if not entry.get("selected"):
            continue
        pose_data = entry.get("pose")
        pose = (
            PoseMatch(
                key=pose_data["key"],
                zh=pose_data["zh"],
                en=pose_data["en"],
                side=pose_data["side"],
                score=pose_data["score"],
                checks=[],
            )
            if pose_data
            else None
        )
        frame = FramePose(
            idx=-1,
            frame_no=entry["frame_no"],
            t=entry["t"],
            lm=None,
            norm=None,
            bbox=tuple(entry["bbox"]),
            visibility=entry["components"].get("visibility", 0.0),
            sharpness=0.0,
        )
        picks.append(
            Candidate(
                frame=frame,
                segment_id=entry["segment_id"],
                segment_duration=entry["segment_duration"],
                velocity=entry["velocity"],
                components=entry["components"],
                quality=entry["quality"],
                pose=pose,
                cluster=entry["cluster"],
                alignment=entry.get("alignment"),
                selected=True,
                grid_slot=entry.get("grid_slot"),
                note=entry.get("note", ""),
            )
        )

    picks.sort(key=lambda c: (c.grid_slot if c.grid_slot is not None else 0))
    return video_path, picks


def write_report(
    path: Path,
    info: VideoInfo,
    selection: SelectionReport,
    n_candidates: int,
    grid_name: str,
) -> None:
    lines: list[str] = []
    add = lines.append

    add("# 瑜伽正位九宫格 · 复盘")
    add("")
    add(f"- 视频：`{info.path.name}`（{_fmt_time(info.duration)}，{info.width}×{info.height}，{info.fps:.1f} fps）")
    add(f"- 候选帧：{n_candidates} 张，聚出 {selection.n_clusters} 个体式")
    add(f"- 入选：{len(selection.picks)} 张 → `{grid_name}`")
    if selection.n_filled:
        add(
            f"- ⚠️ 只识别到 {selection.n_clusters} 个不同体式，不足 "
            f"{selection.requested} 个，有 {selection.n_filled} 格用同体式的另一次保持补位"
        )
    add("")

    add("## 入选的九张")
    add("")
    add("| # | 时间 | 体式 | 正位分 | 画质分 | 保持 | 备注 |")
    add("|---|------|------|--------|--------|------|------|")
    for cand in selection.picks:
        slot = (cand.grid_slot or 0) + 1
        alignment = "—" if cand.alignment is None else f"{cand.alignment:.2f}"
        add(
            f"| {slot} | {_fmt_time(cand.t)} | {cand.pose_label} | {alignment} "
            f"| {cand.quality:.2f} | {cand.segment_duration:.1f}s | {cand.note or ''} |"
        )
    add("")

    add("## 正位反馈")
    add("")
    add("下面列出每张里偏离目标较多的检查项。角度（`°`）都是在图像平面上量的，")
    add("身体明显斜对镜头时数值会偏小，看趋势比看绝对值有意义。没有单位的数值")
    add("以躯干长度（肩中点到髋中点）为 1，方向类的量则是与铅垂线夹角的余弦：")
    add("`+1` 完全直立，`0` 水平，`-1` 完全倒置。")
    add("")

    for cand in selection.picks:
        slot = (cand.grid_slot or 0) + 1
        add(f"### {slot}. {cand.pose_label} · {_fmt_time(cand.t)}")
        add("")
        if cand.pose is None:
            add("这一帧没有匹配到已知体式模板，只按画质和保持稳定性入选，没有正位评分。")
            add("")
            continue

        weak = cand.pose.weak_checks()
        if not weak:
            add(f"正位分 {cand.pose.score:.2f}，各项检查都在容差内。")
            add("")
            continue

        add(f"正位分 {cand.pose.score:.2f}。偏离较多的项：")
        add("")
        for check in weak:
            if check.value != check.value:  # nan
                continue
            # 角度用整数够了；无单位的量（躯干长度、方向余弦）小数才有信息量。
            digits = 0 if check.unit == "°" else 2
            add(
                f"- **{check.label}**：实测 {check.value:.{digits}f}{check.unit}，"
                f"目标 {check.target:.{digits}f}{check.unit} ± {check.tol:.{digits}f}"
                f"（得分 {check.score:.2f}）"
            )
        add("")

    add("## 想换掉某一格")
    add("")
    add("`candidates/` 里是所有候选帧，`scores.json` 里是它们的分数。改法：")
    add("")
    add("1. 在 `scores.json` 里把不想要的那张的 `selected` 改成 `false`；")
    add("2. 挑一张想要的，把它的 `selected` 改成 `true`，并把 `grid_slot` 设成腾出来的格号；")
    add("3. 跑 `python -m yoga_grid grid <输出目录>` 重拼，不用重跑整条流水线。")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")
