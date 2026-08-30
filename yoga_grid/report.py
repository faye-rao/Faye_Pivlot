"""落盘产物：scores.json（可回改重拼）与 report.md（正位反馈）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import landmarks as L
from .extract import FramePose, VideoInfo
from .poses import MIN_TRUSTED_VISIBILITY, PoseMatch
from .score import Candidate
from .select import SelectionReport


def _add_occlusion_note(add, occluded: list, weak: list) -> None:
    """把「这一帧有哪些项读到了被遮挡的关键点」写出来。

    为什么值得单独说：``normalize()`` 丢掉了置信度，规则层看不见哪些关键点是
    MediaPipe 猜的，而侧面机位下远侧那半边身体每一帧都被挡住（实测某支视频
    右膝置信度中位数 0.315、80% 的帧低于 0.5）。猜出来的数值看着完全合理，
    所以**沉默地报一个数字，等于断言自己看不见的事**。

    刻意只提示、不改分数：实测把低置信度的一侧排除掉会改变 0 个识别结果，
    而且有反例 —— 用户确认正确的那帧侧角伸展式，被遮挡的前腿膝角读 113°
    是**对的**，排除它等于丢掉一个正确测量。所以这里只负责说清楚。
    """
    if not occluded:
        return
    names = "、".join(c.label for c in occluded)
    already = {id(c) for c in weak}
    hidden_but_passing = [c for c in occluded if id(c) not in already]
    add(
        f"⚠️ 有 {len(occluded)} 项用到了被遮挡的关键点（{names}）—— "
        f"机位在身体侧面时远侧的手脚会被近侧挡住，MediaPipe 给的是推测值。"
    )
    if hidden_but_passing:
        add("")
        add(
            f"其中 {len(hidden_but_passing)} 项**看着达标其实没测准**："
            + "、".join(f"{c.label}（置信度 {c.confidence:.2f}）" for c in hidden_but_passing)
            + "。想让这些项可信，机位要能同时看到两侧，或者换个方向再录一遍。"
        )
    add("")


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
        # gate 和 orientation 一样是乘在总分上的系数，必须一起存下来：
        # 少了它，一个被定义性门槛压到 0.2 的分数在文件里看起来和「各项都
        # 做得差」一模一样，而两者该做的事完全不同 —— 前者说明这一帧根本
        # 不是这个体式，后者才是纠正建议。
        "gate": round(pose.gate, 4),
        "checks": [
            {
                "label": c.label,
                "value": None if c.value != c.value else round(c.value, 2),  # nan 安全
                "target": c.target,
                "tol": c.tol,
                "unit": c.unit,
                "score": round(c.score, 4),
                # 这一项用到的关键点里最低的置信度。侧面机位下远侧半身是
                # MediaPipe 猜的，而猜出来的数值看着完全合理 —— 只有这一列
                # 能看出某个实测值有多少是真的。
                "confidence": round(c.confidence, 3),
            }
            for c in pose.checks
        ],
    }


def _landmarks_field(cand: Candidate, include: bool) -> dict[str, Any]:
    """写回 landmarks，优先原始关键点（33×3），退而写归一化骨架（33×2）。

    退路是为了 ``rescore``：从早期格式的文件重算后再落盘时，若只认原始关键点就
    会把 landmarks 整个丢掉，这个文件之后就再也不能重算了 —— 一次重算把重算能力
    自己弄没了。两种形状读取端都认（见 ``_restore_landmarks``），所以有什么写什么。
    """
    if not include:
        return {}
    source = cand.frame.lm if cand.frame.lm is not None else cand.frame.norm
    if source is None:
        return {}
    return {"landmarks": [[round(float(v), 4) for v in pt] for pt in source]}


def dump_json(
    path: Path,
    info: VideoInfo,
    candidates: list[Candidate],
    selection: SelectionReport,
    params: dict[str, Any],
    n_detected: int,
    n_segments: int,
    include_landmarks: bool = True,
) -> None:
    """落盘 scores.json。

    ``include_landmarks`` 会额外存下每个候选帧的原始关键点（33×3：归一化 x、y
    和置信度，约 0.8 KB/帧），两个用途：

    * 离线复算体式模板 —— 没有它，想调一项容差就得重跑整支视频，而调容差恰恰
      需要反复试；
    * 重拼时给人脸打面具 —— 面具要画在原始画面坐标里。

    存原始关键点而不是归一化骨架：归一化骨架能从原始点算回来，反过来不行
    （平移和尺度信息已经被消掉了）。存前者，一个字段就够。
    """
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
                **_landmarks_field(c, include_landmarks),
            }
            for c in candidates
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _restore_landmarks(
    landmarks: Any, frame_w: int, frame_h: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """把 scores.json 里的 landmarks 还原成 (原始关键点, 归一化骨架)。

    支持两种历史格式，靠列数区分：

    * **33×3**（当前）—— 原始关键点：归一化 x、y 加置信度。归一化骨架由它算出，
      算之前要先乘回画面宽高，否则画面长宽比会把角度拉歪。
    * **33×2**（早期）—— 已经归一化过的骨架。它直接就是 ``norm``，不能再乘一次
      宽高（那会把本已消掉平移和尺度的坐标彻底算错，而且是静默的）。这种文件
      没有置信度，所以拿不到人脸位置，遮脸只能跳过。
    """
    if not landmarks:
        return None, None

    array = np.asarray(landmarks, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != L.N_LANDMARKS:
        return None, None

    if array.shape[1] >= 3:
        lm = array
        norm = L.normalize(L.to_pixels(lm, frame_w or 1000, frame_h or 1000))
        return lm, norm

    # 早期格式：已是归一化骨架，直接用；没有置信度，lm 留空。
    if not _looks_normalised(array):
        # 误判会静默产出垃圾（所有姿势挤成一簇），宁可拒收。
        return None, None
    return None, array


def _looks_normalised(array: np.ndarray) -> bool:
    """检查一个 33×2 数组是否真的是躯干归一化骨架。

    归一化骨架有两个决定性特征：髋中点落在原点，肩中点到髋中点的距离约等于 1
    （因为那个距离**就是**归一化用的单位）。而一份被误当成骨架的原始归一化坐标
    数值全在 [0,1] 里、髋中点远离原点，两条都不满足。

    值得单独校验：列数虽然能区分本项目写过的两种格式，但一旦判错，后果是所有
    姿势的距离都塌到极小、聚成一簇，结果全错却不报错 —— 这种失败最难查。
    """
    hip = (array[L.IDX["left_hip"]] + array[L.IDX["right_hip"]]) / 2.0
    shoulder = (array[L.IDX["left_shoulder"]] + array[L.IDX["right_shoulder"]]) / 2.0
    if float(np.linalg.norm(hip)) > 0.25:
        return False
    torso = float(np.linalg.norm(shoulder - hip))
    return 0.5 <= torso <= 2.0


def load_candidates(
    path: Path, selected_only: bool = False
) -> tuple[Path, list[Candidate], dict[str, Any]]:
    """从 scores.json 读回候选帧。

    重建下游要用的字段（帧号、外框、时间、体式名、各项分数），以及原始关键点
    和由它算出的归一化骨架（若当初存了）—— 前者供人脸遮挡用，后者供离线复算
    体式模板用。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    video_path = Path(payload["video"]["path"])
    frame_w = int(payload["video"].get("width") or 0)
    frame_h = int(payload["video"].get("height") or 0)

    picks: list[Candidate] = []
    for entry in payload["candidates"]:
        if selected_only and not entry.get("selected"):
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
        lm, norm = _restore_landmarks(entry.get("landmarks"), frame_w, frame_h)
        frame = FramePose(
            idx=-1,
            frame_no=entry["frame_no"],
            t=entry["t"],
            lm=lm,
            norm=norm,
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
                selected=bool(entry.get("selected")),
                grid_slot=entry.get("grid_slot"),
                note=entry.get("note", ""),
            )
        )

    return video_path, picks, payload


def load_picks(path: Path) -> tuple[Path, list[Candidate]]:
    """从 scores.json 读回**被选中的**帧，按格号排序，供 ``grid`` 子命令重拼。"""
    video_path, picks, _ = load_candidates(path, selected_only=True)
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

        # 门槛把分数压下来时先说这件事：否则「正位分 0.20」加一串检查项，
        # 看图的人会以为是自己做得差，而实际结论是「这一帧不是这个体式」。
        if cand.pose.gate < 0.999 or cand.pose.orientation < 0.999:
            factor = cand.pose.gate * cand.pose.orientation
            add(
                f"正位分 {cand.pose.score:.2f}，其中定义性门槛把分数乘掉了 "
                f"{1 - factor:.0%} —— 这一帧可能**不是**这个体式，"
                f"下面的偏离项未必是你要改的地方。"
            )
            add("")

        weak = cand.pose.weak_checks()
        occluded = [
            c for c in cand.pose.checks if c.confidence < MIN_TRUSTED_VISIBILITY
        ]

        if not weak:
            add(f"正位分 {cand.pose.score:.2f}，各项检查都在容差内。")
            add("")
            _add_occlusion_note(add, occluded, weak)
            continue

        add(f"正位分 {cand.pose.score:.2f}。偏离较多的项：")
        add("")
        for check in weak:
            if check.value != check.value:  # nan
                continue
            # 角度用整数够了；无单位的量（躯干长度、方向余弦）小数才有信息量。
            digits = 0 if check.unit == "°" else 2
            # 依赖被遮挡关键点的项，数值后面直接标出来 —— 让人别照着它调姿势。
            flag = (
                f"，**置信度仅 {check.confidence:.2f}，这个数不可信**"
                if check.confidence < MIN_TRUSTED_VISIBILITY
                else ""
            )
            add(
                f"- **{check.label}**：实测 {check.value:.{digits}f}{check.unit}，"
                f"目标 {check.target:.{digits}f}{check.unit} ± {check.tol:.{digits}f}"
                f"（得分 {check.score:.2f}{flag}）"
            )
        add("")
        _add_occlusion_note(add, occluded, weak)

    add("## 想换掉某一格")
    add("")
    add("`candidates/` 里是所有候选帧，`scores.json` 里是它们的分数。改法：")
    add("")
    add("1. 在 `scores.json` 里把不想要的那张的 `selected` 改成 `false`；")
    add("2. 挑一张想要的，把它的 `selected` 改成 `true`，并把 `grid_slot` 设成腾出来的格号；")
    add("3. 跑 `python -m yoga_grid grid <输出目录>` 重拼，不用重跑整条流水线。")
    add("")

    path.write_text("\n".join(lines), encoding="utf-8")
