"""命令行入口。

    python -m yoga_grid 练习.mp4                    # 完整流水线
    python -m yoga_grid grid out                    # 改完 scores.json 后只重拼
    python -m yoga_grid rescore out                 # 用当前模板重算识别，跳过姿态估计
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from . import compat, faces, naming, reference, report, rescore
from .cluster import cluster_candidates
from .compare import build_comparison
from .explain import explain
from .extract import extract, iter_frames_at
from .grid import GridStyle, build_grid, find_cjk_font
from .model import resolve_model
from .poses import TEMPLATES
from .score import build_candidates
from .select import select
from .stability import auto_threshold, find_holds, velocities

# 产物文件名的 (前缀, 扩展名)。实际落盘时会插入日期和当天序号，
# 形如 九宫格_20260828_01.jpg —— 一天里多次运行才不会互相覆盖。
GRID_BASE = ("九宫格", ".jpg")
COMPARE_BASE = ("标准对照图", ".jpg")
REPORT_BASE = ("report", ".md")
OUTPUT_BASES = [GRID_BASE, COMPARE_BASE, REPORT_BASE]


def _resize_long_side(frame, long_side: int):
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= long_side:
        return frame
    scale = long_side / longest
    return cv2.resize(
        frame, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _add_grid_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--size", type=int, default=2048, help="输出边长，像素（默认 2048）")
    parser.add_argument("--gap", type=int, default=16, help="格子间隔（默认 16）")
    parser.add_argument("--margin", type=int, default=24, help="外边距（默认 24）")
    parser.add_argument(
        "--pad", type=float, default=1.35, dest="pad_factor",
        help="裁切框相对人体外框的放大倍数（默认 1.35）",
    )
    parser.add_argument(
        "--pad-mode", choices=("blur", "solid", "crop"), default="blur",
        help="裁切框超出画面时如何补：blur 模糊底图 / solid 纯色 / crop 压进画面（会切到身体）",
    )
    parser.add_argument("--no-labels", action="store_true", help="不在格子上写体式名和时间")
    parser.add_argument("--font", default=None, help="中文字体文件路径（默认自动查找）")


def _grid_style(args: argparse.Namespace) -> GridStyle:
    return GridStyle(
        size=args.size,
        gap=args.gap,
        margin=args.margin,
        pad_factor=args.pad_factor,
        pad_mode=args.pad_mode,
        labels=not args.no_labels,
        font_path=args.font,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yoga_grid",
        description="从瑜伽练习视频里抓取正位帧，拼成九宫格。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="完整流水线（默认命令）")
    run.add_argument("video", type=Path, nargs="?", help="视频文件")
    run.add_argument("-o", "--out", type=Path, default=Path("out"), help="输出目录")
    run.add_argument("--interval", type=float, default=0.5, help="抽帧间隔，秒")
    run.add_argument("--work-size", type=int, default=720, help="姿态估计时缩放到的长边像素")
    run.add_argument(
        "--model", default="full", help="姿态模型：lite / full / heavy，或 .task 文件路径"
    )
    run.add_argument("--count", type=int, default=9, help="要挑几张")
    run.add_argument(
        "--vel-thresh", default="auto",
        help="判定「静止」的速度阈值（躯干长度/秒），auto 为自适应",
    )
    run.add_argument("--min-hold", type=float, default=1.0, help="保持段最短时长，秒")
    run.add_argument("--per-segment", type=int, default=2, help="每个保持段最多留几帧候选")
    run.add_argument(
        "--cluster-dist", type=float, default=0.35,
        help="体式聚类距离阈值，越小分得越细（单位：躯干长度）",
    )
    run.add_argument(
        "--mirror-distinct", action="store_true",
        help="把左右版本的同一体式当作两个体式（默认当作同一个）",
    )
    run.add_argument(
        "--no-merge-same-pose", action="store_true",
        help="不合并判为同一体式的多个簇（默认合并，避免九宫格里出现两个下犬式）",
    )
    run.add_argument(
        "--exclude", default="",
        help="排除的体式 key，逗号分隔，例如 mountain",
    )
    run.add_argument("--order", choices=("time", "score"), default="time", help="九宫格排列顺序")
    run.add_argument("--no-candidates", action="store_true", help="不导出候选帧缩略图")
    run.add_argument(
        "--compare", action="store_true",
        help="生成标准体式对照图（骨架线稿版；默认不出）",
    )
    run.add_argument(
        "--no-face-mask", action="store_true",
        help="不给露出的人脸盖卡通面具（默认盖）",
    )
    run.add_argument(
        "--no-landmarks", action="store_true",
        help="scores.json 里不存关键点（省约 0.8 KB/帧，但之后无法离线复算模板、重拼时也无法遮脸）",
    )
    run.add_argument("--list-poses", action="store_true", help="列出内置体式模板后退出")
    _add_grid_options(run)

    grid = sub.add_parser("grid", help="按 scores.json 里的 selected 标记重新拼图")
    grid.add_argument("out", type=Path, help="上次运行的输出目录（含 scores.json）")
    grid.add_argument("--video", type=Path, default=None, help="视频路径（默认取 scores.json 里记录的）")
    grid.add_argument("--no-face-mask", action="store_true", help="不给露出的人脸盖卡通面具")
    grid.add_argument("--compare", action="store_true", help="生成标准体式对照图（骨架线稿版；默认不出）")
    _add_grid_options(grid)

    again = sub.add_parser(
        "rescore",
        help="用当前模板从 scores.json 的骨架重算识别与选帧（跳过姿态估计，秒级）",
    )
    again.add_argument("out", type=Path, help="上次运行的输出目录（含 scores.json）")
    again.add_argument("--video", type=Path, default=None,
                       help="视频路径（默认取 scores.json 里记录的）")
    again.add_argument("--count", type=int, default=9, help="要挑几张")
    again.add_argument("--cluster-dist", type=float, default=0.35,
                       help="体式聚类距离阈值，越小分得越细（单位：躯干长度）")
    again.add_argument("--mirror-distinct", action="store_true",
                       help="把左右版本的同一体式当作两个体式")
    again.add_argument("--no-merge-same-pose", action="store_true",
                       help="不合并判为同一体式的多个簇")
    again.add_argument("--exclude", default="", help="排除的体式 key，逗号分隔")
    again.add_argument("--order", choices=("time", "score"), default="time",
                       help="九宫格排列顺序")
    again.add_argument("--no-face-mask", action="store_true", help="不给露出的人脸盖卡通面具")
    again.add_argument("--compare", action="store_true", help="生成标准体式对照图（骨架线稿版；默认不出）")
    _add_grid_options(again)

    ref = sub.add_parser("reference", help="导出标准体式库（每个体式一张对照卡）")
    ref.add_argument("-o", "--out", type=Path, default=Path("体式库"), help="输出目录")
    ref.add_argument("--width", type=int, default=1180, help="卡片宽度，像素")
    ref.add_argument("--font", default=None, help="中文字体文件路径")

    why = sub.add_parser("explain", help="解释某一帧为什么入选或落选")
    why.add_argument("out", type=Path, help="上次运行的输出目录（含 scores.json）")
    why.add_argument(
        "--at", type=float, default=None, metavar="秒",
        help="要追查的那一帧的时间戳，例如候选文件名 0129.25s_... 就传 129.25",
    )
    why.add_argument("--all", action="store_true", help="列出全部体式簇，不截断")

    return parser


def cmd_run(args: argparse.Namespace) -> int:
    if args.list_poses:
        print("内置体式模板：")
        for template in TEMPLATES:
            kind = "对称" if template.symmetric else "左右"
            print(f"  {template.key:10s} {template.zh:6s} {template.en:24s} ({kind}，门槛 {template.min_score:.2f})")
        return 0

    if args.video is None:
        print("请指定视频文件。用法：python -m yoga_grid 练习.mp4", file=sys.stderr)
        return 2
    if not args.video.is_file():
        print(f"找不到视频：{args.video}", file=sys.stderr)
        return 2

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Windows 上 OpenCV 读不了非 ASCII 路径，必要时先复制到临时 ASCII 路径。
    video, cleanup_video = compat.prepare_video(args.video)
    try:
        return _run_pipeline(args, video, out_dir)
    finally:
        cleanup_video()


def _run_pipeline(args: argparse.Namespace, video: Path, out_dir: Path) -> int:
    started = time.monotonic()
    model_path = resolve_model(args.model)

    print(f"[1/6] 抽帧与姿态估计（每 {args.interval}s 一帧）…", file=sys.stderr)
    info, frames = extract(
        video,
        model_path,
        interval=args.interval,
        work_size=args.work_size,
    )
    # 复制过的话 info.path 会是临时路径；记录用户给的原路径，
    # 否则 scores.json 存下的路径下次 `grid` 子命令就找不到了。
    info.path = args.video
    n_detected = sum(1 for f in frames if f.detected)
    print(
        f"      视频 {info.duration:.0f}s，采样 {len(frames)} 帧，检出人体 {n_detected} 帧",
        file=sys.stderr,
    )
    if n_detected == 0:
        print("没有任何一帧检出人体。检查画面里人是否足够大、是否被严重遮挡。", file=sys.stderr)
        return 1

    print("[2/6] 检测保持段…", file=sys.stderr)
    vel = velocities(frames)
    threshold = (
        auto_threshold(vel) if args.vel_thresh == "auto" else float(args.vel_thresh)
    )
    segments = find_holds(frames, vel, threshold, min_hold=args.min_hold)
    print(
        f"      速度阈值 {threshold:.3f} 躯干/秒，找到 {len(segments)} 个保持段",
        file=sys.stderr,
    )
    if not segments:
        print(
            "没找到保持段。可以放宽条件：--min-hold 0.5，或 --vel-thresh 0.6。",
            file=sys.stderr,
        )
        return 1

    print("[3/6] 候选帧打分…", file=sys.stderr)
    exclude = frozenset(k.strip() for k in args.exclude.split(",") if k.strip())
    candidates = build_candidates(
        frames, segments, vel, threshold,
        per_segment=args.per_segment,
        exclude_poses=exclude,
    )
    recognized = sum(1 for c in candidates if c.pose is not None)
    print(f"      {len(candidates)} 张候选，其中 {recognized} 张识别出体式", file=sys.stderr)
    if not candidates:
        print("保持段里没有可用帧。", file=sys.stderr)
        return 1

    print("[4/6] 体式聚类…", file=sys.stderr)
    n_clusters = cluster_candidates(
        candidates, threshold=args.cluster_dist, mirror_same=not args.mirror_distinct
    )
    print(f"      聚出 {n_clusters} 个体式", file=sys.stderr)

    selection = select(
        candidates,
        count=args.count,
        order=args.order,
        merge_same_pose=not args.no_merge_same_pose,
    )
    print(f"      合并同体式后 {selection.n_clusters} 个体式", file=sys.stderr)
    if selection.n_filled:
        print(
            f"      ⚠️ 体式种类不足 {args.count} 个，{selection.n_filled} 格用同体式的另一次保持补位",
            file=sys.stderr,
        )
    if len(selection.picks) < args.count:
        print(
            f"      ⚠️ 候选帧只够 {len(selection.picks)} 张（要 {args.count} 张），"
            f"格子会空出来。可以 --interval 0.3 抽密一点，或 --min-hold 0.5 放宽保持时长",
            file=sys.stderr,
        )

    print("[5/6] 回读原分辨率帧…", file=sys.stderr)
    pick_frame_nos = {c.frame.frame_no for c in selection.picks}
    want = (
        sorted({c.frame.frame_no for c in candidates})
        if not args.no_candidates
        else sorted(pick_frame_nos)
    )
    cand_by_frame = {c.frame.frame_no: c for c in candidates}
    cand_dir = out_dir / "candidates"
    if not args.no_candidates:
        cand_dir.mkdir(parents=True, exist_ok=True)

    raw_picks = {}
    n_masked = 0
    for frame_no, bgr in iter_frames_at(video, want):
        # 先在整帧上遮脸，再派给缩略图和拼图 —— 裁剪会换坐标系，在整帧上做一次
        # 就不会有哪条产出漏掉。
        if not args.no_face_mask:
            n_masked += faces.mask_face(bgr, cand_by_frame[frame_no].frame.lm)
        if frame_no in pick_frame_nos:
            raw_picks[frame_no] = bgr.copy()
        if not args.no_candidates:
            cand = cand_by_frame[frame_no]
            key = cand.pose.key if cand.pose else "unknown"
            name = (
                f"{cand.t:07.2f}s_c{cand.cluster:02d}_{key}"
                f"_q{cand.quality:.2f}{'_PICK' if cand.selected else ''}.jpg"
            )
            compat.imwrite(
                cand_dir / name,
                _resize_long_side(bgr, 640),
                [cv2.IMWRITE_JPEG_QUALITY, 88],
            )

    if not args.no_face_mask:
        print(f"      {n_masked}/{len(want)} 帧检测到人脸并已遮挡", file=sys.stderr)

    print("[6/6] 合成九宫格…", file=sys.stderr)
    style = _grid_style(args)
    if style.labels and args.font is None and find_cjk_font() is None:
        print("      找不到中文字体，标签改用英文体式名（可用 --font 指定）", file=sys.stderr)

    # 同一次运行的产物共用一个序号，事后才对得上是哪次跑的。
    seq = naming.run_sequence(out_dir, OUTPUT_BASES)
    image = build_grid(raw_picks, selection.picks, style, frames_dir=out_dir / "frames")
    grid_path = naming.stamped_path(out_dir, *GRID_BASE, sequence=seq)
    image.save(grid_path, quality=94, subsampling=1)

    compare_path = None
    if args.compare:
        compare_path = naming.stamped_path(out_dir, *COMPARE_BASE, sequence=seq)
        build_comparison(raw_picks, selection.picks, style, font_path=args.font).save(
            compare_path, quality=92, subsampling=1
        )

    params = {
        "interval": args.interval,
        "work_size": args.work_size,
        "model": args.model,
        "vel_threshold": round(threshold, 4),
        "min_hold": args.min_hold,
        "per_segment": args.per_segment,
        "cluster_dist": args.cluster_dist,
        "mirror_distinct": args.mirror_distinct,
        "exclude": sorted(exclude),
        "count": args.count,
        "order": args.order,
        "sampled_frames": len(frames),
    }
    report.dump_json(
        out_dir / "scores.json", info, candidates, selection, params,
        n_detected=n_detected, n_segments=len(segments),
        include_landmarks=not args.no_landmarks,
    )
    report_path = naming.stamped_path(out_dir, *REPORT_BASE, sequence=seq)
    report.write_report(
        report_path, info, selection,
        n_candidates=len(candidates), grid_name=grid_path.name,
    )

    elapsed = time.monotonic() - started
    print(f"\n完成，用时 {elapsed:.0f}s", file=sys.stderr)
    print(f"  九宫格   {grid_path}", file=sys.stderr)
    if compare_path is not None:
        print(f"  对照图   {compare_path}", file=sys.stderr)
    print(f"  单张     {out_dir / 'frames'}", file=sys.stderr)
    if not args.no_candidates:
        print(f"  候选帧   {cand_dir}", file=sys.stderr)
    print(f"  复盘     {report_path}", file=sys.stderr)
    print(f"  分数     {out_dir / 'scores.json'}", file=sys.stderr)
    return 0


def cmd_grid(args: argparse.Namespace) -> int:
    scores = args.out / "scores.json"
    if not scores.is_file():
        print(f"找不到 {scores}", file=sys.stderr)
        return 2

    recorded_video, picks = report.load_picks(scores)
    video = args.video or recorded_video
    if not video.is_file():
        print(f"找不到视频：{video}（可用 --video 指定）", file=sys.stderr)
        return 2
    if not picks:
        print("scores.json 里没有 selected 为 true 的帧。", file=sys.stderr)
        return 1

    style = _grid_style(args)
    usable, cleanup_video = compat.prepare_video(video)
    by_frame = {c.frame.frame_no: c for c in picks}
    try:
        raw = {}
        for frame_no, bgr in iter_frames_at(usable, list(by_frame)):
            if not args.no_face_mask:
                faces.mask_face(bgr, by_frame[frame_no].frame.lm)
            raw[frame_no] = bgr
    finally:
        cleanup_video()
    seq = naming.run_sequence(args.out, OUTPUT_BASES)
    image = build_grid(raw, picks, style, frames_dir=args.out / "frames")
    grid_path = naming.stamped_path(args.out, *GRID_BASE, sequence=seq)
    image.save(grid_path, quality=94, subsampling=1)
    print(f"已用 {len(picks)} 张重拼：{grid_path}", file=sys.stderr)
    if args.compare:
        compare_path = naming.stamped_path(args.out, *COMPARE_BASE, sequence=seq)
        build_comparison(raw, picks, style, font_path=args.font).save(
            compare_path, quality=92, subsampling=1
        )
        print(f"对照图：{compare_path}", file=sys.stderr)
    return 0


def cmd_rescore(args: argparse.Namespace) -> int:
    scores = args.out / "scores.json"
    if not scores.is_file():
        print(f"找不到 {scores}", file=sys.stderr)
        return 2

    recorded_video, candidates, payload = report.load_candidates(scores)
    video = args.video or recorded_video
    if not video.is_file():
        print(f"找不到视频：{video}（可用 --video 指定）", file=sys.stderr)
        return 2

    candidates, dropped = rescore.usable(candidates)
    if rescore.warn_if_no_landmarks(dropped, dropped + len(candidates)):
        return 1
    if not candidates:
        print("scores.json 里没有可用候选。", file=sys.stderr)
        return 1

    started = time.monotonic()
    exclude = frozenset(k.strip() for k in args.exclude.split(",") if k.strip())

    print(f"[1/4] 用当前模板重新识别 {len(candidates)} 个候选…", file=sys.stderr)
    recognized = rescore.rematch(candidates, exclude)
    print(f"      识别出体式 {recognized}/{len(candidates)}", file=sys.stderr)

    print("[2/4] 体式聚类与选帧…", file=sys.stderr)
    n_clusters = cluster_candidates(
        candidates, threshold=args.cluster_dist, mirror_same=not args.mirror_distinct
    )
    selection = select(
        candidates, count=args.count, order=args.order,
        merge_same_pose=not args.no_merge_same_pose,
    )
    print(
        f"      聚出 {n_clusters} 个体式，合并同体式后 {selection.n_clusters} 个，"
        f"入选 {len(selection.picks)} 张",
        file=sys.stderr,
    )
    if selection.n_filled:
        print(f"      ⚠️ {selection.n_filled} 格用同体式的另一次保持补位", file=sys.stderr)
    if len(selection.picks) < args.count:
        print(
            f"      ⚠️ 候选只够 {len(selection.picks)} 张（要 {args.count} 张）；"
            "要更多候选得重跑完整流水线并放宽 --interval / --min-hold",
            file=sys.stderr,
        )

    print("[3/4] 回读入选帧的原分辨率像素…", file=sys.stderr)
    usable_video, cleanup_video = compat.prepare_video(video)
    by_frame = {c.frame.frame_no: c for c in selection.picks}
    n_masked = 0
    try:
        raw = {}
        for frame_no, bgr in iter_frames_at(usable_video, list(by_frame)):
            if not args.no_face_mask:
                n_masked += faces.mask_face(bgr, by_frame[frame_no].frame.lm)
            raw[frame_no] = bgr
    finally:
        cleanup_video()
    if not args.no_face_mask:
        print(f"      {n_masked}/{len(by_frame)} 帧检测到人脸并已遮挡", file=sys.stderr)

    print(
        "[4/4] 重出九宫格" + ("与对照图…" if args.compare else "…"),
        file=sys.stderr,
    )
    style = _grid_style(args)
    seq = naming.run_sequence(args.out, OUTPUT_BASES)
    image = build_grid(raw, selection.picks, style, frames_dir=args.out / "frames")
    grid_path = naming.stamped_path(args.out, *GRID_BASE, sequence=seq)
    image.save(grid_path, quality=94, subsampling=1)

    compare_path = None
    if args.compare:
        compare_path = naming.stamped_path(args.out, *COMPARE_BASE, sequence=seq)
        build_comparison(raw, selection.picks, style, font_path=args.font).save(
            compare_path, quality=92, subsampling=1
        )

    info = rescore.video_info_from_payload(payload, video)
    params = dict(payload.get("params") or {})
    params.update({
        "count": args.count,
        "cluster_dist": args.cluster_dist,
        "mirror_distinct": args.mirror_distinct,
        "exclude": sorted(exclude),
        "order": args.order,
        "rescored": True,
    })
    report.dump_json(
        scores, info, candidates, selection, params,
        n_detected=int((payload.get("summary") or {}).get("detected_frames") or 0),
        n_segments=int((payload.get("summary") or {}).get("hold_segments") or 0),
    )
    report_path = naming.stamped_path(args.out, *REPORT_BASE, sequence=seq)
    report.write_report(
        report_path, info, selection,
        n_candidates=len(candidates), grid_name=grid_path.name,
    )

    print(f"\n完成，用时 {time.monotonic() - started:.1f}s（未重跑姿态估计）", file=sys.stderr)
    print(f"  九宫格   {grid_path}", file=sys.stderr)
    if compare_path is not None:
        print(f"  对照图   {compare_path}", file=sys.stderr)
    print(f"  复盘     {report_path}", file=sys.stderr)
    print(f"  分数     {scores}", file=sys.stderr)
    return 0


def cmd_reference(args: argparse.Namespace) -> int:
    from .poses import TEMPLATES

    args.out.mkdir(parents=True, exist_ok=True)
    for template in TEMPLATES:
        card = reference.render_reference_card(
            template.key, width=args.width, font_path=args.font
        )
        card.save(args.out / f"{template.key}_{template.zh}.png")
    print(f"已导出 {len(TEMPLATES)} 张体式卡到 {args.out}", file=sys.stderr)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    scores = args.out / "scores.json"
    if not scores.is_file():
        print(f"找不到 {scores}", file=sys.stderr)
        return 2
    print(explain(scores, focus=args.at, max_clusters=0 if args.all else 18))
    return 0


def main(argv: list[str] | None = None) -> int:
    compat.configure_console()
    argv = list(sys.argv[1:] if argv is None else argv)
    # 让 `python -m yoga_grid 视频.mp4` 免写 run 子命令。
    known = {"run", "grid", "rescore", "explain", "reference", "-h", "--help"}
    if argv and argv[0] not in known:
        argv.insert(0, "run")

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "rescore":
        return cmd_rescore(args)
    if args.command == "reference":
        return cmd_reference(args)
    if args.command == "explain":
        return cmd_explain(args)
    if args.command == "grid":
        return cmd_grid(args)
    if args.command == "run":
        return cmd_run(args)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
