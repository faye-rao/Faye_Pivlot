"""Command line entry point: ``python -m yoga_coach``."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .poses import POSES, get_pose

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="yoga-coach",
        description="用摄像头看你的瑜伽体式并给出纠正建议 / Real-time yoga posture feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例 / examples:\n"
            "  yoga-coach                        # 打开默认摄像头，自动识别体式\n"
            "  yoga-coach --pose warrior2        # 只练战士二式\n"
            "  yoga-coach --source practice.mp4  # 分析一段录像\n"
            "  yoga-coach --source photo.jpg     # 分析一张照片并打印报告\n"
        ),
    )
    parser.add_argument(
        "--source",
        default="0",
        help="摄像头编号、视频文件或图片路径（默认 0，即第一个摄像头）",
    )
    parser.add_argument(
        "--pose",
        default="auto",
        help="要练习的体式 key，或 auto 让程序自动判断（默认 auto）",
    )
    parser.add_argument("--list-poses", action="store_true", help="列出支持的体式后退出")
    parser.add_argument(
        "--model",
        default="lite",
        choices=("lite", "full", "heavy"),
        help="姿态检测模型：lite 最快，heavy 最准（默认 lite）",
    )
    parser.add_argument("--lang", default="zh", choices=("zh", "en"), help="界面语言")
    parser.add_argument("--font", default=None, help="中文字体文件路径（找不到中文字体时用）")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不要左右镜像画面（摄像头默认镜像，像照镜子一样）",
    )
    parser.add_argument("--headless", action="store_true", help="不开窗口，只在终端输出建议")
    parser.add_argument("--details", action="store_true", help="启动时就显示每项检查的具体数值")
    parser.add_argument("--speak", action="store_true", help="用语音朗读纠正建议（需要 pyttsx3）")
    parser.add_argument("--record", metavar="PATH", help="把带标注的画面保存到视频/图片文件")
    parser.add_argument("--width", type=int, default=1280, help="摄像头采集宽度")
    parser.add_argument("--height", type=int, default=720, help="摄像头采集高度")
    parser.add_argument(
        "--hold-target",
        type=float,
        default=20.0,
        help="保持多少秒算完成一组（默认 20）",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.5,
        help="关键点可信度低于该值时视为看不见（0-1，默认 0.5）",
    )
    return parser


def list_poses(lang: str = "zh") -> None:
    width = max(len(p.key) for p in POSES) + 2
    for pose in POSES:
        name = pose.name.get(lang)
        print(f"{pose.key:<{width}}{name}  ({pose.sanskrit})")
        print(f"{'':<{width}}{pose.view.get(lang)}")


def _resolve_source(source: str) -> int | str:
    if source.isdigit():
        return int(source)
    return source


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_poses:
        list_poses(args.lang)
        return 0

    pose = None if args.pose == "auto" else _lookup_pose(args.pose)
    if pose is False:  # unknown key; _lookup_pose already reported it
        return 2

    source = _resolve_source(args.source)
    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
        return run_image(args, pose)
    return run_stream(args, pose, source)


def _lookup_pose(key: str):
    try:
        return get_pose(key)
    except KeyError as exc:
        print(exc.args[0], file=sys.stderr)
        print("用 --list-poses 查看全部体式。", file=sys.stderr)
        return False


def run_image(args, pose) -> int:
    """Analyse a single photo and print a full report."""
    import cv2

    from .console import report
    from .detector import PoseDetector
    from .evaluator import evaluate, rank_poses

    path = args.source
    frame = cv2.imread(path)
    if frame is None:
        print(f"读不到图片：{path}", file=sys.stderr)
        return 1

    with PoseDetector(
        args.model, video_mode=False, min_visibility=args.min_visibility
    ) as detector:
        skeleton = detector.detect(frame)

    if skeleton is None:
        print("图片里没有检测到人。", file=sys.stderr)
        return 1

    result = evaluate(skeleton, pose) if pose else rank_poses(skeleton)[0]
    report(result, lang=args.lang)

    if args.record:
        from .checks import Text
        from .render import Overlay
        from .session import SessionState

        state = SessionState(
            result=result,
            score=result.score,
            corrections=result.corrections(),
            hold_seconds=0.0,
            best_hold=0.0,
            holds_completed=0,
            notice=None
            if result.confident
            else Text(
                "身体没有完整入镜，评分仅供参考",
                "Body only partly in frame -- treat the score loosely",
            ),
        )
        overlay = Overlay(lang=args.lang, font_path=args.font)
        annotated = overlay.draw(frame, state, skeleton, show_details=args.details)
        cv2.imwrite(args.record, annotated)
        print(f"\n标注图已保存到 {args.record}")
    return 0


def run_stream(args, pose, source) -> int:
    """Run the live loop over a camera or a video file."""
    import cv2

    from .console import ConsoleReporter
    from .detector import PoseDetector
    from .session import CoachSession

    capture = cv2.VideoCapture(source)
    is_camera = isinstance(source, int)
    if is_camera:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not capture.isOpened():
        target = f"摄像头 {source}" if is_camera else f"文件 {source}"
        print(f"打不开{target}。请检查设备编号、路径和摄像头权限。", file=sys.stderr)
        return 1

    mirror = is_camera and not args.no_mirror
    headless = args.headless or not _can_open_window()
    if args.headless is False and headless:
        print("当前环境没有图形界面，自动切换到终端模式。", file=sys.stderr)

    session = CoachSession(pose=pose, hold_target=args.hold_target)
    reporter = ConsoleReporter(lang=args.lang)
    overlay = None
    # Needed for the window, and also when recording headlessly: a saved clip
    # without the annotations would be no more use than the original footage.
    if not headless or args.record:
        from .render import Overlay

        overlay = Overlay(lang=args.lang, font_path=args.font)

    speaker = None
    if args.speak:
        from .voice import Speaker

        speaker = Speaker()

    writer = None
    show_details = args.details
    window = "Yoga Coach"
    start = time.monotonic()

    try:
        with PoseDetector(
            args.model, video_mode=True, min_visibility=args.min_visibility
        ) as detector:
            if not headless:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                if is_camera:
                    now = time.monotonic() - start
                else:
                    # Use the file's own clock so the hold timer is right even
                    # when the frames are decoded faster than real time.
                    position = capture.get(cv2.CAP_PROP_POS_MSEC)
                    now = position / 1000.0 if position > 0 else time.monotonic() - start

                skeleton = detector.detect(frame, int(now * 1000))
                state = session.update(skeleton, now)

                if speaker is not None and state.corrections:
                    advice = state.corrections[0].advice()
                    if advice is not None:
                        speaker.say(advice.get(args.lang), now)

                canvas = frame
                if overlay is not None:
                    canvas = overlay.draw(frame, state, skeleton, show_details=show_details)
                if headless:
                    reporter.update(state, now)
                else:
                    cv2.imshow(window, canvas)

                if args.record:
                    writer = writer or _open_writer(cv2, args.record, canvas, capture)
                    writer.write(canvas)

                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    if key == ord("d"):
                        show_details = not show_details
                    if key == ord("r"):
                        session.reset()
    except KeyboardInterrupt:
        pass
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if speaker is not None:
            speaker.close()
        if not headless:
            cv2.destroyAllWindows()

    print(
        f"\n本次练习：最长保持 {session.best_hold:.0f} 秒，"
        f"完成 {session.holds_completed} 组。"
    )
    if args.record:
        print(f"录像已保存到 {args.record}")
    return 0


def _open_writer(cv2, path: str, frame, capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1 or fps > 120:
        fps = 25.0
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def _can_open_window() -> bool:
    """Guess whether a GUI is available, so headless servers degrade quietly."""
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


if __name__ == "__main__":
    raise SystemExit(main())
