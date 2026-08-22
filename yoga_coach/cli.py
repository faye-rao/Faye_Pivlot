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
    parser.add_argument(
        "--speak",
        action="store_true",
        help="语音播报：报体式名、到位提示、纠正建议、完成一组（需要 pyttsx3）",
    )
    parser.add_argument(
        "--speak-test",
        action="store_true",
        help="不开摄像头，单独测试语音：列出系统音色并试播几句后退出",
    )
    parser.add_argument(
        "--speak-rate",
        type=int,
        default=165,
        help="语音语速，词/分钟（默认 165，越小越慢）",
    )
    parser.add_argument("--record", metavar="PATH", help="把带标注的画面保存到视频/图片文件")
    parser.add_argument(
        "--log",
        metavar="PATH.csv",
        help="把每项检查的实测值逐条记到 CSV（练完不用手记，直接看文件）",
    )
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

    if args.speak_test:
        return speak_test(args)

    pose = None if args.pose == "auto" else _lookup_pose(args.pose)
    if pose is False:  # unknown key; _lookup_pose already reported it
        return 2

    source = _resolve_source(args.source)
    if isinstance(source, str) and Path(source).suffix.lower() in IMAGE_SUFFIXES:
        return run_image(args, pose)
    return run_stream(args, pose, source)


def speak_test(args) -> int:
    """Exercise speech on its own, so it can be diagnosed off the mat.

    Speaks several cues in a row on purpose: the failure this exists to catch
    was an engine that managed exactly one utterance and then went quiet.
    """
    import time as _time

    from .checks import Text
    from .voice import Speaker, describe_voices

    voices = describe_voices()
    if voices:
        print("系统音色：")
        print("\n".join(voices))
    else:
        print("没有列出任何音色（pyttsx3 未安装，或驱动不可用）。")
    print()

    speaker = Speaker(lang=args.lang, rate=args.speak_rate)
    if not speaker.enabled:
        print("语音不可用，上面的报错说明了原因。", file=sys.stderr)
        return 1

    print(f"实际播报语言：{speaker.lang}")
    lines = [
        Text("语音测试，第一句", "Voice test, line one"),
        Text("前膝移回脚踝正上方", "Bring the front knee over the ankle"),
        Text("到位了，保持住", "That's it, hold there"),
        Text("完成一组，保持了 20 秒", "Round complete, held 20 seconds"),
    ]
    for index, line in enumerate(lines, start=1):
        print(f"  {index}. {line.get(speaker.lang)}")
        # force=True bypasses the rate limit; each cue then waits for the
        # engine to finish, which is what proves the thread survives.
        speaker.say(line, index * 100.0, force=True)
        for _ in range(100):
            if speaker._queue.empty():
                break
            _time.sleep(0.1)
        if not speaker.enabled:
            print(f"\n第 {index} 句之后语音就停了——这正是要排查的故障。", file=sys.stderr)
            speaker.close()
            return 1
    _time.sleep(1.0)
    speaker.close()
    print("\n四句都播完了，语音正常。")
    return 0


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
    from .logbook import Logbook
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
    logbook = Logbook()
    overlay = None
    # Needed for the window, and also when recording headlessly: a saved clip
    # without the annotations would be no more use than the original footage.
    if not headless or args.record:
        from .render import Overlay

        overlay = Overlay(lang=args.lang, font_path=args.font)

    speaker = None
    announcer = None
    if args.speak:
        from .announce import Announcer
        from .checks import Text
        from .voice import Speaker

        speaker = Speaker(lang=args.lang, rate=args.speak_rate)
        announcer = Announcer()
        if speaker.enabled:
            # Say something before the practice starts, so you find out the
            # audio works while you are still looking at the screen.
            opening = (
                Text("教练已就绪，自动识别体式", "Coach ready, detecting poses")
                if pose is None
                else Text(
                    f"教练已就绪，{pose.name.zh}。{pose.view.zh}",
                    f"Coach ready. {pose.name.en}. {pose.view.en}",
                )
            )
            speaker.say(opening, 0.0, force=True)

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

                logbook.record(state, now, collect_rows=bool(args.log))

                if announcer is not None and speaker is not None:
                    cue = announcer.update(state, now)
                    if cue is not None and not speaker.say(
                        cue.text, now, force=cue.force
                    ):
                        # Engine was busy and dropped it; let the announcer
                        # offer the cue again instead of counting it as said.
                        announcer.undo()

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
                        if announcer is not None:
                            announcer.reset()
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
    for line in logbook.summary_lines():
        print(line)
    if args.log:
        logbook.write_csv(args.log)
        print(f"\n逐帧记录已保存到 {args.log}")
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
