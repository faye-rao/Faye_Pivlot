#!/usr/bin/env python3
"""End-to-end smoke test against the real MediaPipe stack.

The unit tests deliberately never touch MediaPipe -- that is what makes them
fast and camera-free.  The gap that leaves is real: a MediaPipe release that
renames an API, a wheel that will not load its native libraries, or a moved
model URL breaks the program without failing a single test.

This script closes that gap by exercising the parts the unit tests cannot:
download the model, build a detector, run inference, and push a frame all the
way through evaluation and rendering.  It needs the full runtime dependencies
(``requirements.txt``) and, on Linux, the system OpenGL libraries.

Run it locally the same way CI does::

    python tools/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def check(label: str, condition: bool, detail: str = "") -> bool:
    mark = "ok  " if condition else "FAIL"
    print(f"[{mark}] {label}{'  ' + detail if detail else ''}", flush=True)
    return condition


def main() -> int:
    import cv2

    from yoga_coach import evaluate, get_pose
    from yoga_coach.detector import PoseDetector, ensure_model
    from yoga_coach.render import Overlay
    from yoga_coach.session import CoachSession

    ok = True
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    model = ensure_model("lite")
    ok &= check(
        "模型下载与缓存",
        model.exists() and model.stat().st_size > 1_000_000,
        f"{model} ({model.stat().st_size // 1024} KB)",
    )

    # Still-image mode.
    with PoseDetector("lite", video_mode=False) as detector:
        skeleton = detector.detect(frame)
    ok &= check("图片模式推理", skeleton is None, "空白画面应返回 None")

    # Video mode, including the repeated-timestamp path that a jittery webcam
    # clock produces.
    with PoseDetector("lite", video_mode=True) as detector:
        for i in range(5):
            detector.detect(frame, i * 33)
        repeated = detector.detect(frame, 0)
    ok &= check("视频模式推理", repeated is None)
    ok &= check("重复时间戳不抛异常", True)

    # The rules and rendering layers, driven by a synthetic skeleton so this
    # part does not depend on there being a person in the frame.
    sys.path.insert(0, str(ROOT / "tests"))
    from figures import WARRIOR_II_RIGHT, figure  # noqa: E402

    posture = figure(WARRIOR_II_RIGHT)
    result = evaluate(posture, get_pose("warrior2"))
    ok &= check(
        "规则层评分", result.score > 95 and result.side == "right", f"{result.score:.0f} 分"
    )

    session = CoachSession(pose=get_pose("warrior2"))
    session.update(posture, 0.0)
    state = session.update(posture, 0.1)
    canvas = Overlay(lang="en").draw(frame.copy(), state, posture)
    ok &= check(
        "叠加渲染",
        canvas.shape == frame.shape and not np.array_equal(canvas, frame),
    )

    # Writing a video is where codec problems show up.
    out = ROOT / "smoke_out.mp4"
    writer = cv2.VideoWriter(
        str(out), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (640, 480)
    )
    for _ in range(5):
        writer.write(canvas)
    writer.release()
    ok &= check("录制写盘", out.exists() and out.stat().st_size > 0)
    out.unlink(missing_ok=True)

    print("\n全部通过。" if ok else "\n有检查未通过。", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
