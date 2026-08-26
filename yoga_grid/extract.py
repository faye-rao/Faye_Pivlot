"""第一遍：稀疏抽帧 + 姿态估计。

只在内存里留下每帧的关键点和几个标量指标（几 KB/帧），不留图像 ——
十分钟视频的 720p 帧全存下来是 GB 级的。真正要用的九张图在第二遍
（``grid.py``）按帧号回视频里取，还能拿到原始分辨率。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import landmarks as L


@dataclass
class VideoInfo:
    path: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass
class FramePose:
    """一个采样帧的姿态与画质指标。"""

    idx: int          # 在采样序列里的下标
    frame_no: int     # 原视频里的帧号
    t: float          # 时间戳（秒）
    lm: np.ndarray | None          # (33, 3) 归一化 x, y, visibility
    norm: np.ndarray | None        # (33, 2) 髋心归一化坐标
    bbox: tuple[float, float, float, float] | None  # 归一化 x0,y0,x1,y1
    visibility: float
    sharpness: float               # 拉普拉斯方差，越大越清晰

    @property
    def detected(self) -> bool:
        return self.lm is not None


def probe(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not (fps > 0) or fps > 1000:
            fps = 30.0
        return VideoInfo(
            path=path,
            fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def _resize_long_side(frame: np.ndarray, long_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    longest = max(h, w)
    if long_side <= 0 or longest <= long_side:
        return frame
    scale = long_side / longest
    return cv2.resize(
        frame, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _sharpness(gray: np.ndarray, bbox: tuple[float, float, float, float] | None) -> float:
    """人体区域的拉普拉斯方差。整帧算会被背景细节带偏，所以只看人。"""
    h, w = gray.shape[:2]
    if bbox is not None:
        x0 = int(np.clip(bbox[0], 0.0, 1.0) * w)
        y0 = int(np.clip(bbox[1], 0.0, 1.0) * h)
        x1 = int(np.clip(bbox[2], 0.0, 1.0) * w)
        y1 = int(np.clip(bbox[3], 0.0, 1.0) * h)
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            gray = gray[y0:y1, x0:x1]
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _largest_pose(result, width: int, height: int) -> np.ndarray | None:
    """多人入画时取外框最大的那个（通常就是练习者本人）。"""
    if not result.pose_landmarks:
        return None

    best: np.ndarray | None = None
    best_area = -1.0
    for person in result.pose_landmarks:
        lm = np.array(
            [(p.x, p.y, min(p.visibility, p.presence)) for p in person],
            dtype=np.float64,
        )
        x0, y0, x1, y1 = L.bbox_norm(lm)
        area = max(x1 - x0, 0.0) * width * max(y1 - y0, 0.0) * height
        if area > best_area:
            best_area, best = area, lm
    return best


def extract(
    path: Path,
    model_path: Path,
    interval: float = 0.5,
    work_size: int = 720,
    num_poses: int = 1,
    min_confidence: float = 0.5,
    progress: bool = True,
) -> tuple[VideoInfo, list[FramePose]]:
    """按 ``interval`` 秒抽帧，对每帧做姿态估计。"""
    import mediapipe as mp

    info = probe(path)
    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=num_poses,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
    )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")

    step = max(1, round(info.fps * interval))
    expected = (info.frame_count // step) if info.frame_count > 0 else 0
    frames: list[FramePose] = []

    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        frame_no = 0
        while True:
            if not cap.grab():
                break
            if frame_no % step == 0:
                ok, bgr = cap.retrieve()
                if ok and bgr is not None:
                    frames.append(
                        _process(landmarker, bgr, len(frames), frame_no, frame_no / info.fps, work_size)
                    )
                    if progress and expected and len(frames) % 40 == 0:
                        pct = 100.0 * len(frames) / max(expected, 1)
                        print(
                            f"  抽帧 {len(frames)}/{expected} ({pct:.0f}%)",
                            file=sys.stderr,
                        )
            frame_no += 1
    finally:
        landmarker.close()
        cap.release()

    # 容器元数据里的帧数常常不准，用实际读到的帧数补正时长。
    if frames:
        info.frame_count = max(info.frame_count, frames[-1].frame_no + 1)
    return info, frames


def _process(
    landmarker, bgr: np.ndarray, idx: int, frame_no: int, t: float, work_size: int
) -> FramePose:
    import mediapipe as mp

    small = _resize_long_side(bgr, work_size)
    h, w = small.shape[:2]
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(image)

    lm = _largest_pose(result, w, h)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    if lm is None:
        return FramePose(idx, frame_no, t, None, None, None, 0.0, _sharpness(gray, None))

    bbox = L.bbox_norm(lm)
    norm = L.normalize(L.to_pixels(lm, w, h))
    return FramePose(
        idx=idx,
        frame_no=frame_no,
        t=t,
        lm=lm,
        norm=norm,
        bbox=bbox,
        visibility=L.mean_visibility(lm),
        sharpness=_sharpness(gray, bbox),
    )


def iter_frames_at(path: Path, frame_numbers: list[int]) -> Iterator[tuple[int, np.ndarray]]:
    """第二遍：按帧号流式产出原始分辨率的帧。

    顺序遍历 + grab() 跳帧，不做随机 seek —— 对长 GOP 的视频，随机 seek
    既慢又可能落在错误的关键帧上。

    刻意做成生成器：候选帧可能有一两百个，1080p 全存进 dict 是 GB 级内存。
    """
    wanted = sorted(set(frame_numbers))
    if not wanted:
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频：{path}")

    try:
        target_iter = iter(wanted)
        target = next(target_iter)
        frame_no = 0
        while True:
            if not cap.grab():
                break
            if frame_no == target:
                ok, bgr = cap.retrieve()
                if ok and bgr is not None:
                    yield frame_no, bgr
                try:
                    target = next(target_iter)
                except StopIteration:
                    break
            frame_no += 1
    finally:
        cap.release()


def read_frames_at(path: Path, frame_numbers: list[int]) -> dict[int, np.ndarray]:
    """``iter_frames_at`` 的字典版，只在帧数很少（比如入选的九张）时用。"""
    return dict(iter_frames_at(path, frame_numbers))
