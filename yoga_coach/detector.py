"""MediaPipe Pose Landmarker wrapper.

This is the only module that imports MediaPipe, so the geometry, the pose
rules and the tests stay independent of it.

MediaPipe's Tasks API needs a ``.task`` model bundle on disk.  The file is
downloaded once into a cache directory (``~/.cache/yoga_coach/models`` by
default) and reused afterwards, so only the first run needs the network.
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

from .landmarks import DEFAULT_MIN_VISIBILITY, Skeleton

MODEL_BASE_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"

#: Model variants, cheapest first.  ``lite`` keeps up with a webcam on a
#: laptop CPU; ``heavy`` is noticeably more accurate on unusual angles such as
#: Downward Dog but wants a GPU or a lot of patience.
MODELS: dict[str, str] = {
    "lite": f"{MODEL_BASE_URL}/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": f"{MODEL_BASE_URL}/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": f"{MODEL_BASE_URL}/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}


def model_cache_dir() -> Path:
    override = os.environ.get("YOGA_COACH_MODEL_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "yoga_coach" / "models"


def ensure_model(variant: str = "lite", *, quiet: bool = False) -> Path:
    """Return the local path to a model bundle, downloading it if needed."""
    if variant not in MODELS:
        raise ValueError(f"unknown model {variant!r}; pick one of {', '.join(MODELS)}")

    cache = model_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"pose_landmarker_{variant}.task"
    if path.exists() and path.stat().st_size > 0:
        return path

    url = MODELS[variant]
    if not quiet:
        print(f"首次运行：正在下载姿态模型 {variant} ...", file=sys.stderr)
    # Download to a temporary name first so an interrupted download does not
    # leave a truncated file that later runs would happily load.
    partial = path.with_suffix(".task.part")
    try:
        with urllib.request.urlopen(url, timeout=120) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out)
        partial.replace(path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if not quiet:
        print(f"模型已保存到 {path}", file=sys.stderr)
    return path


class PoseDetector:
    """Detects one body per frame and returns it as a :class:`Skeleton`.

    Use as a context manager, or call :meth:`close` when finished -- the
    underlying graph holds native resources.
    """

    def __init__(
        self,
        model: str = "lite",
        *,
        video_mode: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_visibility: float = DEFAULT_MIN_VISIBILITY,
        quiet: bool = False,
    ) -> None:
        # Imported lazily: importing MediaPipe costs a second or two and pulls
        # in native libraries, which the unit tests should not have to pay for.
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        self._mp = mp
        self._vision = vision
        self.min_visibility = min_visibility
        self.video_mode = video_mode
        self._last_timestamp_ms = -1

        model_path = ensure_model(model, quiet=quiet)
        running_mode = (
            vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        try:
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        except OSError as exc:
            # MediaPipe's native library links against the system OpenGL/EGL
            # stack, which minimal Linux images often lack.  The raw
            # "libEGL.so.1: cannot open shared object file" is not much help.
            if "so" in str(exc) and sys.platform.startswith("linux"):
                raise RuntimeError(
                    f"MediaPipe 无法加载系统图形库（{exc}）。"
                    "在 Debian/Ubuntu 上执行："
                    "sudo apt install libegl1 libgles2 libgl1 libglib2.0-0"
                ) from exc
            raise

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> Skeleton | None:
        """Run the detector on one OpenCV BGR frame.

        In video mode timestamps must strictly increase; a repeated or
        out-of-order value is nudged forward by a millisecond rather than
        raising, because webcam clocks are not always monotonic.
        """
        rgb = frame_bgr[:, :, ::-1]
        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb),
        )

        if self.video_mode:
            ts = 0 if timestamp_ms is None else int(timestamp_ms)
            if ts <= self._last_timestamp_ms:
                ts = self._last_timestamp_ms + 1
            self._last_timestamp_ms = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        if not result.pose_landmarks:
            return None
        return Skeleton.from_list(
            result.pose_landmarks[0], min_visibility=self.min_visibility
        )

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
