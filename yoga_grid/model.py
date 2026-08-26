"""MediaPipe Pose Landmarker 模型文件的获取与缓存。"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models/pose_landmarker"

VARIANTS = {
    "lite": f"{_BASE}/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    "full": f"{_BASE}/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "heavy": f"{_BASE}/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task",
}


def cache_dir() -> Path:
    root = os.environ.get("YOGA_GRID_CACHE")
    if root:
        return Path(root).expanduser()
    return Path.home() / ".cache" / "yoga_grid" / "models"


def resolve_model(spec: str = "full") -> Path:
    """把 'lite'/'full'/'heavy' 或一个文件路径解析成本地模型文件。

    变体名会在必要时下载到缓存目录；已存在则直接复用。
    """
    if spec not in VARIANTS:
        path = Path(spec).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"模型文件不存在：{path}\n"
                f"也可以直接用变体名：{', '.join(VARIANTS)}"
            )
        return path

    url = VARIANTS[spec]
    target = cache_dir() / Path(url).name
    if target.is_file() and target.stat().st_size > 0:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载姿态模型 {spec} -> {target}", file=sys.stderr)
    tmp = target.with_suffix(target.suffix + ".part")
    try:
        # urllib 默认读取 http_proxy / https_proxy 环境变量。
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as fh:
            while chunk := response.read(1 << 20):
                fh.write(chunk)
        tmp.replace(target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return target
