"""跨平台兼容层，主要是 Windows 上的非 ASCII 路径与控制台编码。

OpenCV 的文件读写走的是窄字符 C API，在 Windows 上遇到中文路径会直接失败 ——
不抛异常，只是静默返回 False 或 ``isOpened() == False``。练习视频叫
`我的练习.mp4`、输出目录叫 `瑜伽`，都会踩到。这里把这些坑一次性填掉。

Linux / macOS 上这些函数全是直通，没有额外开销。
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

_IS_WINDOWS = os.name == "nt"


def configure_console() -> None:
    """把 stdout / stderr 切到 UTF-8，避免中文进度信息炸掉。

    Windows 直连控制台时 Python 走 WriteConsoleW，中文没问题；但一旦输出被
    重定向到文件或管道，就改用 locale 编码（cp1252 / cp936），打印中文会抛
    UnicodeEncodeError —— 于是「把日志存下来看」这个再正常不过的操作会让
    整个程序崩掉。
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass  # 已被替换成不支持重配的流，忽略


def cv2_path(path: Path | str) -> str:
    """转成 OpenCV 在本平台能接受的路径字符串。

    Windows 上若路径含非 ASCII 字符，尝试取 8.3 短路径（纯 ASCII）。
    短路径可能被卷标策略禁用，那时原样返回，由调用方走 ``prepare_video``
    的复制兜底。
    """
    text = str(path)
    if not _IS_WINDOWS or text.isascii():
        return text

    try:
        import ctypes

        buffer = ctypes.create_unicode_buffer(len(text) + 260)
        length = ctypes.windll.kernel32.GetShortPathNameW(  # type: ignore[attr-defined]
            text, buffer, len(buffer)
        )
        if length and buffer.value and buffer.value.isascii():
            return buffer.value
    except (ImportError, AttributeError, OSError):
        pass
    return text


def imwrite(path: Path | str, image: np.ndarray, params: list[int] | None = None) -> None:
    """``cv2.imwrite`` 的 Unicode 安全版本。

    先在内存里编码，再用 Python 写字节 —— 绕开 OpenCV 的窄字符路径限制。
    """
    path = Path(path)
    suffix = path.suffix or ".jpg"
    ok, buffer = cv2.imencode(suffix, image, params or [])
    if not ok:
        raise RuntimeError(f"图像编码失败（格式 {suffix}）：{path}")
    path.write_bytes(buffer.tobytes())


def open_capture(path: Path | str) -> cv2.VideoCapture:
    """打开视频，失败时抛出带可读信息的异常。"""
    capture = cv2.VideoCapture(cv2_path(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(
            f"打不开视频：{path}\n"
            "可能原因：文件损坏、格式不受本机 OpenCV 支持，"
            "或路径含非 ASCII 字符（Windows）。"
        )
    return capture


def prepare_video(path: Path) -> tuple[Path, Callable[[], None]]:
    """确认视频可被 OpenCV 打开，返回 (可用路径, 清理函数)。

    非 ASCII 路径且短路径也不奏效时，复制一份到临时 ASCII 路径。这一步只在
    整条流水线开头做一次 —— 视频动辄几百 MB，不能每次开流都复制。
    """

    def noop() -> None:
        return None

    probe = cv2.VideoCapture(cv2_path(path))
    if probe.isOpened():
        probe.release()
        return path, noop
    probe.release()

    if str(path).isascii():
        raise RuntimeError(
            f"打不开视频：{path}\n"
            "文件可能损坏，或本机 OpenCV 不支持这个封装格式（试试转成 .mp4）。"
        )

    # 兜底：复制到临时 ASCII 路径再打开。
    tmp_dir = Path(tempfile.mkdtemp(prefix="yoga_grid_"))
    tmp_path = tmp_dir / f"video{path.suffix or '.mp4'}"
    print(f"路径含非 ASCII 字符，OpenCV 无法直接读取，先复制到 {tmp_path}", file=sys.stderr)
    shutil.copy2(path, tmp_path)

    probe = cv2.VideoCapture(str(tmp_path))
    opened = probe.isOpened()
    probe.release()
    if not opened:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise RuntimeError(
            f"打不开视频：{path}\n"
            "复制到 ASCII 路径后仍然打不开，文件可能损坏或格式不受支持。"
        )

    return tmp_path, lambda: shutil.rmtree(tmp_dir, ignore_errors=True)
