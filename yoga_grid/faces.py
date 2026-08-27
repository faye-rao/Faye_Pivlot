"""人脸遮挡：往露出的人脸上盖一个卡通面具。

**这里只有画笔。**「该不该遮、遮在哪、遮多大」全在 ``heads.py`` —— 那是纯
几何、能用手搭骨架测试的判断层；本模块拿到一个 ``Head`` 就落笔，不做判断。

用的是 MediaPipe Pose 自带的面部关键点（鼻、双眼、双耳、嘴角），不额外跑
人脸检测模型 —— 这些点姿态估计时已经算出来了，够用来定位头部中心、大小和
倾角。

遮挡直接画在**原始整帧**上，而不是裁好的格子上：裁剪会改变坐标系，在整帧上
做一次，后续的九宫格、单张图、候选缩略图就全都是遮好的，不会漏。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import heads

SKIN = (176, 214, 240)      # BGR，暖米色
OUTLINE = (96, 120, 150)
FEATURE = (54, 62, 78)
BLUSH = (150, 168, 240)


def _rotate(offset: tuple[float, float], degrees: float) -> tuple[float, float]:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    x, y = offset
    return (x * c - y * s, x * s + y * c)


def draw_cartoon_face(
    frame: np.ndarray, center: tuple[int, int], radius: int, angle: float
) -> None:
    """就地画一个卡通脸：圆脸 + 两只眼 + 微笑，随头部倾角旋转。

    ``angle`` 是头部横轴（指向解剖学左侧）在画面里的方向角，0 = 头正立。
    """
    cx, cy = center
    r = radius

    cv2.circle(frame, (cx, cy), r, SKIN, -1, lineType=cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, OUTLINE, max(2, r // 14), lineType=cv2.LINE_AA)

    eye_r = max(2, int(r * 0.13))
    for sign in (-1, 1):
        ox, oy = _rotate((sign * r * 0.36, -r * 0.16), angle)
        cv2.circle(
            frame, (int(cx + ox), int(cy + oy)), eye_r, FEATURE, -1, lineType=cv2.LINE_AA
        )

    # 腮红
    for sign in (-1, 1):
        ox, oy = _rotate((sign * r * 0.55, r * 0.22), angle)
        cv2.ellipse(
            frame, (int(cx + ox), int(cy + oy)),
            (max(2, int(r * 0.16)), max(1, int(r * 0.10))),
            angle, 0, 360, BLUSH, -1, lineType=cv2.LINE_AA,
        )

    # 微笑：一段下半弧
    mx, my = _rotate((0.0, r * 0.20), angle)
    cv2.ellipse(
        frame, (int(cx + mx), int(cy + my)),
        (max(3, int(r * 0.38)), max(2, int(r * 0.26))),
        angle, 20, 160, FEATURE, max(2, r // 16), lineType=cv2.LINE_AA,
    )


def mask_face(
    frame: np.ndarray,
    lm: np.ndarray | None,
    min_confidence: float = heads.MIN_FACE_CONFIDENCE,
    deadband: float = heads.FACING_DEADBAND,
) -> bool:
    """该遮就在 ``frame`` 上盖卡通面具。返回是否真的盖了。

    判断全部委托给 ``heads.head_to_mask``：头不在画面里、或明确背对镜头时
    不遮，侧脸和判不清的一律遮。
    """
    height, width = frame.shape[:2]
    head = heads.head_to_mask(lm, width, height, min_confidence, deadband)
    if head is None:
        return False

    center = (int(round(head.center[0])), int(round(head.center[1])))
    draw_cartoon_face(frame, center, max(1, int(round(head.radius))), head.angle)
    return True
