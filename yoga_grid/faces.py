"""人脸遮挡：检测到人脸就盖一个卡通面具。

只用 MediaPipe Pose 自带的面部关键点（鼻、双眼、双耳、嘴角），不额外跑人脸
检测模型 —— 这些点姿态估计时已经算出来了，够用来定位头部中心、大小和倾角。

遮挡直接画在**原始整帧**上，而不是裁好的格子上：裁剪会改变坐标系，在整帧上
做一次，后续的九宫格、单张图、候选缩略图就全都是遮好的，不会漏。
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from . import landmarks as L

# 判断「脸露出来了」用到的关键点。用鼻、眼、嘴而不用耳朵 —— 侧脸时远侧的耳朵
# 会被遮住，但脸其实是露着的。
_FACE_KEYS = ("nose", "left_eye", "right_eye", "mouth_left", "mouth_right")

SKIN = (176, 214, 240)      # BGR，暖米色
OUTLINE = (96, 120, 150)
FEATURE = (54, 62, 78)
BLUSH = (150, 168, 240)


def face_visibility(lm: np.ndarray) -> float:
    """面部关键点的平均置信度 —— 越高越可能是正脸对着镜头。"""
    idx = [L.IDX[n] for n in _FACE_KEYS]
    return float(np.clip(lm[idx, 2].mean(), 0.0, 1.0))


def _face_geometry(
    lm: np.ndarray, width: int, height: int
) -> tuple[tuple[int, int], int, float] | None:
    """估出 (头部中心, 半径, 倾角)。信息不足时返回 None。"""
    pts = L.to_pixels(lm, width, height)

    nose = pts[L.IDX["nose"]]
    left_ear, right_ear = pts[L.IDX["left_ear"]], pts[L.IDX["right_ear"]]
    left_eye, right_eye = pts[L.IDX["left_eye"]], pts[L.IDX["right_eye"]]

    ear_vis = min(lm[L.IDX["left_ear"], 2], lm[L.IDX["right_ear"], 2])
    eye_span = float(np.linalg.norm(left_eye - right_eye))
    ear_span = float(np.linalg.norm(left_ear - right_ear))

    # 半径：优先用耳距（最接近头宽），其次眼距，最后退回躯干比例。
    # 系数取得比「脸」略大 —— 要盖住整个头，只盖五官会像贴了张小贴纸，
    # 发际线和轮廓还在，遮挡也就不彻底。
    if ear_vis >= 0.5 and ear_span > 4.0:
        radius = ear_span * 0.98
        center = (left_ear + right_ear) / 2.0
        axis = right_ear - left_ear
    elif eye_span > 3.0:
        radius = eye_span * 2.45
        center = (left_eye + right_eye) / 2.0
        axis = right_eye - left_eye
    else:
        torso = L.torso_scale(pts)
        if not (torso > 1e-6):
            return None
        radius = torso * 0.34
        center = nose.copy()
        axis = pts[L.IDX["right_shoulder"]] - pts[L.IDX["left_shoulder"]]

    # 头部中心比鼻尖略靠后（朝耳线方向），面具才不会偏到脸前面。
    center = center * 0.65 + nose * 0.35

    if radius < 6.0:
        return None
    angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))
    return (int(round(center[0])), int(round(center[1]))), int(round(radius)), angle


def _rotate(offset: tuple[float, float], degrees: float) -> tuple[float, float]:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    x, y = offset
    return (x * c - y * s, x * s + y * c)


def draw_cartoon_face(
    frame: np.ndarray, center: tuple[int, int], radius: int, angle: float
) -> None:
    """就地画一个卡通脸：圆脸 + 两只眼 + 微笑，随头部倾角旋转。"""
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
    frame: np.ndarray, lm: np.ndarray | None, min_visibility: float = 0.55
) -> bool:
    """人脸够清楚就在 ``frame`` 上盖卡通面具。返回是否真的盖了。

    ``min_visibility`` 是「脸露出来了」的门槛。背对镜头时 MediaPipe 仍会预测
    面部关键点，但置信度低 —— 用它来区分，避免朝天/背身的帧被无端盖一张脸。
    """
    if lm is None:
        return False
    if face_visibility(lm) < min_visibility:
        return False

    height, width = frame.shape[:2]
    geometry = _face_geometry(lm, width, height)
    if geometry is None:
        return False

    center, radius, angle = geometry
    # 脸完全在画面外就不用画了
    if center[0] + radius < 0 or center[0] - radius > width:
        return False
    if center[1] + radius < 0 or center[1] - radius > height:
        return False

    draw_cartoon_face(frame, center, radius, angle)
    return True
