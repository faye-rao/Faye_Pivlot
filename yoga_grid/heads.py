"""头部朝向与位置：遮不遮脸、遮在哪、遮多大。

**纯几何，不 import cv2 / mediapipe。** 遮脸是隐私功能，判断错一次就是一张
真脸被发出去，所以判断和落笔拆开：这里只吃一个 (33, 3) 的关键点数组，输出
「朝向 / 中心 / 半径 / 倾角」；``faces.py`` 只负责按这个结果画。这样判断逻辑
能用手搭的骨架直接测（见 ``tests/test_faces.py``），不需要视频、模型和窗口。

朝向判据：**手性（左右次序）**，不是置信度
----------------------------------------
MediaPipe 的 ``left_*`` / ``right_*`` 是**解剖学**左右：面朝镜头时，人的左耳
出现在画面**右**侧；转身背对镜头，同一个左耳跑到画面左侧。所以「左耳→右耳」
这个向量的朝向会在转身时翻转，而置信度不会 —— 背对镜头时面部关键点照样是
高置信度的（模型知道那些部位在画面里、没被别的东西挡住，它并不报告「你正
对着我吗」）。

只比 x 大小不够：人在体式里会整体旋转（肩倒立、下犬、侧卧），翻转过来时
左右在画面上的次序同样会颠倒。所以把横轴投到**躯干朝上方向**的垂线上取
有符号量：

    facing = cross(up_hat, left_ear - right_ear) / 躯干长度

正 = 面朝镜头，负 = 背对镜头，0 附近 = 侧脸（两耳在画面上重叠，符号没有
意义）。绕视线轴整体旋转时 up 和横轴一起转，叉积不变 —— 所以倒立的正脸
仍然判成正脸。

侧脸落在死区里，这时**照样遮**：侧脸是露着的脸，漏掉才是隐私事故；把一张
卡通脸盖在侧对镜头的人头上只是难看。取舍写在 README 的「人脸遮挡」一节。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import landmarks as L

# 朝向三态。侧脸和判据退化都归到 UNCLEAR —— 两者的处置相同（遮）。
FRONT = "front"
AWAY = "away"
UNCLEAR = "unclear"

# 「这一帧里到底有没有头」的门槛。取面部关键点里**最高**的那个，不取平均：
# 侧脸时远侧的眼和嘴角被挡住、置信度接近 0，平均值会被拖到门槛以下，而脸
# 其实是露着的。朝向由手性负责判，这个门槛只负责挡掉「头根本不在画面里」。
MIN_FACE_CONFIDENCE = 0.25

# 拿来定横轴的那一对点，两侧都至少要有这么点置信度才用。
_MIN_PAIR_CONFIDENCE = 0.2

# 手性死区，单位是躯干长度。正脸约 ±0.30（两耳间距≈0.3 个躯干长），
# 0.08 大致对应「距正侧面 15° 以内」，在这个范围里符号已经不可信。
FACING_DEADBAND = 0.08

# 面部关键点。判「有没有头」用鼻、眼、嘴，不用耳朵 —— 见 MIN_FACE_CONFIDENCE。
_FACE_KEYS = ("nose", "left_eye", "right_eye", "mouth_left", "mouth_right")

# 半径与躯干长度的比例。真人躯干（肩中点→髋中点）约 50 cm，头宽约 15 cm，
# 所以「盖住整个头」的半径≈0.30 个躯干长 —— 和耳距估计（ear_span * 0.98）
# 在正脸时正好对上。0.45 是上限，只用来兜住关键点跳飞的帧。
_RADIUS_FROM_TORSO = 0.30
_RADIUS_MAX_TORSO = 0.45

# 耳距 / 眼距换算成半径的系数。这两个跨度只会随头部转动而**缩短**（正侧面时
# 两耳几乎重叠），不会变长，所以几个估计里取最大的那个才是对的方向。
_RADIUS_FROM_EAR_SPAN = 0.98
_RADIUS_FROM_EYE_SPAN = 2.45

# 横轴短到这个比例以下就不用它算倾角了（正侧面时它只剩噪声），
# 改用躯干朝上方向的垂线。
_AXIS_MIN_RATIO = 0.35


@dataclass(frozen=True)
class Head:
    """一个可以直接拿去画的头部：像素坐标的中心、半径、倾角（度）。"""

    center: tuple[float, float]
    radius: float
    angle: float
    facing: str
    facing_score: float


# --------------------------------------------------------------------------
# 基础量
# --------------------------------------------------------------------------


def face_confidence(lm: np.ndarray) -> float:
    """面部关键点里最可信的那个的置信度。

    取 max 而不是 mean：只要有一个面部点被稳稳定位，头就在画面里。
    """
    idx = [L.IDX[n] for n in _FACE_KEYS]
    return float(np.clip(lm[idx, 2].max(), 0.0, 1.0))


def _up_axis(pts: np.ndarray) -> np.ndarray | None:
    """躯干朝上方向的单位向量（髋中点 → 肩中点，指向头那一端）。"""
    up = L.midpoint(pts, "left_shoulder", "right_shoulder") - L.midpoint(
        pts, "left_hip", "right_hip"
    )
    n = float(np.linalg.norm(up))
    if n < 1e-6:
        # 躯干在投影里退化成一点（正对镜头蜷起来）时退回「肩中点 → 鼻」。
        up = pts[L.IDX["nose"]] - L.midpoint(pts, "left_shoulder", "right_shoulder")
        n = float(np.linalg.norm(up))
        if n < 1e-6:
            return None
    return up / n


def _lateral_axis(lm: np.ndarray, pts: np.ndarray) -> np.ndarray | None:
    """头部横轴：从右侧点指向左侧点（解剖学左右）。

    优先双耳，退回双眼。两对都不可信时返回 None。

    刻意是**优先级**而不是「两对取共识」：耳朵正脸背身都看得见，基线又最长，
    符号最稳；眼睛在背身时压根看不见，位置是模型猜的、横向次序很可能是错的。
    要求两对一致等于把最不可信的信号引进最需要它可靠的那条路 —— 背身帧会
    因为「两对不一致」退回「判不清」，于是又被遮上一张脸，正是要修的那个错。
    """
    for left, right in (("left_ear", "right_ear"), ("left_eye", "right_eye")):
        conf = min(lm[L.IDX[left], 2], lm[L.IDX[right], 2])
        if conf < _MIN_PAIR_CONFIDENCE:
            continue
        axis = pts[L.IDX[left]] - pts[L.IDX[right]]
        if float(np.linalg.norm(axis)) > 1e-6:
            return axis
    return None


def _span(lm: np.ndarray, pts: np.ndarray, left: str, right: str) -> float | None:
    """一对左右点的像素间距；置信度不够时返回 None。"""
    if min(lm[L.IDX[left], 2], lm[L.IDX[right], 2]) < _MIN_PAIR_CONFIDENCE:
        return None
    return float(np.linalg.norm(pts[L.IDX[left]] - pts[L.IDX[right]]))


# --------------------------------------------------------------------------
# 朝向
# --------------------------------------------------------------------------


def facing_score(lm: np.ndarray, width: int, height: int) -> float:
    """有符号朝向量：正 = 面朝镜头，负 = 背对镜头，单位是躯干长度。

    判据算不出来（没躯干、没可信的左右点对）时返回 nan。
    """
    pts = L.to_pixels(lm, width, height)
    up = _up_axis(pts)
    lateral = _lateral_axis(lm, pts)
    if up is None or lateral is None:
        return float("nan")
    torso = L.torso_scale(pts)
    if not (torso > 1e-6):
        return float("nan")
    cross = float(up[0] * lateral[1] - up[1] * lateral[0])
    return cross / torso


def facing_of(score: float, deadband: float = FACING_DEADBAND) -> str:
    """把有符号朝向量分档成 FRONT / AWAY / UNCLEAR。"""
    if math.isnan(score):
        return UNCLEAR
    if score > deadband:
        return FRONT
    if score < -deadband:
        return AWAY
    return UNCLEAR


def facing(
    lm: np.ndarray, width: int, height: int, deadband: float = FACING_DEADBAND
) -> str:
    """FRONT / AWAY / UNCLEAR。UNCLEAR 包含侧脸和判据退化。"""
    return facing_of(facing_score(lm, width, height), deadband)


# --------------------------------------------------------------------------
# 位置与大小
# --------------------------------------------------------------------------


def head_geometry(lm: np.ndarray, width: int, height: int) -> Head | None:
    """估出头部的中心、半径、倾角。信息不足时返回 None。

    半径刻意取「几个估计里最大的」再用躯干长度封顶：耳距和眼距在侧脸时被
    透视压短，直接乘系数会得到一张只盖住五官的小贴纸，头顶和轮廓还露着。
    """
    pts = L.to_pixels(lm, width, height)
    torso = L.torso_scale(pts)
    if not (torso > 1e-6):
        return None

    nose = pts[L.IDX["nose"]]
    ear_span = _span(lm, pts, "left_ear", "right_ear")
    eye_span = _span(lm, pts, "left_eye", "right_eye")

    estimates = [torso * _RADIUS_FROM_TORSO]
    if ear_span is not None:
        estimates.append(ear_span * _RADIUS_FROM_EAR_SPAN)
    if eye_span is not None:
        estimates.append(eye_span * _RADIUS_FROM_EYE_SPAN)
    radius = min(max(estimates), torso * _RADIUS_MAX_TORSO)
    if not (radius > 1e-6):
        return None

    # 中心：耳中点最接近头部中心，其次眼中点，最后退回鼻尖；
    # 再往鼻子方向拉一点，免得面具整体偏到脑后。
    if ear_span is not None:
        base = L.midpoint(pts, "left_ear", "right_ear")
    elif eye_span is not None:
        base = L.midpoint(pts, "left_eye", "right_eye")
    else:
        base = nose
    center = base * 0.65 + nose * 0.35

    # 倾角：横轴够长就用它（跟着头的倾斜走），否则用躯干朝上方向的垂线。
    up = _up_axis(pts)
    axis = _lateral_axis(lm, pts)
    if axis is None or float(np.linalg.norm(axis)) < radius * _AXIS_MIN_RATIO:
        axis = None if up is None else np.array([-up[1], up[0]])
    if axis is None:
        angle = 0.0
    else:
        angle = math.degrees(math.atan2(float(axis[1]), float(axis[0])))

    score = facing_score(lm, width, height)
    return Head(
        center=(float(center[0]), float(center[1])),
        radius=float(radius),
        angle=angle,
        facing=facing_of(score),
        facing_score=score,
    )


def _fully_offscreen(head: Head, width: int, height: int) -> bool:
    cx, cy = head.center
    r = head.radius
    return cx + r < 0 or cx - r > width or cy + r < 0 or cy - r > height


def head_to_mask(
    lm: np.ndarray | None,
    width: int,
    height: int,
    min_confidence: float = MIN_FACE_CONFIDENCE,
    deadband: float = FACING_DEADBAND,
) -> Head | None:
    """要遮的话，返回遮哪；不该遮就返回 None。

    只有两种情况不遮：**头不在画面里**（置信度门槛、整圆出框），以及
    **明确背对镜头**（手性为负且超出死区）。侧脸和判不清的一律遮 ——
    宁可多盖一张卡通脸，不能漏一张真脸。
    """
    if lm is None:
        return None
    if face_confidence(lm) < min_confidence:
        return None

    head = head_geometry(lm, width, height)
    if head is None:
        return None
    if head.facing_score < -deadband:
        return None
    if _fully_offscreen(head, width, height):
        return None
    return head
