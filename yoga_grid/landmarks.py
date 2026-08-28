"""MediaPipe Pose 33 关键点的索引表与几何工具。

坐标约定
--------
* ``lm``     形状 (33, 3)，列为 (x, y, visibility)。x/y 是 MediaPipe 的归一化
             坐标（相对图像宽/高各自归一），身体超出画面时可能落在 [0,1] 之外。
* ``pts``    形状 (33, 2) 的像素坐标。x/y 已乘回图像宽高，所以两轴同尺度，
             角度计算才有意义 —— 直接用归一化坐标算角度会被画面长宽比拉歪。
* ``norm``   形状 (33, 2)，以髋中点为原点、以躯干长度为单位的像素坐标。
             平移和尺度都被消除，可跨帧、跨机位比较。

图像坐标系 y 轴向下，所有「更高」都意味着 y 更小。
"""

from __future__ import annotations

import numpy as np

# MediaPipe Pose 的 33 个关键点，顺序即索引。
NAMES: list[str] = [
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
]

IDX: dict[str, int] = {name: i for i, name in enumerate(NAMES)}
N_LANDMARKS = len(NAMES)

# 用于姿态比较和打分的主干关键点：躯干四肢 + 鼻尖。
# 刻意排除手指、眼、耳、嘴 —— 它们在全身镜头里抖动大，会污染距离度量。
CORE = [
    IDX[n]
    for n in (
        "nose",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
        "left_hip",
        "right_hip",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_foot_index",
        "right_foot_index",
    )
]

# 用于计算人体外框的关键点：比 CORE 多带上耳朵和脚跟，
# 免得裁图时把头顶或脚跟切掉。
EXTENT = CORE + [IDX[n] for n in ("left_ear", "right_ear", "left_heel", "right_heel")]

# 左右镜像置换表：MIRROR[i] 是 i 的对侧关键点索引。
MIRROR = np.arange(N_LANDMARKS)
for _i, _name in enumerate(NAMES):
    if _name.startswith("left_"):
        _j = IDX["right_" + _name[len("left_") :]]
        MIRROR[_i], MIRROR[_j] = _j, _i

_OPPOSITE_SIDE = {"left": "right", "right": "left"}


# --------------------------------------------------------------------------
# 基础换算
# --------------------------------------------------------------------------


def to_pixels(lm: np.ndarray, width: int, height: int) -> np.ndarray:
    """归一化坐标 -> 像素坐标，形状 (33, 2)。"""
    pts = lm[:, :2].astype(np.float64, copy=True)
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def midpoint(pts: np.ndarray, left: str, right: str) -> np.ndarray:
    return (pts[IDX[left]] + pts[IDX[right]]) / 2.0


def torso_scale(pts: np.ndarray) -> float:
    """躯干长度（肩中点到髋中点），作为该帧的尺度单位。

    人蜷起来（婴儿式）或正对镜头时躯干投影会很短，所以逐级回退到肩宽、
    再回退到整体展开尺寸，避免除出一个爆炸的比例。
    """
    shoulder = midpoint(pts, "left_shoulder", "right_shoulder")
    hip = midpoint(pts, "left_hip", "right_hip")
    torso = float(np.linalg.norm(shoulder - hip))

    spread = float(np.linalg.norm(pts[CORE].max(axis=0) - pts[CORE].min(axis=0)))
    # 躯干投影短于整体展开的 12% 时不可信 —— 换肩宽。
    if torso > 1e-6 and (spread <= 1e-6 or torso / spread > 0.12):
        return torso

    shoulder_width = float(
        np.linalg.norm(pts[IDX["left_shoulder"]] - pts[IDX["right_shoulder"]])
    )
    if shoulder_width > 1e-6:
        return shoulder_width * 1.5
    return max(spread, 1e-6)


def normalize(pts: np.ndarray) -> np.ndarray:
    """像素坐标 -> 髋中点为原点、躯干长度为单位的坐标。"""
    hip = midpoint(pts, "left_hip", "right_hip")
    return (pts - hip) / torso_scale(pts)


def mirror_pose(norm: np.ndarray) -> np.ndarray:
    """左右镜像：交换对侧关键点并翻转 x。"""
    out = norm[MIRROR].copy()
    out[:, 0] *= -1.0
    return out


def bbox_norm(
    lm: np.ndarray, vis_thresh: float = 0.5, indices: list[int] | None = None
) -> tuple[float, float, float, float]:
    """人体外框，归一化坐标 (x0, y0, x1, y1)。

    只用足够可信的关键点；可信点太少时退回全部点，宁可框大也别框错。
    """
    idx = EXTENT if indices is None else indices
    visible = [i for i in idx if lm[i, 2] >= vis_thresh]
    if len(visible) < 4:
        visible = list(idx)
    p = lm[visible, :2]
    return (
        float(p[:, 0].min()),
        float(p[:, 1].min()),
        float(p[:, 0].max()),
        float(p[:, 1].max()),
    )


def mean_visibility(lm: np.ndarray, indices: list[int] | None = None) -> float:
    idx = CORE if indices is None else indices
    return float(np.clip(lm[idx, 2].mean(), 0.0, 1.0))


# --------------------------------------------------------------------------
# 角度与方向
# --------------------------------------------------------------------------


def angle_at(vertex: np.ndarray, a: np.ndarray, c: np.ndarray) -> float:
    """vertex 处由 vertex->a 与 vertex->c 张开的夹角，单位度，退化时返回 nan。"""
    v1 = a - vertex
    v2 = c - vertex
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-9 or n2 < 1e-9:
        return float("nan")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_to_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """线段 a-b 与铅垂线的夹角，0 = 竖直，90 = 水平。不区分朝向。"""
    v = b - a
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(abs(v[1]) / n, 0.0, 1.0))))


def angle_to_horizontal(a: np.ndarray, b: np.ndarray) -> float:
    """线段 a-b 与水平线的夹角，0 = 水平，90 = 竖直。不区分朝向。"""
    v = b - a
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(abs(v[0]) / n, 0.0, 1.0))))


# --------------------------------------------------------------------------
# 带侧别解析的访问器
# --------------------------------------------------------------------------

# 虚拟关键点：由若干真实点算出来的中点。
_VIRTUAL = {
    "hip_mid": lambda p: midpoint(p, "left_hip", "right_hip"),
    "shoulder_mid": lambda p: midpoint(p, "left_shoulder", "right_shoulder"),
    "knee_mid": lambda p: midpoint(p, "left_knee", "right_knee"),
    "ankle_mid": lambda p: midpoint(p, "left_ankle", "right_ankle"),
    "wrist_mid": lambda p: midpoint(p, "left_wrist", "right_wrist"),
    "elbow_mid": lambda p: midpoint(p, "left_elbow", "right_elbow"),
    "foot_mid": lambda p: midpoint(p, "left_foot_index", "right_foot_index"),
    # 胸腰交界附近，用于把「后弯分布在整段胸椎」这类要点锚定到脊柱中段。
    "spine_mid": lambda p: (
        midpoint(p, "left_shoulder", "right_shoulder") * 0.5
        + midpoint(p, "left_hip", "right_hip") * 0.5
    ),
}


#: 每个虚拟点由哪些真实关键点算出来 —— 追踪置信度时要把它们都算进去。
#: 和 ``_VIRTUAL`` 一一对应；加虚拟点时两处都要改，测试会盯着。
_VIRTUAL_PARTS: dict[str, tuple[str, ...]] = {
    "hip_mid": ("left_hip", "right_hip"),
    "shoulder_mid": ("left_shoulder", "right_shoulder"),
    "knee_mid": ("left_knee", "right_knee"),
    "ankle_mid": ("left_ankle", "right_ankle"),
    "wrist_mid": ("left_wrist", "right_wrist"),
    "elbow_mid": ("left_elbow", "right_elbow"),
    "foot_mid": ("left_foot_index", "right_foot_index"),
    "spine_mid": ("left_shoulder", "right_shoulder", "left_hip", "right_hip"),
}


class PoseView:
    """按侧别解析关键点名的只读视图。

    体式模板不知道练习者朝哪边，于是用 ``s_`` 前缀表示「主侧」、``o_`` 表示
    「另一侧」。同一个模板用 side="left" 和 side="right" 各跑一遍取高分，
    就自动兼容了左右两个方向的同一体式。

    距离和 dy 的单位是躯干长度（前提是传进来的是 normalize() 的输出）。

    置信度（可选）
    -------------
    ``vis`` 是 (33,) 的置信度数组，来自 MediaPipe 原始关键点的第三列。
    传了它，就可以问「刚才那次测量用到的关键点里，最低的置信度是多少」——
    见 :meth:`trace` 和 :meth:`traced_confidence`。

    为什么需要：``normalize()`` 只保留 (33, 2)，置信度那一列被丢掉了，于是
    规则层**无法知道哪些关键点是真的**。侧面机位下远侧那半边身体每一帧都被
    挡住，MediaPipe 给的是猜测（实测某支视频右膝置信度中位数 0.315、80% 的帧
    低于 0.5），而它倾向于照着可见那一侧复制，所以数值看着完全合理。

    ``vis`` 为 None 时所有置信度一律按 1.0 处理 —— 手搭骨架的测试和早期
    只存了归一化骨架的 scores.json 都走这条路，行为和没有这个特性时一样。
    """

    __slots__ = ("pts", "side", "vis", "_touched")

    def __init__(
        self, pts: np.ndarray, side: str = "left", vis: np.ndarray | None = None
    ) -> None:
        self.pts = pts
        self.side = side
        self.vis = vis
        self._touched: set[int] | None = None

    def _resolve(self, name: str) -> str:
        if name.startswith("s_"):
            return f"{self.side}_{name[2:]}"
        if name.startswith("o_"):
            return f"{_OPPOSITE_SIDE[self.side]}_{name[2:]}"
        return name

    def trace(self) -> None:
        """开始记录接下来读了哪些关键点。"""
        self._touched = set()

    def traced_confidence(self) -> float:
        """记录期间读到的关键点里**最低**的置信度，并停止记录。

        取 min 而不是均值：一次测量只要用到一个猜出来的关键点，结果就不可信，
        其余点再确定也补不回来。

        没传 ``vis``、或者一个关键点都没读到时返回 1.0。
        """
        touched, self._touched = self._touched, None
        if self.vis is None or not touched:
            return 1.0
        return float(min(self.vis[i] for i in touched))

    def pt(self, name: str) -> np.ndarray:
        if name in _VIRTUAL:
            if self._touched is not None:
                self._touched.update(IDX[n] for n in _VIRTUAL_PARTS[name])
            return _VIRTUAL[name](self.pts)
        name = self._resolve(name)
        if self._touched is not None:
            self._touched.add(IDX[name])
        return self.pts[IDX[name]]

    def ang(self, a: str, vertex: str, c: str) -> float:
        """vertex 处的关节角。"""
        return angle_at(self.pt(vertex), self.pt(a), self.pt(c))

    def vert(self, a: str, b: str) -> float:
        return angle_to_vertical(self.pt(a), self.pt(b))

    def horiz(self, a: str, b: str) -> float:
        return angle_to_horizontal(self.pt(a), self.pt(b))

    def dy(self, a: str, b: str) -> float:
        """b 相对 a 向下的位移（躯干长度为单位）。正 = b 比 a 低。"""
        return float(self.pt(b)[1] - self.pt(a)[1])

    def dist(self, a: str, b: str) -> float:
        return float(np.linalg.norm(self.pt(a) - self.pt(b)))
