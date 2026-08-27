"""每个体式的标准骨架，用于回归测试模板之间不互相误判。

坐标约定与 ``landmarks`` 一致：像素坐标，y 轴向下，肩中点到髋中点的距离
（即躯干长度）统一取 100，方便读数 —— 归一化后各项 dy / dist 的值就直接是
「几个躯干长」。

这些骨架不是真人数据，是按解剖几何手搭的「教科书版」体式。它们的用途不是
证明模板在真实视频上准，而是**钉住模板之间的区分度**：模板集一大，最容易
出的问题是 A 体式被 B 的模板认走（上犬式被三角伸展式认走就是真实发生过的
例子）。有了这张网，加新模板时若破坏了既有区分，测试立刻会红。
"""

from __future__ import annotations

import math

import numpy as np

from yoga_grid import landmarks as L


def skeleton(**joints: tuple[float, float]) -> np.ndarray:
    """按关键点名搭骨架，未指定的点留在原点。"""
    pts = np.zeros((L.N_LANDMARKS, 2), dtype=np.float64)
    for name, (x, y) in joints.items():
        pts[L.IDX[name]] = (x, y)
    return pts


def rotate(pts: np.ndarray, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return pts @ np.array([[c, s], [-s, c]])


# --------------------------------------------------------------------------
# 站姿
# --------------------------------------------------------------------------


def mountain() -> np.ndarray:
    """山式：笔直站立，双臂垂于体侧。"""
    return skeleton(
        nose=(0, -130),
        left_ear=(-10, -125), right_ear=(10, -125),
        left_shoulder=(-20, -100), right_shoulder=(20, -100),
        left_elbow=(-24, -50), right_elbow=(24, -50),
        left_wrist=(-26, 0), right_wrist=(26, 0),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-15, 100), right_knee=(15, 100),
        left_ankle=(-15, 200), right_ankle=(15, 200),
        left_heel=(-15, 205), right_heel=(15, 205),
        left_foot_index=(-15, 215), right_foot_index=(15, 215),
    )


def warrior2() -> np.ndarray:
    """战士二式：左腿为前腿屈约 98°，右腿后展伸直，双臂肩高水平展开。"""
    return skeleton(
        nose=(0, -130),
        left_ear=(-10, -125), right_ear=(10, -125),
        left_shoulder=(-20, -100), right_shoulder=(20, -100),
        left_elbow=(-80, -100), right_elbow=(80, -100),
        left_wrist=(-140, -100), right_wrist=(140, -100),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-115, 15), left_ankle=(-115, 115),
        right_knee=(75, 58), right_ankle=(135, 116),
        left_heel=(-115, 120), right_heel=(135, 121),
        left_foot_index=(-135, 125), right_foot_index=(152, 126),
    )


def warrior1() -> np.ndarray:
    """战士一式：与战士二式同样的下肢，但双臂上举过头。"""
    return skeleton(
        nose=(0, -130),
        left_ear=(-10, -125), right_ear=(10, -125),
        left_shoulder=(-20, -100), right_shoulder=(20, -100),
        left_elbow=(-25, -155), right_elbow=(25, -155),
        left_wrist=(-14, -200), right_wrist=(14, -200),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-115, 15), left_ankle=(-115, 115),
        right_knee=(75, 58), right_ankle=(135, 116),
        left_heel=(-115, 120), right_heel=(135, 121),
        left_foot_index=(-135, 125), right_foot_index=(152, 126),
    )


def warrior3() -> np.ndarray:
    """战士三式：站左腿竖直，躯干与右腿都水平前后伸展，双臂向前。"""
    return skeleton(
        nose=(-130, 0),
        left_ear=(-122, -8), right_ear=(-122, 8),
        left_shoulder=(-95, -10), right_shoulder=(-105, 10),
        left_elbow=(-180, -10), right_elbow=(-190, 10),
        left_wrist=(-265, -10), right_wrist=(-275, 10),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-15, 100), left_ankle=(-15, 200),
        right_knee=(105, 0), right_ankle=(195, 0),
        left_heel=(-15, 205), right_heel=(200, 5),
        left_foot_index=(-15, 215), right_foot_index=(215, 0),
    )


def triangle() -> np.ndarray:
    """三角伸展式：双腿伸直大幅分开，躯干侧倾约 55°，双臂成竖直一线。"""
    return skeleton(
        nose=(-100, -80),
        left_ear=(-95, -75), right_ear=(-88, -85),
        left_shoulder=(-93.4, -41.0), right_shoulder=(-70.4, -73.8),
        left_elbow=(-86, 0), right_elbow=(-78, -116),
        left_wrist=(-90, 60), right_wrist=(-74, -175),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-80, 100), left_ankle=(-145, 200),
        right_knee=(80, 100), right_ankle=(145, 200),
        left_heel=(-150, 205), right_heel=(150, 205),
        left_foot_index=(-165, 210), right_foot_index=(160, 210),
    )


def parsvakonasana() -> np.ndarray:
    """侧角伸展式：前膝屈 90°，躯干侧倾约 60°，下手落地、上臂过头伸展。"""
    return skeleton(
        nose=(-105, -75),
        left_ear=(-102, -68), right_ear=(-95, -82),
        left_shoulder=(-96.6, -32.7), right_shoulder=(-76.6, -67.3),
        left_elbow=(-108, 40), right_elbow=(-70, -128),
        left_wrist=(-120, 110), right_wrist=(-54, -207),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-115, 15), left_ankle=(-115, 115),
        right_knee=(75, 58), right_ankle=(135, 116),
        left_heel=(-115, 120), right_heel=(135, 121),
        left_foot_index=(-135, 125), right_foot_index=(152, 126),
    )


def tree() -> np.ndarray:
    """树式：站左腿竖直，右膝外开、右脚贴左腿，双臂上举。"""
    return skeleton(
        nose=(0, -130),
        left_ear=(-10, -125), right_ear=(10, -125),
        left_shoulder=(-20, -100), right_shoulder=(20, -100),
        left_elbow=(-25, -155), right_elbow=(25, -155),
        left_wrist=(-8, -205), right_wrist=(8, -205),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-15, 100), left_ankle=(-15, 200),
        right_knee=(95, 60), right_ankle=(20, 75),
        left_heel=(-15, 205), right_heel=(15, 78),
        left_foot_index=(-15, 215), right_foot_index=(10, 80),
    )


def anjaneyasana() -> np.ndarray:
    """新月式：前膝屈 90°，后膝跪地屈曲，双臂上举过头，躯干直立。"""
    return skeleton(
        nose=(0, -130),
        left_ear=(-10, -125), right_ear=(10, -125),
        left_shoulder=(-20, -100), right_shoulder=(20, -100),
        left_elbow=(-25, -155), right_elbow=(25, -155),
        left_wrist=(-14, -200), right_wrist=(14, -200),
        left_hip=(-15, 0), right_hip=(15, 0),
        # 前腿：大腿近水平、小腿竖直
        left_knee=(-110, 20), left_ankle=(-110, 120),
        # 后腿：膝落地在髋后下方，小腿向后平放
        right_knee=(70, 105), right_ankle=(160, 125),
        left_heel=(-110, 125), right_heel=(165, 128),
        left_foot_index=(-130, 130), right_foot_index=(180, 118),
    )


# --------------------------------------------------------------------------
# 跪姿 / 坐姿
# --------------------------------------------------------------------------


def ardha_hanumanasana() -> np.ndarray:
    """半神猴式：前腿伸直贴地，后膝跪地，躯干前折over前腿。"""
    return skeleton(
        nose=(-150, -18),
        left_ear=(-142, -26), right_ear=(-142, -10),
        # 躯干前折：肩中点在髋前方、略高
        left_shoulder=(-118, -30), right_shoulder=(-112, -14),
        left_elbow=(-160, 30), right_elbow=(-154, 46),
        left_wrist=(-190, 78), right_wrist=(-184, 94),
        left_hip=(-8, 40), right_hip=(2, 40),
        # 前腿伸直，接近水平贴地
        left_knee=(-95, 60), left_ankle=(-182, 80),
        # 后膝跪地，屈约 90°
        right_knee=(30, 130), right_ankle=(115, 108),
        left_heel=(-190, 84), right_heel=(122, 104),
        left_foot_index=(-198, 62), right_foot_index=(132, 118),
    )


def pigeon() -> np.ndarray:
    """鸽子式：前腿屈膝外旋落地，后腿向后伸直贴地，髋部沉地，躯干直立。"""
    return skeleton(
        nose=(-18, -120),
        left_ear=(-26, -114), right_ear=(-10, -114),
        left_shoulder=(-30, -88), right_shoulder=(-16, -80),
        left_elbow=(-36, -30), right_elbow=(-22, -22),
        left_wrist=(-40, 26), right_wrist=(-26, 34),
        left_hip=(-8, 12), right_hip=(4, 12),
        # 前腿：膝外开在体前，小腿横向
        left_knee=(-105, 30), left_ankle=(-40, 66),
        # 后腿：向后伸直贴地
        right_knee=(96, 26), right_ankle=(188, 40),
        left_heel=(-30, 70), right_heel=(196, 44),
        left_foot_index=(-14, 74), right_foot_index=(210, 34),
    )


def child() -> np.ndarray:
    """婴儿式：双膝深屈坐跟，躯干折叠贴腿，双臂前伸。"""
    return skeleton(
        nose=(-96, 60),
        left_ear=(-88, 50), right_ear=(-88, 70),
        left_shoulder=(-62, 34), right_shoulder=(-56, 52),
        left_elbow=(-124, 56), right_elbow=(-118, 74),
        left_wrist=(-186, 72), right_wrist=(-180, 90),
        left_hip=(30, 46), right_hip=(40, 46),
        left_knee=(-46, 78), right_knee=(-36, 78),
        left_ankle=(46, 84), right_ankle=(56, 84),
        left_heel=(52, 82), right_heel=(62, 82),
        left_foot_index=(58, 92), right_foot_index=(68, 92),
    )


def bridge() -> np.ndarray:
    """桥式：肩背贴地，髋部抬高，屈膝约 90°。"""
    return skeleton(
        nose=(-128, 96),
        left_ear=(-120, 88), right_ear=(-120, 104),
        left_shoulder=(-96, 90), right_shoulder=(-90, 106),
        left_elbow=(-40, 106), right_elbow=(-34, 122),
        left_wrist=(16, 116), right_wrist=(22, 132),
        left_hip=(-6, 34), right_hip=(4, 34),
        left_knee=(84, 46), right_knee=(94, 46),
        left_ankle=(78, 132), right_ankle=(88, 132),
        left_heel=(74, 138), right_heel=(84, 138),
        left_foot_index=(96, 140), right_foot_index=(106, 140),
    )


# --------------------------------------------------------------------------
# 俯卧支撑 / 反向支撑
# --------------------------------------------------------------------------


def downdog() -> np.ndarray:
    """下犬式：髋部为最高点，躯干与手臂成一线，四肢伸直。"""
    return skeleton(
        nose=(-100, -20),
        left_ear=(-95, -26), right_ear=(-95, -14),
        left_shoulder=(-78, -40), right_shoulder=(-82, -40),
        left_elbow=(-119, -20), right_elbow=(-121, -20),
        left_wrist=(-160, 0), right_wrist=(-160, 0),
        left_hip=(2, -140), right_hip=(-2, -140),
        left_knee=(81, -70), right_knee=(79, -70),
        left_ankle=(160, 0), right_ankle=(160, 0),
        left_heel=(165, 5), right_heel=(165, 5),
        left_foot_index=(185, 0), right_foot_index=(185, 0),
    )


def plank() -> np.ndarray:
    """平板式：身体成一直线略前高后低，双臂伸直垂地，低头。"""
    return skeleton(
        nose=(-135, 30),
        left_ear=(-128, 22), right_ear=(-128, 38),
        left_shoulder=(-105, 0), right_shoulder=(-95, 0),
        left_elbow=(-105, 50), right_elbow=(-95, 50),
        left_wrist=(-105, 100), right_wrist=(-95, 100),
        left_hip=(-7, 21), right_hip=(3, 21),
        left_knee=(86, 40), right_knee=(92, 40),
        left_ankle=(177, 60), right_ankle=(183, 60),
        left_heel=(184, 66), right_heel=(190, 66),
        left_foot_index=(180, 96), right_foot_index=(186, 96),
    )


def chaturanga() -> np.ndarray:
    """四柱支撑式：身体成一直线接近水平，屈肘约 90°、上臂贴肋，低头。"""
    return skeleton(
        nose=(-134, 42),
        left_ear=(-127, 34), right_ear=(-127, 50),
        left_shoulder=(-104, 14), right_shoulder=(-94, 14),
        # 屈肘 90°：肘在肩后方同高，腕在肘正下方
        left_elbow=(-56, 18), right_elbow=(-46, 18),
        left_wrist=(-60, 60), right_wrist=(-50, 60),
        left_hip=(-6, 26), right_hip=(4, 26),
        left_knee=(88, 34), right_knee=(94, 34),
        left_ankle=(182, 42), right_ankle=(188, 42),
        left_heel=(189, 48), right_heel=(195, 48),
        left_foot_index=(185, 78), right_foot_index=(191, 78),
    )


def updog() -> np.ndarray:
    """上犬式：双臂伸直撑地，胸腔上提、肩高于髋，双腿向后伸直贴地，仰头。"""
    return skeleton(
        nose=(-115, -85),
        left_ear=(-108, -78), right_ear=(-104, -74),
        left_shoulder=(-88.5, -55), right_shoulder=(-78.5, -55),
        left_elbow=(-89, -12), right_elbow=(-79, -12),
        left_wrist=(-90, 30), right_wrist=(-80, 30),
        left_hip=(-5, 0), right_hip=(5, 0),
        left_knee=(85, 15), right_knee=(95, 15),
        left_ankle=(175, 30), right_ankle=(185, 30),
        left_heel=(182, 34), right_heel=(192, 34),
        left_foot_index=(195, 38), right_foot_index=(205, 38),
    )


def side_plank() -> np.ndarray:
    """侧板式：单臂撑地、身体成斜直线，上臂向天空伸展，双腿伸直叠放。"""
    return skeleton(
        nose=(-128, -34),
        left_ear=(-120, -28), right_ear=(-118, -24),
        left_shoulder=(-96, -12), right_shoulder=(-92, -6),
        # 支撑臂（左）伸直向下撑地
        left_elbow=(-98, 40), left_wrist=(-100, 92),
        # 上臂（右）向天空伸展
        right_elbow=(-84, -70), right_wrist=(-78, -134),
        left_hip=(-2, 8), right_hip=(4, 12),
        left_knee=(88, 26), right_knee=(94, 30),
        left_ankle=(178, 44), right_ankle=(184, 48),
        left_heel=(185, 50), right_heel=(191, 54),
        left_foot_index=(196, 52), right_foot_index=(202, 56),
    )


def reverse_plank() -> np.ndarray:
    """反板式：双臂伸直撑于身后，髋部抬起成斜直线，胸腔朝上、头后仰。"""
    return skeleton(
        nose=(-132, -46),
        left_ear=(-124, -40), right_ear=(-124, -24),
        left_shoulder=(-102, -14), right_shoulder=(-92, -14),
        left_elbow=(-100, 36), right_elbow=(-90, 36),
        left_wrist=(-98, 86), right_wrist=(-88, 86),
        left_hip=(-6, 10), right_hip=(4, 10),
        left_knee=(86, 30), right_knee=(92, 30),
        left_ankle=(178, 50), right_ankle=(184, 50),
        left_heel=(172, 58), right_heel=(178, 58),
        left_foot_index=(198, 56), right_foot_index=(204, 56),
    )


# key -> 骨架构造函数。测试遍历这张表。
CANONICAL: dict[str, callable] = {
    "mountain": mountain,
    "warrior1": warrior1,
    "warrior2": warrior2,
    "warrior3": warrior3,
    "triangle": triangle,
    "parsvakonasana": parsvakonasana,
    "tree": tree,
    "anjaneyasana": anjaneyasana,
    "ardha_hanumanasana": ardha_hanumanasana,
    "pigeon": pigeon,
    "child": child,
    "bridge": bridge,
    "downdog": downdog,
    "plank": plank,
    "chaturanga": chaturanga,
    "updog": updog,
    "side_plank": side_plank,
    "reverse_plank": reverse_plank,
}


def uttanasana() -> np.ndarray:
    """站立前屈式：双腿伸直竖直，躯干自髋向下倒垂，头低于髋，双手落地。"""
    return skeleton(
        nose=(-16, 112),
        left_ear=(-24, 104), right_ear=(-8, 104),
        # 躯干倒垂：肩中点在髋下方
        left_shoulder=(-22, 86), right_shoulder=(-2, 86),
        left_elbow=(-26, 140), right_elbow=(-6, 140),
        left_wrist=(-28, 194), right_wrist=(-8, 194),
        left_hip=(-15, 0), right_hip=(15, 0),
        left_knee=(-15, 100), right_knee=(15, 100),
        left_ankle=(-15, 200), right_ankle=(15, 200),
        left_heel=(-15, 205), right_heel=(15, 205),
        left_foot_index=(-15, 215), right_foot_index=(15, 215),
    )


CANONICAL["uttanasana"] = uttanasana
