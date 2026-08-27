"""标准体式库：每个体式的标准骨架、发力要点，以及线稿渲染。

坐标约定与 ``landmarks`` 一致：像素坐标，y 轴向下，肩中点到髋中点的距离
（即躯干长度）统一取 100，方便读数 —— 归一化后各项 dy / dist 的值就直接是
「几个躯干长」。

这些骨架不是真人数据，是按解剖几何手搭的「教科书版」体式，有两个用途：

1. **钉住模板之间的区分度**（``tests/test_templates.py``）。模板集一大，最容易
   出的问题是 A 体式被 B 的模板认走 —— 上犬式被三角伸展式认走就是真实发生过
   的例子。有了这张网，加新模板时若破坏了既有区分，测试立刻会红。
2. **渲染标准体式对照图**。参考图是用打分模板的目标几何**本身**画出来的，
   而不是找一张来源不明的照片 —— 这样「你的姿势 vs 标准图」的差距就和正位分
   扣掉的分是同一件事。换成外部照片，图里的角度未必等于模板的目标值，
   对照反而会误导。

发力要点（``CUES``）是通用教学口令，只描述该体式的常见对位与发力方向，
不针对个人。它不知道你的柔韧度、代偿或旧伤，不能替代老师。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import landmarks as L


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


def parsvottanasana() -> np.ndarray:
    """金字塔式：双脚前后错开、双腿伸直，躯干折叠于前腿之上，双手落地。"""
    return skeleton(
        nose=(-95, 120),
        left_ear=(-88, 112), right_ear=(-84, 116),
        left_shoulder=(-81, 62), right_shoulder=(-51, 88),
        left_elbow=(-92, 126), right_elbow=(-66, 142),
        left_wrist=(-100, 190), right_wrist=(-78, 196),
        left_hip=(-8, 0), right_hip=(8, 0),
        # 前腿（左）与后腿（右）都伸直，落点前后错开
        left_knee=(-38, 95), left_ankle=(-68, 190),
        right_knee=(38, 95), right_ankle=(68, 190),
        left_heel=(-74, 196), right_heel=(76, 196),
        left_foot_index=(-88, 192), right_foot_index=(84, 190),
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
    "parsvottanasana": parsvottanasana,
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


# --------------------------------------------------------------------------
# 发力要点
# --------------------------------------------------------------------------
#
# 通用教学口令，描述该体式的对位与发力方向。刻意不写「应该感觉到……」这类
# 主观体感，也不写针对个人的调整 —— 那需要老师看着你本人给。

CUES: dict[str, tuple[str, ...]] = {
    "mountain": (
        "双脚四角均匀踩地，重心落在足弓中央",
        "大腿前侧上提收紧，膝盖不向后锁死",
        "尾骨微向下，肋骨下沉不外翻",
        "双肩后展下沉，耳、肩、髋在一条铅垂线上",
    ),
    "uttanasana": (
        "从髋部折叠，不是从腰椎弯",
        "坐骨向上，与脚跟形成上下对抗",
        "膝盖可微屈 —— 优先保住背部延展，而不是硬把腿伸直",
        "颈部完全放松，头顶自然垂向地面",
    ),
    "warrior1": (
        "前膝对准第二、三脚趾，不向内塌",
        "后脚外缘压实地面，后腿主动伸直",
        "髋部尽量转向正前方，尾骨向下",
        "上举时肩胛下沉，不耸肩挤颈",
    ),
    "warrior2": (
        "前膝在脚踝正上方，对准第二、三脚趾",
        "后脚外缘下压，后腿从髋到脚跟主动伸直",
        "躯干保持正中 —— 不前倾，也不向前腿侧倒",
        "双臂向两端延展，肩胛下沉远离耳朵",
    ),
    "warrior3": (
        "支撑腿大腿收紧，膝盖不锁死",
        "骨盆保持水平，抬起腿那侧的髋不向上翻",
        "从指尖到抬起腿的脚跟拉成一条直线",
        "核心收紧，肋骨不向地面塌",
    ),
    "triangle": (
        "双腿主动伸直，前腿大腿上提，膝盖不超伸",
        "两侧侧腰等长，下侧腰不塌陷",
        "胸腔朝侧上方旋转打开，不要向下扣",
        "下手轻放，重量不压在腿上",
    ),
    "parsvakonasana": (
        "前膝屈约 90°，对准第二、三脚趾",
        "后脚外缘压实，后腿从髋到脚跟延展",
        "下侧腰不塌，胸腔向上旋转打开",
        "上臂过耳延展，与后腿连成一条斜线",
    ),
    "parsvottanasana": (
        "从髋部折叠，背部保持延展而不是拱起",
        "骨盆两侧保持等高，不向后腿那侧歪",
        "前腿膝盖可微屈，避免超伸",
        "后脚外缘压实，后腿从髋到脚跟主动伸直",
        "颈部放松，不主动伸头去找腿",
    ),
    "tree": (
        "支撑脚均匀压地，脚趾放松不抓地",
        "抬起脚放在大腿内侧或小腿 —— 不压在膝关节上",
        "抬起腿的膝向侧后方打开，骨盆保持中立",
        "尾骨向下，肋骨不外翻",
    ),
    "anjaneyasana": (
        "前膝在脚踝正上方，不超过脚尖",
        "后脚背贴地，后侧髋前方向下延展",
        "尾骨向下卷，避免用腰部塌陷来代偿",
        "上举时从腋窝向上延展，肩胛下沉",
    ),
    "ardha_hanumanasana": (
        "髋部落在后膝正上方",
        "前腿脚趾回勾，膝盖可微屈避免超伸",
        "从髋折叠，脊柱保持延展而不是蜷曲",
        "骨盆保持水平，不向一侧歪",
    ),
    "pigeon": (
        "骨盆保持中正，两侧坐骨等高",
        "前腿小腿角度按髋部开度调整，不强求横平",
        "后侧髋前方向下沉，脚背贴地",
        "后腿膝盖若有压力，先把前侧坐骨垫高",
    ),
    "child": (
        "大脚趾相触，膝盖可以分开",
        "坐骨向脚跟方向沉",
        "额头落地，或垫抱枕支撑，颈部放松",
        "把呼吸送到后背肋骨",
    ),
    "bridge": (
        "双脚与髋同宽，脚跟在膝盖正下方",
        "先卷尾骨再抬髋，脊柱逐节离地",
        "大腿内侧内旋，膝盖不向外翻",
        "肩胛内收压地，为胸腔打开腾出空间",
    ),
    "downdog": (
        "双手压实，食指根部不要抬起",
        "坐骨向上向后推，与双手形成对抗",
        "膝盖可微屈，用它换脊柱的延展",
        "大腿内旋，脚跟朝地面方向沉",
    ),
    "updog": (
        "双手压实，肩胛下沉远离耳朵",
        "大腿主动离地，脚背压实",
        "后弯分布在整段胸椎，不集中在下背",
        "颈部保持延展，不过度后折",
    ),
    "plank": (
        "肩在腕正上方，整个手掌压实",
        "腹部收紧，髋部不塌也不上翘",
        "大腿收紧，脚跟向后推",
        "后颈延展，视线落在双手前方地面",
    ),
    "chaturanga": (
        "肘部贴肋，前臂垂直地面",
        "肩不低于肘 —— 宁可停高一点，也不要塌肩",
        "全身成一条直线，髋部不上翘",
        "核心与大腿持续收紧",
    ),
    "side_plank": (
        "支撑手在肩正下方，压实整个手掌",
        "髋部主动上提，不向下坠",
        "双腿收紧叠放，脚外缘或前脚掌压地",
        "上侧肩向后打开，头颈保持中立",
    ),
    "reverse_plank": (
        "双手在肩后下方，指尖朝脚的方向",
        "髋部主动上推，胸腔向上打开",
        "双腿收紧，脚背向下延展",
        "颈部后仰保持在舒适范围，不硬压",
    ),
}


def cues(key: str) -> tuple[str, ...]:
    return CUES.get(key, ())


def pose_names() -> dict[str, str]:
    """key -> 中文名。从模板表取，避免两处各写一份名字。"""
    from .poses import TEMPLATES_BY_KEY

    return {k: t.zh for k, t in TEMPLATES_BY_KEY.items()}


# --------------------------------------------------------------------------
# 线稿渲染
# --------------------------------------------------------------------------

# 要连线的骨段。头部单独处理（颈线 + 头圈），因为鼻尖是个虚拟的「头」位置，
# 直接连到肩会画出一条穿过脖子的怪线。
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot_index"),
    ("right_ankle", "right_foot_index"),
)

INK = (44, 54, 66)
INK_SOFT = (120, 132, 148)
PAPER = (247, 245, 240)
ACCENT = (214, 106, 74)   # 发力方向箭头与锚点标记


def _figure_extent(pts: np.ndarray) -> tuple[float, float, float, float, float]:
    """返回人像的外接框 (min_x, min_y, max_x, max_y) 和头圈半径。"""
    torso = L.torso_scale(pts)
    head_r = torso * 0.26
    nose = pts[L.IDX["nose"]]
    drawn = [pts[L.IDX[a]] for edge in SKELETON_EDGES for a in edge]
    xs = [p[0] for p in drawn] + [nose[0] - head_r, nose[0] + head_r]
    ys = [p[1] for p in drawn] + [nose[1] - head_r, nose[1] + head_r]
    return min(xs), min(ys), max(xs), max(ys), head_r


def _thick_line(draw, a, b, width: float, fill) -> None:
    """带圆头的粗线 —— PIL 的 line 没有 round cap，用两端补圆点实现。"""
    draw.line([a, b], fill=fill, width=max(1, round(width)))
    r = width / 2.0
    for x, y in (a, b):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=fill)


def _draw_figure(draw, pts: np.ndarray, place, scale: float, ink=INK) -> None:
    """画一具人像：躯干带体积，四肢为粗线，头为圆圈。

    躯干画成一条很粗的线（肩中点到髋中点），而不是「左右肩连左右髋」的四边形 ——
    侧面视角下左右肩几乎重合，四边形会退化成一条缝，反而没有体积。
    """
    torso = L.torso_scale(pts)
    head_r = torso * 0.26
    nose = pts[L.IDX["nose"]]
    shoulder_mid = L.midpoint(pts, "left_shoulder", "right_shoulder")
    hip_mid = L.midpoint(pts, "left_hip", "right_hip")

    torso_w = torso * 0.34 * scale
    limb_w = torso * 0.115 * scale
    joint_r = limb_w * 0.62

    _thick_line(draw, place(shoulder_mid), place(hip_mid), torso_w, ink)
    _thick_line(draw, place(pts[L.IDX["left_shoulder"]]),
                place(pts[L.IDX["right_shoulder"]]), torso_w * 0.72, ink)
    _thick_line(draw, place(pts[L.IDX["left_hip"]]),
                place(pts[L.IDX["right_hip"]]), torso_w * 0.72, ink)

    # 颈线停在头圈边缘，不画进圈里
    neck_a, neck_b = place(shoulder_mid), place(nose)
    dx, dy = neck_b[0] - neck_a[0], neck_b[1] - neck_a[1]
    length = max((dx * dx + dy * dy) ** 0.5, 1e-6)
    stop = head_r * scale
    _thick_line(
        draw, neck_a,
        (neck_b[0] - dx / length * stop, neck_b[1] - dy / length * stop),
        limb_w * 1.15, ink,
    )

    torso_edges = {
        frozenset(("left_shoulder", "right_shoulder")),
        frozenset(("left_hip", "right_hip")),
        frozenset(("left_shoulder", "left_hip")),
        frozenset(("right_shoulder", "right_hip")),
    }
    for a, b in SKELETON_EDGES:
        if frozenset((a, b)) in torso_edges:
            continue
        _thick_line(draw, place(pts[L.IDX[a]]), place(pts[L.IDX[b]]), limb_w, ink)

    for a, b in SKELETON_EDGES:
        for name in (a, b):
            x, y = place(pts[L.IDX[name]])
            draw.ellipse([x - joint_r, y - joint_r, x + joint_r, y + joint_r], fill=ink)

    hx, hy = place(nose)
    hr = head_r * scale
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr],
                 outline=ink, width=max(2, round(limb_w * 0.85)))


def render_pose(
    key: str,
    size: int = 420,
    ink: tuple[int, int, int] = INK,
    paper: tuple[int, int, int] | None = PAPER,
) -> "Image.Image":
    """把标准骨架画成线稿，返回 size×size 的图。

    ``paper=None`` 返回透明背景（RGBA），便于叠到别的底图上。
    """
    from PIL import Image, ImageDraw

    build = CANONICAL.get(key)
    if build is None:
        raise KeyError(f"没有 {key} 的标准骨架")
    pts = build()

    min_x, min_y, max_x, max_y, _ = _figure_extent(pts)
    pad = size * 0.10
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    scale = (size - 2 * pad) / span
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    def to_canvas(p) -> tuple[float, float]:
        return (size / 2.0 + (p[0] - cx) * scale, size / 2.0 + (p[1] - cy) * scale)

    mode = "RGB" if paper is not None else "RGBA"
    image = Image.new(mode, (size, size), paper if paper is not None else (0, 0, 0, 0))
    _draw_figure(ImageDraw.Draw(image), pts, to_canvas, scale, ink)
    return image


# --------------------------------------------------------------------------
# 带锚点的发力要点 + 对照卡渲染
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cue:
    """一条锚定到具体部位的发力要点。

    ``anchor`` 是关键点名或虚拟点名（``hip_mid``、``spine_mid`` 等）。
    ``arrow`` 是发力方向的单位向量，在图形自身坐标系里（y 轴向下），
    ``None`` 表示这条要点讲的是对位而非方向，不画箭头。
    """

    text: str
    anchor: str
    arrow: tuple[float, float] | None = None


_UP = (0.0, -1.0)
_DOWN = (0.0, 1.0)

# 带锚点的要点。上面的 CUES 是纯文字版（供 report.md 用），这里是画在图上的版本，
# 一条要点只讲一件事，才好用引线指到具体部位。
ANNOTATED: dict[str, tuple[Cue, ...]] = {
    "mountain": (
        Cue("双肩后展下沉", "left_shoulder", _DOWN),
        Cue("耳、肩、髋在一条铅垂线上", "spine_mid"),
        Cue("尾骨微向下，肋骨不外翻", "hip_mid", _DOWN),
        Cue("大腿前侧上提，膝不锁死", "left_knee", _UP),
        Cue("双脚四角均匀踩地", "left_foot_index", _DOWN),
    ),
    "uttanasana": (
        Cue("从髋部折叠，不是从腰弯", "hip_mid"),
        Cue("坐骨向上，与脚跟对抗", "hip_mid", _UP),
        Cue("颈部完全放松，\n头顶自然垂向地面", "nose", _DOWN),
        Cue("膝可微屈，优先保住背部延展", "left_knee"),
        Cue("双脚踩实，重心在足弓", "left_foot_index", _DOWN),
    ),
    "warrior1": (
        Cue("肩胛下沉，不耸肩挤颈", "left_shoulder", _DOWN),
        Cue("双臂上举，从腋窝向上延展", "left_wrist", _UP),
        Cue("髋部朝向正前方，尾骨向下", "hip_mid", _DOWN),
        Cue("前膝对准第二、三脚趾，不内塌", "left_knee"),
        Cue("后脚外缘压实，后腿主动伸直", "right_ankle", _DOWN),
    ),
    "warrior2": (
        Cue("双臂向两端延展，肩胛下沉", "left_wrist", (-1.0, 0.0)),
        Cue("躯干保持正中，不前倾不侧倒", "spine_mid"),
        Cue("前膝在踝正上方，\n对准第二、三脚趾", "left_knee", _DOWN),
        Cue("后腿从髋到脚跟主动伸直", "right_knee"),
        Cue("后脚外缘下压", "right_foot_index", _DOWN),
    ),
    "warrior3": (
        Cue("从指尖到抬起腿的脚跟\n拉成一条直线", "left_wrist", (-1.0, 0.0)),
        Cue("核心收紧，肋骨不塌", "spine_mid", _UP),
        Cue("骨盆保持水平，\n抬起腿的髋不外翻", "hip_mid"),
        Cue("支撑腿大腿收紧，膝不锁死", "left_knee", _UP),
        Cue("抬起腿向后蹬远", "right_ankle", (1.0, 0.0)),
    ),
    "triangle": (
        Cue("上手向天空延展", "right_wrist", _UP),
        Cue("胸腔朝侧上方打开，不向下扣", "spine_mid"),
        Cue("两侧侧腰等长，下侧腰不塌", "hip_mid"),
        Cue("下手轻放，\n重量不压在腿上", "left_wrist", _DOWN),
        Cue("前腿大腿上提，膝不超伸", "left_knee", _UP),
        Cue("双腿主动伸直踩实", "right_ankle", _DOWN),
    ),
    "parsvakonasana": (
        Cue("上臂过耳延展，\n与后腿成一条斜线", "right_wrist", _UP),
        Cue("下侧腰不塌，胸腔向上旋转", "spine_mid"),
        Cue("下手轻落地，不撑住体重", "left_wrist", _DOWN),
        Cue("前膝屈 90°，对准第二、三脚趾", "left_knee", _DOWN),
        Cue("后脚外缘压实，后腿延展", "right_ankle", _DOWN),
    ),
    "parsvottanasana": (
        Cue("从髋部折叠，背部保持延展", "hip_mid"),
        Cue("骨盆两侧保持等高，不向后腿侧歪", "hip_mid"),
        Cue("前腿膝盖可微屈，避免超伸", "left_knee"),
        Cue("后脚外缘压实，后腿主动伸直", "right_ankle", _DOWN),
        Cue("颈部放松，不主动去找腿", "nose", _DOWN),
    ),
    "tree": (
        Cue("双臂上举，肩胛仍下沉", "left_wrist", _UP),
        Cue("尾骨向下，肋骨不外翻", "hip_mid", _DOWN),
        Cue("抬起腿的膝向侧后方打开", "right_knee", (1.0, 0.2)),
        Cue("抬起脚放大腿内侧或小腿，\n不压在膝关节上", "right_ankle"),
        Cue("支撑脚均匀压地，脚趾放松", "left_foot_index", _DOWN),
    ),
    "anjaneyasana": (
        Cue("从腋窝向上延展，肩胛下沉", "left_wrist", _UP),
        Cue("尾骨向下卷，\n避免腰部塌陷代偿", "hip_mid", _DOWN),
        Cue("前膝在踝正上方，不超过脚尖", "left_knee", _DOWN),
        Cue("后侧髋前方向下延展", "right_knee", _DOWN),
        Cue("后脚背贴地", "right_foot_index", _DOWN),
    ),
    "ardha_hanumanasana": (
        Cue("从髋折叠，脊柱延展不蜷曲", "spine_mid"),
        Cue("骨盆保持水平，不向一侧歪", "hip_mid"),
        Cue("髋部落在后膝正上方", "right_knee", _UP),
        Cue("前腿膝可微屈，避免超伸", "left_knee"),
        Cue("前腿脚趾回勾", "left_foot_index", (-0.3, -0.95)),
    ),
    "pigeon": (
        Cue("骨盆保持中正，两侧坐骨等高", "hip_mid"),
        Cue("躯干直立，不向前塌", "spine_mid", _UP),
        Cue("前腿小腿角度按髋部开度调整，\n不强求横平", "left_knee"),
        Cue("后侧髋前方向下沉", "right_knee", _DOWN),
        Cue("后腿脚背贴地", "right_foot_index", _DOWN),
    ),
    "child": (
        Cue("额头落地或垫支撑，颈部放松", "nose", _DOWN),
        Cue("呼吸送到后背肋骨", "spine_mid"),
        Cue("坐骨向脚跟方向沉", "hip_mid", (1.0, 0.3)),
        Cue("双臂向前延展", "left_wrist", (-1.0, 0.0)),
        Cue("大脚趾相触，膝盖可分开", "left_foot_index"),
    ),
    "bridge": (
        Cue("肩胛内收压地，\n为胸腔打开腾空间", "left_shoulder", _DOWN),
        Cue("先卷尾骨再抬髋，\n脊柱逐节离地", "hip_mid", _UP),
        Cue("大腿内侧内旋，膝不外翻", "left_knee", _UP),
        Cue("脚跟在膝正下方，双脚与髋同宽", "left_ankle", _DOWN),
    ),
    "downdog": (
        Cue("双手压实，食指根部不抬起", "left_wrist", _DOWN),
        Cue("坐骨向上向后推，与手对抗", "hip_mid", _UP),
        Cue("膝可微屈，换取脊柱延展", "left_knee"),
        Cue("大腿内旋", "spine_mid"),
        Cue("脚跟朝地面下沉", "left_heel", _DOWN),
    ),
    "updog": (
        Cue("双手压实地面，十指张开", "left_wrist", _DOWN),
        Cue("肩胛下沉，远离耳朵", "left_shoulder", _DOWN),
        Cue("颈部延展，不过度后折", "nose", (-0.6, -0.8)),
        Cue("后弯分布在整段胸椎，\n不集中在下背", "spine_mid"),
        Cue("大腿主动离地", "left_knee", _UP),
        Cue("脚背压实地面", "left_foot_index", _DOWN),
    ),
    "plank": (
        Cue("后颈延展，\n视线落在双手前方地面", "nose", (-0.5, 0.85)),
        Cue("肩在腕正上方，手掌压实", "left_wrist", _DOWN),
        Cue("腹部收紧，髋不塌不翘", "hip_mid", _UP),
        Cue("大腿收紧", "left_knee", _UP),
        Cue("脚跟向后推", "left_heel", (1.0, 0.0)),
    ),
    "chaturanga": (
        Cue("肩不低于肘 —— \n宁可停高一点也不塌肩", "left_shoulder"),
        Cue("肘贴肋，前臂垂直地面", "left_elbow", _DOWN),
        Cue("全身成一条直线，髋不上翘", "hip_mid"),
        Cue("核心与大腿持续收紧", "left_knee", _UP),
        Cue("脚跟向后推", "left_heel", (1.0, 0.0)),
    ),
    "side_plank": (
        Cue("上侧肩向后打开，\n头颈保持中立", "right_wrist", _UP),
        Cue("支撑手在肩正下方，整掌压实", "left_wrist", _DOWN),
        Cue("髋部主动上提，不下坠", "hip_mid", _UP),
        Cue("双腿收紧叠放", "left_knee"),
        Cue("脚外缘或前脚掌压地", "left_ankle", _DOWN),
    ),
    "reverse_plank": (
        Cue("颈部后仰保持舒适范围，不硬压", "nose", (-0.5, -0.85)),
        Cue("双手在肩后下方，\n指尖朝脚的方向", "left_wrist", _DOWN),
        Cue("髋部主动上推，胸腔向上打开", "hip_mid", _UP),
        Cue("双腿收紧", "left_knee", _UP),
        Cue("脚背向下延展", "left_foot_index", _DOWN),
    ),
}


def annotated_cues(key: str) -> tuple[Cue, ...]:
    return ANNOTATED.get(key, ())


def _resolve(pts: np.ndarray, name: str) -> np.ndarray:
    """把锚点名解析成坐标，支持真实关键点和虚拟点。"""
    from .landmarks import _VIRTUAL

    if name in _VIRTUAL:
        return _VIRTUAL[name](pts)
    return pts[L.IDX[name]]


def _wrap(draw, text: str, font, max_width: float) -> list[str]:
    """按像素宽度折行。中文没有空格，所以逐字累加，同时尊重手写的 \\n。"""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for char in paragraph:
            if draw.textlength(current + char, font=font) <= max_width or not current:
                current += char
            else:
                lines.append(current)
                current = char
        lines.append(current)
    return lines


def _layout_labels(
    anchors: list[float], heights: list[float], top: float, bottom: float, gap: float
) -> list[float]:
    """给标签排纵向位置：初值取各自锚点的高度，再推开重叠部分。

    均匀分布看起来整齐，但会让引线大幅斜穿人像 —— 标签在顶部、锚点在底部。
    贴近锚点排，引线就短而近乎水平，图才读得顺。这是标注图的标准做法：
    先按锚点定位，再解冲突。
    """
    order = sorted(range(len(anchors)), key=lambda i: anchors[i])
    placed = [0.0] * len(anchors)

    # 自上而下：每个标签不得与上一个重叠
    cursor = top
    for i in order:
        half = heights[i] / 2.0
        y = max(anchors[i], cursor + half)
        placed[i] = y
        cursor = y + half + gap

    # 若挤出了下边界，自下而上回推
    overflow = cursor - gap - bottom
    if overflow > 0:
        cursor = bottom
        for i in reversed(order):
            half = heights[i] / 2.0
            y = min(placed[i], cursor - half)
            placed[i] = y
            cursor = y - half - gap
        # 再夹一次上边界，宁可轻微重叠也不要跑出画布
        cursor = top
        for i in order:
            half = heights[i] / 2.0
            placed[i] = max(placed[i], cursor + half)
            cursor = placed[i] + half + gap

    return placed


def _column_ratio(span_x: float, span_y: float) -> float:
    """文字栏占卡片宽度的比例。

    横向体式（四柱、平板、上犬）的人像受中间栏宽度限制，固定栏宽会把它挤得
    很小；这类体式的要点也更容易竖着排开，文字栏可以窄一些。
    """
    return 0.205 if span_x / max(span_y, 1e-6) > 1.8 else 0.255


def card_height_for(key: str, width: int) -> int:
    """按体式的长宽比推荐卡片高度。

    横向体式（上犬、平板）用固定方形卡时，人像被中间栏的宽度卡住，上下会空出
    一大片；纵向体式（三角式、山式）则相反，人像被高度卡住而左右留白。所以让
    高度跟着体式的长宽比走。
    """
    min_x, min_y, max_x, max_y, _ = _figure_extent(CANONICAL[key]())
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)
    ratio = _column_ratio(span_x, span_y)
    figure_w = width * (1.0 - 2 * ratio - 2 * 0.035)
    needed = figure_w * span_y / span_x + width * 0.22
    return int(min(max(needed, width * 0.50), width * 1.20))


def render_reference_card(
    key: str,
    width: int = 1180,
    height: int | None = None,
    font_path: str | None = None,
) -> "Image.Image":
    """渲染一张标准体式对照卡：线稿 + 锚定到部位的发力要点。

    要点分左右两栏排在人像两侧，按锚点的高低排序以减少引线交叉 —— 解剖图的
    常规做法。锚点在人像左半边的走左栏，右半边的走右栏，再做数量均衡。
    """
    from PIL import Image, ImageDraw

    from .grid import find_cjk_font
    from .poses import TEMPLATES_BY_KEY

    build = CANONICAL.get(key)
    if build is None:
        raise KeyError(f"没有 {key} 的标准骨架")
    template = TEMPLATES_BY_KEY.get(key)
    cue_list = annotated_cues(key)
    if height is None:
        height = card_height_for(key, width)

    path = font_path or find_cjk_font()
    def load(px: int):
        from PIL import ImageFont

        if path and Path(path).is_file():
            try:
                return ImageFont.truetype(path, px)
            except OSError:
                pass
        return ImageFont.load_default()

    font_title = load(max(20, round(height * 0.048)))
    font_sub = load(max(13, round(height * 0.024)))
    font_cue = load(max(14, round(height * 0.027)))
    font_foot = load(max(11, round(height * 0.019)))

    image = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(image)

    margin = round(width * 0.035)
    title = template.zh if template else key
    subtitle = template.en if template else ""
    draw.text((margin, margin), title, font=font_title, fill=INK)
    if subtitle:
        tw = draw.textlength(title, font=font_title)
        draw.text(
            (margin + tw + round(width * 0.014), margin + round(height * 0.020)),
            subtitle, font=font_sub, fill=INK_SOFT,
        )
    header_h = margin + round(height * 0.085)
    draw.line(
        [(margin, header_h), (width - margin, header_h)], fill=(224, 220, 212), width=2
    )

    # 人像占中间一栏，两侧留给要点。栏宽随体式长宽比调整，见 _column_ratio。
    _ex = _figure_extent(build())
    column_w = round(width * _column_ratio(_ex[2] - _ex[0], _ex[3] - _ex[1]))
    figure_box = (
        margin + column_w,
        header_h + round(height * 0.015),
        width - margin - column_w,
        height - round(height * 0.085),
    )
    figure_w = figure_box[2] - figure_box[0]
    figure_h = figure_box[3] - figure_box[1]

    pts = build()
    min_x, min_y, max_x, max_y, _ = _figure_extent(pts)
    span_x, span_y = max_x - min_x, max_y - min_y
    scale = min(figure_w / max(span_x, 1e-6), figure_h / max(span_y, 1e-6)) * 0.98
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    fx = (figure_box[0] + figure_box[2]) / 2.0
    fy = (figure_box[1] + figure_box[3]) / 2.0

    def place(p) -> tuple[float, float]:
        return (fx + (p[0] - cx) * scale, fy + (p[1] - cy) * scale)

    _draw_figure(draw, pts, place, scale)
    # PIL 的 width 只接受 int，numpy 的 float64 会报 TypeError。
    limb_w = max(4, round(float(L.torso_scale(pts)) * 0.115 * scale))

    # ---- 要点分栏 ----
    if cue_list:
        resolved = [(cue, place(_resolve(pts, cue.anchor))) for cue in cue_list]
        mid_x = (figure_box[0] + figure_box[2]) / 2.0
        left = [rc for rc in resolved if rc[1][0] <= mid_x]
        right = [rc for rc in resolved if rc[1][0] > mid_x]
        # 均衡两栏数量，把离中线最近的挪到人少的那栏。
        while len(left) - len(right) > 1:
            left.sort(key=lambda rc: -rc[1][0])
            right.append(left.pop(0))
        while len(right) - len(left) > 1:
            right.sort(key=lambda rc: rc[1][0])
            left.append(right.pop(0))
        left.sort(key=lambda rc: rc[1][1])
        right.sort(key=lambda rc: rc[1][1])

        text_w = column_w - round(width * 0.030)
        arrow_len = max(22, round(height * 0.062))

        line_h = round(font_cue.size * 1.42) if hasattr(font_cue, "size") else 20
        top = header_h + round(height * 0.045)
        bottom = height - round(height * 0.095)

        for side, items in (("left", left), ("right", right)):
            if not items:
                continue
            wrapped = [_wrap(draw, cue.text, font_cue, text_w) for cue, _ in items]
            heights = [line_h * len(w) for w in wrapped]
            ys = _layout_labels(
                [xy[1] for _, xy in items], heights, top, bottom, round(height * 0.035)
            )
            for i, (cue, anchor_xy) in enumerate(items):
                block_y = ys[i]
                lines = wrapped[i]
                block_h = heights[i]
                ty = block_y - block_h / 2.0

                if side == "left":
                    tx = margin
                    tip = (margin + text_w, block_y)
                else:
                    tx = width - margin - text_w
                    tip = (width - margin - text_w, block_y)

                # 引线：先横一小段离开文字，再直连锚点，读起来不乱。
                stub = tip[0] + (round(width * 0.012) if side == "left" else -round(width * 0.012))
                draw.line([tip, (stub, block_y)], fill=INK_SOFT, width=2)
                draw.line([(stub, block_y), anchor_xy], fill=INK_SOFT, width=2)
                draw.ellipse(
                    [anchor_xy[0] - 5, anchor_xy[1] - 5, anchor_xy[0] + 5, anchor_xy[1] + 5],
                    fill=ACCENT,
                )

                for j, line in enumerate(lines):
                    lw = draw.textlength(line, font=font_cue)
                    lx = tx if side == "left" else (width - margin - lw)
                    draw.text((lx, ty + j * line_h), line, font=font_cue, fill=INK)

                if cue.arrow is not None:
                    ux, uy = cue.arrow
                    norm = max((ux * ux + uy * uy) ** 0.5, 1e-6)
                    ux, uy = ux / norm, uy / norm
                    # 起点从关节往外挪开一点，箭杆不压在关节圆点上
                    ax = anchor_xy[0] + ux * arrow_len * 0.35
                    ay = anchor_xy[1] + uy * arrow_len * 0.35
                    head = arrow_len * 0.38
                    # 杆画到三角底边，不然三角会显得钝
                    sx, sy = ax + ux * (arrow_len - head), ay + uy * (arrow_len - head)
                    draw.line([(ax, ay), (sx, sy)], fill=ACCENT, width=max(2, round(limb_w * 0.5)))
                    px, py = -uy, ux
                    draw.polygon(
                        [
                            (ax + ux * arrow_len, ay + uy * arrow_len),
                            (sx + px * head * 0.42, sy + py * head * 0.42),
                            (sx - px * head * 0.42, sy - py * head * 0.42),
                        ],
                        fill=ACCENT,
                    )

    footer = "标准对位示意 · 通用教学口令，不针对个人，不能替代老师指导"
    draw.text((margin, height - round(height * 0.055)), footer, font=font_foot, fill=INK_SOFT)
    return image
