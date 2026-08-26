"""几何与打分逻辑的单元测试。

用手工搭的骨架，不依赖视频或模型文件 —— 这部分逻辑是整条流水线的判断依据，
出错会安静地选错帧，所以值得单独钉住。

    python -m pytest tests/ -q      或      python tests/test_geometry.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid import landmarks as L  # noqa: E402
from yoga_grid.cluster import agglomerate, pose_distance  # noqa: E402
from yoga_grid.poses import (  # noqa: E402
    _orientation_factor,
    _spine_up,
    match_pose,
    score_by_key,
)
from yoga_grid.landmarks import PoseView  # noqa: E402


# --------------------------------------------------------------------------
# 骨架构造工具
# --------------------------------------------------------------------------


def skeleton(**joints: tuple[float, float]) -> np.ndarray:
    """按关键点名搭一个像素坐标骨架，未指定的点放在原点。

    y 轴向下（图像坐标系），所以「高」= y 小。
    """
    pts = np.zeros((L.N_LANDMARKS, 2), dtype=np.float64)
    for name, (x, y) in joints.items():
        pts[L.IDX[name]] = (x, y)
    return pts


def standing() -> np.ndarray:
    """笔直站立、双臂垂于体侧，肩宽 40、躯干长 100。"""
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
    """战士二式：左腿为前腿，大腿近水平、小腿竖直（膝角约 98°），
    右腿伸直后展，双臂在肩高水平展开。"""
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


def rotate(pts: np.ndarray, degrees: float) -> np.ndarray:
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return pts @ np.array([[c, s], [-s, c]])


def as_landmarks(pts: np.ndarray, width: int, height: int, vis: float = 1.0) -> np.ndarray:
    """像素骨架 -> (33, 3) 归一化 landmark 数组，用于测 bbox / visibility。"""
    lm = np.zeros((L.N_LANDMARKS, 3), dtype=np.float64)
    lm[:, 0] = pts[:, 0] / width
    lm[:, 1] = pts[:, 1] / height
    lm[:, 2] = vis
    return lm


# --------------------------------------------------------------------------
# 角度
# --------------------------------------------------------------------------


def test_angle_at():
    origin = np.array([0.0, 0.0])
    assert abs(L.angle_at(origin, np.array([1.0, 0.0]), np.array([0.0, 1.0])) - 90) < 1e-6
    assert abs(L.angle_at(origin, np.array([1.0, 0.0]), np.array([-1.0, 0.0])) - 180) < 1e-6
    assert abs(L.angle_at(origin, np.array([1.0, 0.0]), np.array([1.0, 0.0]))) < 1e-6
    # 退化输入返回 nan，而不是抛异常或给出假角度
    assert math.isnan(L.angle_at(origin, origin, np.array([1.0, 0.0])))


def test_angle_to_vertical_and_horizontal():
    a = np.array([0.0, 0.0])
    assert abs(L.angle_to_vertical(a, np.array([0.0, 10.0]))) < 1e-6
    assert abs(L.angle_to_vertical(a, np.array([0.0, -10.0]))) < 1e-6   # 不区分朝向
    assert abs(L.angle_to_vertical(a, np.array([10.0, 0.0])) - 90) < 1e-6
    assert abs(L.angle_to_horizontal(a, np.array([10.0, 0.0]))) < 1e-6
    assert abs(L.angle_to_horizontal(a, np.array([0.0, 10.0])) - 90) < 1e-6


def test_torso_scale_and_normalize():
    pts = standing()
    assert abs(L.torso_scale(pts) - 100.0) < 1e-6
    norm = L.normalize(pts)
    # 髋中点归到原点
    assert np.allclose(L.midpoint(norm, "left_hip", "right_hip"), [0.0, 0.0], atol=1e-9)
    # 肩中点在原点正上方一个躯干长度处
    assert np.allclose(L.midpoint(norm, "left_shoulder", "right_shoulder"), [0.0, -1.0], atol=1e-9)


def test_normalize_is_scale_and_translation_invariant():
    pts = standing()
    moved = pts * 2.7 + np.array([500.0, -320.0])
    assert np.allclose(L.normalize(pts), L.normalize(moved), atol=1e-9)


def test_mirror_pose_swaps_sides():
    norm = L.normalize(standing())
    mirrored = L.mirror_pose(norm)
    # 镜像后左肩应落在原右肩的位置
    assert np.allclose(mirrored[L.IDX["left_shoulder"]],
                       norm[L.IDX["right_shoulder"]] * np.array([-1.0, 1.0]), atol=1e-9)
    # 镜像两次回到原状
    assert np.allclose(L.mirror_pose(mirrored), norm, atol=1e-9)


def test_mirror_table_is_an_involution():
    assert np.array_equal(L.MIRROR[L.MIRROR], np.arange(L.N_LANDMARKS))
    assert L.MIRROR[L.IDX["nose"]] == L.IDX["nose"]


# --------------------------------------------------------------------------
# 朝向
# --------------------------------------------------------------------------


def test_spine_up_is_signed():
    upright = PoseView(L.normalize(standing()))
    assert abs(_spine_up(upright) - 1.0) < 1e-6

    inverted = PoseView(L.normalize(rotate(standing(), 180)))
    assert abs(_spine_up(inverted) + 1.0) < 1e-6      # 倒立 -> -1

    sideways = PoseView(L.normalize(rotate(standing(), 90)))
    assert abs(_spine_up(sideways)) < 1e-6            # 水平 -> 0


def test_orientation_factor():
    assert _orientation_factor(1.0, 0.9, 1.06) == 1.0
    assert _orientation_factor(0.9, 0.9, 1.06) == 1.0
    assert _orientation_factor(-1.0, 0.9, 1.06) == 0.0    # 远超区间
    assert 0.0 < _orientation_factor(0.75, 0.9, 1.06) < 1.0
    assert _orientation_factor(float("nan"), 0.9, 1.06) == 1.0  # 算不出来不惩罚


# --------------------------------------------------------------------------
# 体式模板
# --------------------------------------------------------------------------


def test_standing_matches_mountain():
    match = match_pose(L.normalize(standing()))
    assert match is not None, "笔直站立应该匹配到山式"
    assert match.key == "mountain"
    assert match.score > 0.9


def test_inverted_standing_is_not_mountain():
    """倒立的身体不该是山式 —— 这正是有符号朝向门槛要挡住的情况。"""
    match = match_pose(L.normalize(rotate(standing(), 180)))
    assert match is None or match.key != "mountain"

    ungated = score_by_key(L.normalize(rotate(standing(), 180)), "mountain")
    assert ungated is not None
    assert ungated.orientation == 0.0
    assert ungated.score == 0.0


def test_warrior2_matches():
    """前腿屈约 90°、后腿伸直、双臂水平展开。"""
    norm = L.normalize(warrior2())
    match = match_pose(norm)
    assert match is not None
    assert match.key == "warrior2", f"期望 warrior2，实际 {match.key}"
    assert match.score > 0.9, f"正位分只有 {match.score:.3f}"

    # 双臂平举而非上举，必须明确低于战士二式，否则两者容易互相误判。
    w1 = score_by_key(norm, "warrior1")
    assert w1 is not None
    assert w1.score < match.score - 0.2, f"战士一式 {w1.score:.3f} 咬得太近"


def test_downdog_matches():
    """髋部最高，躯干与手臂成一线，四肢伸直。"""
    pts = skeleton(
        nose=(-100, -20),
        left_shoulder=(-78, -40), right_shoulder=(-82, -40),
        left_elbow=(-119, -20), right_elbow=(-121, -20),
        left_wrist=(-160, 0), right_wrist=(-160, 0),
        left_hip=(2, -140), right_hip=(-2, -140),
        left_knee=(81, -70), right_knee=(79, -70),
        left_ankle=(160, 0), right_ankle=(160, 0),
        left_heel=(165, 5), right_heel=(165, 5),
        left_foot_index=(185, 0), right_foot_index=(185, 0),
    )
    match = match_pose(L.normalize(pts))
    assert match is not None
    assert match.key == "downdog", f"期望 downdog，实际 {match.key}"


def test_template_is_side_agnostic():
    """同一个体式左右两个方向应拿到相近的分数。"""
    pts = warrior2()
    left_version = score_by_key(L.normalize(pts), "warrior2")
    mirrored = L.mirror_pose(L.normalize(pts))
    right_version = score_by_key(mirrored, "warrior2")
    assert left_version is not None and right_version is not None
    assert abs(left_version.score - right_version.score) < 1e-6


# --------------------------------------------------------------------------
# 聚类
# --------------------------------------------------------------------------


def test_pose_distance_mirror_invariance():
    norm = L.normalize(standing())
    mirrored = L.mirror_pose(norm)
    assert pose_distance(norm, mirrored, mirror_same=True) < 1e-9
    # 站立姿势本身左右对称，换个不对称的姿势才测得出差别
    asym = norm.copy()
    asym[L.IDX["left_wrist"]] += np.array([0.0, -1.4])
    assert pose_distance(asym, L.mirror_pose(asym), mirror_same=False) > 0.05
    assert pose_distance(asym, L.mirror_pose(asym), mirror_same=True) < 1e-9


def test_pose_distance_grows_with_difference():
    a = L.normalize(standing())
    b = L.normalize(rotate(standing(), 45))
    c = L.normalize(rotate(standing(), 120))
    assert pose_distance(a, a) < 1e-9
    assert pose_distance(a, b) < pose_distance(a, c)


def test_agglomerate_finds_expected_groups():
    # 三对样本，对内距离 0.1，对间距离 5.0
    d = np.array([
        [0.0, 0.1, 5.0, 5.0, 5.0, 5.0],
        [0.1, 0.0, 5.0, 5.0, 5.0, 5.0],
        [5.0, 5.0, 0.0, 0.1, 5.0, 5.0],
        [5.0, 5.0, 0.1, 0.0, 5.0, 5.0],
        [5.0, 5.0, 5.0, 5.0, 0.0, 0.1],
        [5.0, 5.0, 5.0, 5.0, 0.1, 0.0],
    ])
    labels = agglomerate(d, threshold=0.35)
    assert len(set(labels)) == 3
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[4] == labels[5]
    assert labels[0] != labels[2]
    # 簇号按簇内最早样本排序，读 JSON 时才直观
    assert labels == [0, 0, 1, 1, 2, 2]

    # 阈值大到足以吞掉一切时并成一簇
    assert len(set(agglomerate(d, threshold=99.0))) == 1
    # 阈值为 0 时谁也不合并
    assert len(set(agglomerate(d, threshold=0.0))) == 6


def test_agglomerate_handles_empty():
    assert agglomerate(np.zeros((0, 0)), 0.35) == []


# --------------------------------------------------------------------------
# 外框与可见度
# --------------------------------------------------------------------------


def test_bbox_covers_head_and_feet():
    pts = standing()
    lm = as_landmarks(pts, 1000, 1000)
    x0, y0, x1, y1 = L.bbox_norm(lm)
    # 上边界应到耳/鼻，下边界应到脚尖
    assert y0 <= pts[L.IDX["nose"]][1] / 1000 + 1e-9
    assert y1 >= pts[L.IDX["left_foot_index"]][1] / 1000 - 1e-9
    assert x0 <= pts[L.IDX["left_wrist"]][0] / 1000 + 1e-9
    assert x1 >= pts[L.IDX["right_wrist"]][0] / 1000 - 1e-9


def test_bbox_falls_back_when_nothing_visible():
    """可信点不足时不该崩，也不该返回退化外框。"""
    lm = as_landmarks(standing(), 1000, 1000, vis=0.01)
    x0, y0, x1, y1 = L.bbox_norm(lm, vis_thresh=0.5)
    assert x1 > x0 and y1 > y0


def test_mean_visibility():
    assert abs(L.mean_visibility(as_landmarks(standing(), 100, 100, vis=0.8)) - 0.8) < 1e-9


# --------------------------------------------------------------------------
# 打分分量
# --------------------------------------------------------------------------


def test_framing_score_penalises_clipping():
    from yoga_grid.score import _framing_score

    full = _framing_score((0.25, 0.05, 0.75, 0.95))     # 完整且够大
    clipped = _framing_score((-0.20, 0.05, 0.75, 0.95))  # 左侧切掉一块
    assert full > clipped
    assert 0.0 <= clipped <= 1.0
    tiny = _framing_score((0.48, 0.48, 0.52, 0.52))      # 人太小太远
    assert tiny < full


def test_sharpness_ranks():
    from yoga_grid.score import _sharpness_ranks

    assert _sharpness_ranks([]) == []
    assert _sharpness_ranks([5.0]) == [1.0]
    ranks = _sharpness_ranks([10.0, 30.0, 20.0])
    assert ranks[0] == 0.0 and ranks[1] == 1.0 and 0.0 < ranks[2] < 1.0


def _run_all() -> int:
    tests = [
        (name, fn)
        for name, fn in sorted(globals().items())
        if name.startswith("test_") and callable(fn)
    ]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
