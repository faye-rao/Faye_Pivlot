"""遮脸判断的单元测试（``yoga_grid/heads.py``）。

遮脸是隐私功能，两个方向的错都实打实地发生过：

* **漏遮**（脸露着却没盖）—— 真正的事故。原因是原实现拿五个面部关键点
  （鼻、双眼、双嘴角）的**平均置信度**当门槛：侧脸时远侧的眼和嘴角被挡住、
  置信度接近 0，平均值被拖到门槛以下，而脸其实是清清楚楚露着的。
  第二处是半径跟着耳距/眼距缩：转头时这两个跨度被透视压短，而头的投影宽度
  几乎不变，于是圆只盖住五官（耳距落在 4~6 像素这个窄窗口里更直接放弃：
  ``radius < 6`` 返回 None，整帧不遮）。
  第三处只是难看但同源：正脸时倾角算出 180°，卡通脸整个倒过来画。
* **误遮**（背对镜头还盖一张卡通脸）—— 因为置信度根本不表示「你正对着我
  吗」。背身帧的面部关键点照样能是 1.0，任何单一置信度门槛都挡不住。

现在的判据是**手性**：MediaPipe 的 left/right 是解剖学左右，面朝镜头时左耳
在画面右侧，转身就翻到左侧。把「右耳→左耳」投到躯干朝上方向的垂线上取有
符号量，绕视线轴整体旋转（倒立、侧卧）时不变。侧脸落在死区里 —— 一律遮。

骨架全是手搭的像素坐标，几何推导写在注释里；不需要 cv2、mediapipe、
视频或模型文件。

    python tests/test_faces.py       或      python -m pytest tests/ -q
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid import heads  # noqa: E402
from yoga_grid import landmarks as L  # noqa: E402

W = H = 1000        # 测试用的方形画面，像素坐标 = 归一化坐标 × 1000
TORSO = 100.0       # 下面所有骨架的躯干长度（肩中点→髋中点）


# --------------------------------------------------------------------------
# 骨架构造工具
# --------------------------------------------------------------------------


def skeleton(**joints: tuple[float, float]) -> np.ndarray:
    """按关键点名搭一个像素坐标骨架（y 轴向下，「高」= y 小）。"""
    pts = np.zeros((L.N_LANDMARKS, 2), dtype=np.float64)
    for name, (x, y) in joints.items():
        pts[L.IDX[name]] = (x, y)
    return pts


def as_landmarks(
    pts: np.ndarray, vis: float = 0.95, **overrides: float
) -> np.ndarray:
    """像素骨架 -> (33, 3) 归一化 landmark 数组，可按名覆盖单点置信度。

    第三列是 ``min(visibility, presence)``，和 ``extract.py`` 喂进来的一致。
    """
    lm = np.zeros((L.N_LANDMARKS, 3), dtype=np.float64)
    lm[:, 0] = pts[:, 0] / W
    lm[:, 1] = pts[:, 1] / H
    lm[:, 2] = vis
    for name, value in overrides.items():
        lm[L.IDX[name], 2] = value
    return lm


def facing_camera() -> np.ndarray:
    """笔直站立、**面朝镜头**。躯干长 100，肩宽 40，两耳间距 30。

    面朝镜头 => 人的左侧出现在画面右侧，所以 ``left_*`` 的 x 更大。
    两耳间距 30 = 0.30 个躯干长，和真人（头宽≈15cm、躯干≈50cm）一致。
    """
    return skeleton(
        nose=(500, 352),
        left_eye=(506, 348), right_eye=(494, 348),
        left_ear=(515, 355), right_ear=(485, 355),
        mouth_left=(504, 362), mouth_right=(496, 362),
        left_shoulder=(520, 400), right_shoulder=(480, 400),
        left_elbow=(524, 450), right_elbow=(476, 450),
        left_wrist=(528, 500), right_wrist=(472, 500),
        left_hip=(515, 500), right_hip=(485, 500),
        left_knee=(515, 600), right_knee=(485, 600),
        left_ankle=(515, 700), right_ankle=(485, 700),
        left_heel=(515, 705), right_heel=(485, 705),
        left_foot_index=(515, 715), right_foot_index=(485, 715),
    )


def yaw(pts: np.ndarray, degrees: float) -> np.ndarray:
    """绕铅垂中线转 ``degrees``：所有 x 相对中线乘 cos。

    刚体绕铅垂轴旋转在正交投影下就是横向坐标乘 cos —— 0° 正对镜头、
    90° 正侧面（左右点在画面上重合）、180° 背对镜头（投影是中线镜像，
    标签不变，所以左右次序整体翻转）。

    这是个没有厚度的平板模型：真人转到侧面时鼻子会向前突出，这里不会。
    要用带鼻子的侧脸请用 ``profile()``。
    """
    mid = L.midpoint(pts, "left_hip", "right_hip")[0]
    out = pts.copy()
    out[:, 0] = mid + (out[:, 0] - mid) * math.cos(math.radians(degrees))
    return out


def profile() -> np.ndarray:
    """笔直站立、**正侧面**，脸朝画面右侧（镜头看到的是练习者左半边）。

    左右成对的点在画面上几乎重合（残差 ±2 像素），鼻子向朝向那侧突出。
    两耳间距只剩 4 像素 —— 手性的符号在这里没有意义，靠死区兜住。
    """
    return skeleton(
        nose=(515, 352),
        left_eye=(509, 348), right_eye=(507, 348),
        left_ear=(502, 355), right_ear=(498, 355),
        mouth_left=(512, 362), mouth_right=(511, 362),
        left_shoulder=(503, 400), right_shoulder=(497, 400),
        left_elbow=(504, 450), right_elbow=(496, 450),
        left_wrist=(505, 500), right_wrist=(495, 500),
        left_hip=(502, 500), right_hip=(498, 500),
        left_knee=(502, 600), right_knee=(498, 600),
        left_ankle=(502, 700), right_ankle=(498, 700),
        left_heel=(500, 705), right_heel=(496, 705),
        left_foot_index=(512, 715), right_foot_index=(508, 715),
    )


def rotate_about_hips(pts: np.ndarray, degrees: float) -> np.ndarray:
    """绕髋中点整体旋转（模拟倒立、侧卧等机位/体位）。"""
    hip = L.midpoint(pts, "left_hip", "right_hip")
    theta = math.radians(degrees)
    c, s = math.cos(theta), math.sin(theta)
    return (pts - hip) @ np.array([[c, s], [-s, c]]) + hip


def mean_face_confidence(lm: np.ndarray) -> float:
    """原实现用的信号：五个面部关键点的**平均**置信度，门槛 0.55。"""
    idx = [L.IDX[n] for n in ("nose", "left_eye", "right_eye", "mouth_left", "mouth_right")]
    return float(lm[idx, 2].mean())


def masked(lm: np.ndarray, width: int = W, height: int = H) -> bool:
    return heads.head_to_mask(lm, width, height) is not None


# --------------------------------------------------------------------------
# 三档：正脸 / 背身 / 中间的侧脸
# --------------------------------------------------------------------------


def test_facing_camera_is_masked():
    """正脸：手性 +0.30（两耳间距 30 / 躯干 100），必须遮。"""
    lm = as_landmarks(facing_camera())
    assert abs(heads.facing_score(lm, W, H) - 0.30) < 1e-9
    assert heads.facing(lm, W, H) == heads.FRONT

    head = heads.head_to_mask(lm, W, H)
    assert head is not None
    # 圆心落在头上（耳中点 500,355 往鼻子 500,352 拉三成）
    assert abs(head.center[0] - 500.0) < 1.0
    assert 350.0 < head.center[1] < 357.0
    # 半径要盖住整个头：0.30 个躯干长 = 30 像素，直径 60 > 头宽 30
    assert abs(head.radius - 0.30 * TORSO) < 1e-9
    # 头正立 => 倾角 0（原实现用 right_ear - left_ear，正脸时算出 180°，
    # 卡通脸整个倒过来画，嘴变成脑门上的一道倒弧）
    assert abs(head.angle) < 1e-9


def test_facing_away_is_not_masked():
    """背对镜头：手性 −0.30，不遮 —— 卡通脸不该贴在后脑勺上。"""
    lm = as_landmarks(yaw(facing_camera(), 180.0))
    assert abs(heads.facing_score(lm, W, H) + 0.30) < 1e-9
    assert heads.facing(lm, W, H) == heads.AWAY
    assert not masked(lm)


def test_profile_falls_in_the_deadband_and_is_still_masked():
    """侧脸：手性落在死区里（符号不可信），照样遮 —— 侧脸是露着的脸。"""
    lm = as_landmarks(profile())
    score = heads.facing_score(lm, W, H)
    assert abs(score) < heads.FACING_DEADBAND, score
    assert heads.facing(lm, W, H) == heads.UNCLEAR
    assert masked(lm)

    # 残差反过来（脸朝画面左侧）同样在死区里，同样要遮
    flipped = profile()
    flipped[:, 0] = 1000.0 - flipped[:, 0]
    lm2 = as_landmarks(flipped)
    assert abs(heads.facing_score(lm2, W, H)) < heads.FACING_DEADBAND
    assert masked(lm2)


def test_yaw_sweep_masks_everything_up_to_past_profile():
    """整条朝向曲线：只有明确转过侧面之后才不遮。

    手性 = 0.30 × cos(yaw)，死区 0.08 => 分界在 cos(yaw) = ±0.267，
    即 74.5° 和 105.5°。刻意把侧脸和「刚过侧面」都算进要遮的一侧。
    """
    base = facing_camera()
    for degrees in (0, 30, 60, 74, 90, 105):
        lm = as_landmarks(yaw(base, degrees))
        assert masked(lm), f"yaw {degrees}° 应该遮，实际没遮"
    for degrees in (106, 120, 150, 180):
        lm = as_landmarks(yaw(base, degrees))
        assert heads.facing(lm, W, H) == heads.AWAY, f"yaw {degrees}° 应判背身"
        assert not masked(lm), f"yaw {degrees}° 不该遮"


# --------------------------------------------------------------------------
# 两个方向的原始故障
# --------------------------------------------------------------------------


def test_occluded_far_side_landmarks_no_longer_suppress_the_mask():
    """漏遮回归：侧脸时远侧的眼和嘴角置信度接近 0。

    平均置信度 0.50 < 原门槛 0.55，原实现整帧不遮；而近侧点是 0.85，
    脸就在那里。改成取最高值后必须遮。
    """
    lm = as_landmarks(
        profile(),
        vis=0.9,
        nose=0.80, left_eye=0.85, mouth_left=0.75,   # 近侧：看得见
        right_eye=0.05, mouth_right=0.05,            # 远侧：被自己的脸挡住
        right_ear=0.25,
    )
    assert mean_face_confidence(lm) < 0.55        # 原门槛会在这里放弃
    assert heads.face_confidence(lm) >= 0.85      # 新信号看得见近侧的脸
    assert masked(lm)


def test_confident_back_view_is_still_rejected():
    """误遮回归：背身帧的面部关键点可以全是 1.0。

    任何「平均/最高置信度 ≥ 门槛」的判据都会在这里盖一张脸；
    手性不看置信度，照样判背身。
    """
    lm = as_landmarks(yaw(facing_camera(), 180.0), vis=1.0)
    assert mean_face_confidence(lm) == 1.0
    assert heads.face_confidence(lm) == 1.0
    assert not masked(lm)


def test_mask_is_not_shrunk_by_foreshortened_spans():
    """遮不全回归：转头时耳距被透视压短，而头的投影宽度几乎不变。

    转 60° 时耳距只剩 30 × cos60° = 15 像素，原实现半径 = 耳距 × 0.98
    = 14.7 像素 —— 只有该有的一半，圆盖住五官、发际线和轮廓还露在外面。
    真人的头从各个角度看宽度都在 15~19cm（近似圆柱），不该跟着耳距缩。
    """
    pts = yaw(facing_camera(), 60.0)
    ear_span = float(np.linalg.norm(pts[L.IDX["left_ear"]] - pts[L.IDX["right_ear"]]))
    assert abs(ear_span - 15.0) < 1e-9
    assert ear_span * 0.98 < 0.5 * (0.30 * TORSO) + 0.5   # 原实现只有一半大

    head = heads.head_to_mask(as_landmarks(pts), W, H)
    assert head is not None
    assert abs(head.radius - 0.30 * TORSO) < 1e-9   # 由躯干长度兜住，不由耳距决定


def test_blown_up_ear_landmarks_cannot_inflate_the_mask():
    """反方向兜底：耳朵点跳飞时半径被躯干长度封顶，不会盖掉半张图。"""
    pts = facing_camera()
    pts[L.IDX["left_ear"]] = (650, 355)
    pts[L.IDX["right_ear"]] = (350, 355)      # 耳距 300 = 3 个躯干长
    head = heads.head_to_mask(as_landmarks(pts), W, H)
    assert head is not None
    assert head.radius <= 0.45 * TORSO + 1e-9


# --------------------------------------------------------------------------
# 旋转不变性：倒立的正脸还是正脸
# --------------------------------------------------------------------------


def test_facing_is_invariant_to_whole_body_rotation():
    """整体旋转不改变朝向判断，也不改变手性的数值。

    只比 x 大小的做法在这里必错：把正脸骨架转 180°（肩倒立、后弯下腰
    到头朝下），左右耳在画面上的次序就颠倒了。
    """
    base = facing_camera()
    for degrees in (0, 30, 90, 150, 180, 270):
        lm = as_landmarks(rotate_about_hips(base, degrees))
        assert abs(heads.facing_score(lm, W, H) - 0.30) < 1e-9, degrees
        assert heads.facing(lm, W, H) == heads.FRONT, degrees
        assert masked(lm), degrees

    # 倒过来之后左耳的 x 确实比右耳小 —— 朴素的 x 比较会把它判成背身
    upside_down = as_landmarks(rotate_about_hips(base, 180.0))
    assert upside_down[L.IDX["left_ear"], 0] < upside_down[L.IDX["right_ear"], 0]


def test_upside_down_back_view_stays_unmasked():
    """反过来也要成立：倒立的背身不能因为左右次序翻转就被判成正脸。"""
    away = yaw(facing_camera(), 180.0)
    lm = as_landmarks(rotate_about_hips(away, 180.0))
    assert lm[L.IDX["left_ear"], 0] > lm[L.IDX["right_ear"], 0]   # x 次序像正脸
    assert heads.facing(lm, W, H) == heads.AWAY
    assert not masked(lm)


def test_mask_angle_follows_head_tilt():
    """倾角跟着头转：整体转 θ，卡通脸也转 θ。"""
    base = facing_camera()
    for degrees in (0.0, 25.0, -40.0, 90.0):
        head = heads.head_to_mask(as_landmarks(rotate_about_hips(base, degrees)), W, H)
        assert head is not None
        assert abs((head.angle - degrees + 180.0) % 360.0 - 180.0) < 1e-6, degrees


# --------------------------------------------------------------------------
# 没有头可遮的情况
# --------------------------------------------------------------------------


def test_no_landmarks_is_not_masked():
    assert heads.head_to_mask(None, W, H) is None


def test_head_not_in_frame_is_not_masked():
    """头不在画面里：面部点全是低置信度（presence 低），不遮。"""
    lm = as_landmarks(facing_camera(), vis=0.1)
    assert heads.face_confidence(lm) < heads.MIN_FACE_CONFIDENCE
    assert not masked(lm)


def test_head_fully_outside_the_frame_is_not_masked():
    """人整体移到画面左侧之外：圆整个出框，不用画。"""
    pts = facing_camera()
    pts[:, 0] -= 900.0        # 头部中心 -400，半径 30，整圆在 x < 0
    assert not masked(as_landmarks(pts))


def test_degenerate_landmarks_default_to_masking():
    """左右点对全不可信 => 手性算不出来 => 按「宁可多遮」处理。"""
    lm = as_landmarks(
        facing_camera(),
        left_ear=0.05, right_ear=0.05, left_eye=0.05, right_eye=0.05,
    )
    assert math.isnan(heads.facing_score(lm, W, H))
    assert heads.facing(lm, W, H) == heads.UNCLEAR
    assert masked(lm)          # 鼻/嘴角的置信度证明头在画面里


def test_decision_survives_non_square_frames():
    """16:9 画面下横纵尺度不同，手性的符号不变（各向异性缩放不改变定向）。"""
    lm = as_landmarks(facing_camera())
    for width, height in ((1920, 1080), (640, 360), (1000, 1000)):
        assert heads.facing(lm, width, height) == heads.FRONT, (width, height)
        assert masked(lm, width, height), (width, height)


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
