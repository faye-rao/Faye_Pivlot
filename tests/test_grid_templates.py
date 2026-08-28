"""体式模板之间的区分度测试。

模板集越大，最容易出的问题不是「认不出」，而是**A 体式被 B 的模板认走**。
这类错误在真实视频里已经发生过好几次：

* 旋转 180° 的身体被判成山式（缺有符号的朝向门槛）；
* 上犬式被三角伸展式认走（三角式缺站姿约束，两者躯干倾角都在 0.5 附近）；
* 平板式在战士三式模板上拿到 0.8（战士三式缺站姿约束）；
* 双臂上举的战士一式被判成战士二式（战士二式只查两腕连线水平，
  而上举时两腕同样等高）；
* 侧板式被平板式认走（平板式只查主侧那只手，侧板式撑地的手蒙混过关）；
* 反板式被平板式认走（2D 剪影近乎镜像，缺头部朝向这一对称判据）。
* **完全平直的身体被上犬式认走**（真实视频，2026-08）。上犬式的「躯干后弯」
  原先取 150±22/slack 38 —— 衰减到 0 的位置在 210°，于是一个 175.5° 的
  平直身体还能拿 0.91 分，8 帧平板式被认成上犬式 0.98，九宫格里同一个姿势
  出现了两次（两个不同的名字，所以合并逻辑也拦不住）。
* **蹲姿被鸽子式认走，排除鸽子式后又被半神猴式认走**（真实视频，2026-08）。
  蹲姿两条腿都折到约 64°，鸽子式六项里只有「后腿向后伸直」正确地判 0 分，
  另外五项被低髋位满足，于是以 0.77 越过门槛。这一条和上面几条性质不同：
  **蹲姿在模板集里根本没有对应模板**，所以补判据补不到它自己身上，只能给
  被它蒙混的模板加定义性门槛（见 `poses.Gate`）。它也暴露了这张网的盲区
  —— 只测模板互相之间，从不测「库里没有的体式」，见下面的负例测试。

每一条都是先在真实数据上出错、再回头补判据。这个测试把补上的判据钉死：
遍历所有标准骨架，断言每具骨架的最高分模板就是它自己。加模板或调容差时
若破坏了任何一对区分，这里立刻报错，并指出被谁抢走了。

    python tests/test_templates.py      或      python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid.reference import CANONICAL, skeleton  # noqa: E402
from yoga_grid import landmarks as L  # noqa: E402
from yoga_grid.poses import TEMPLATES, TEMPLATES_BY_KEY, match_pose, score_by_key  # noqa: E402


def _canonical(key: str):
    """CANONICAL 存的是构造函数，取出实际骨架。"""
    value = CANONICAL[key]
    return value() if callable(value) else value


def test_every_template_has_a_canonical_skeleton():
    """新加模板必须同时提供标准骨架，否则它的区分度无人把关。"""
    missing = sorted({t.key for t in TEMPLATES} - set(CANONICAL))
    assert not missing, f"以下模板缺标准骨架：{missing}（请在 yoga_grid/reference.py 里补上）"

    unknown = sorted(set(CANONICAL) - {t.key for t in TEMPLATES})
    assert not unknown, f"以下骨架没有对应模板：{unknown}"


def test_each_skeleton_matches_its_own_template():
    """每具标准骨架的最高分模板必须是它自己。"""
    problems: list[str] = []
    for key, build in CANONICAL.items():
        norm = L.normalize(build())
        match = match_pose(norm)
        if match is None:
            own = score_by_key(norm, key)
            problems.append(
                f"{key}: 没有任何模板过线"
                f"（自身模板得分 {own.score:.2f}，门槛 {TEMPLATES_BY_KEY[key].min_score:.2f}）"
            )
        elif match.key != key:
            own = score_by_key(norm, key)
            problems.append(
                f"{key}: 被 {match.key} 认走"
                f"（{match.key} {match.score:.2f} vs 自身 {own.score:.2f}）"
            )
    assert not problems, "模板区分度回归：\n  " + "\n  ".join(problems)


def test_own_template_clears_its_threshold():
    """标准骨架在自身模板上要明显高于门槛，留出真实数据的抖动余量。"""
    problems: list[str] = []
    for key, build in CANONICAL.items():
        template = TEMPLATES_BY_KEY[key]
        match = score_by_key(L.normalize(build()), key)
        assert match is not None
        if match.score < template.min_score + 0.15:
            problems.append(
                f"{key}: {match.score:.2f}，门槛 {template.min_score:.2f}，余量不足 0.15"
            )
    assert not problems, "自身得分余量不足：\n  " + "\n  ".join(problems)


def test_orientation_gate_rejects_wrong_facing():
    """朝向门槛必须挡住方向明显不符的体式。

    抽查三对方向相反的体式：站姿 vs 倒垂、俯卧 vs 仰卧、髋高 vs 肩高。
    """
    pairs = [
        ("mountain", "uttanasana"),      # 直立 vs 向下倒垂
        ("downdog", "updog"),            # 髋最高 vs 肩最高
        ("plank", "reverse_plank"),      # 俯卧 vs 仰卧
    ]
    for a, b in pairs:
        norm_a = L.normalize(CANONICAL[a]())
        norm_b = L.normalize(CANONICAL[b]())
        score_a_on_b = score_by_key(norm_b, a)
        score_b_on_a = score_by_key(norm_a, b)
        own_a = score_by_key(norm_a, a)
        own_b = score_by_key(norm_b, b)
        assert score_a_on_b.score < own_b.score, f"{a} 在 {b} 的骨架上分数过高"
        assert score_b_on_a.score < own_a.score, f"{b} 在 {a} 的骨架上分数过高"


def test_standing_templates_reject_prone_bodies():
    """允许躯干大幅倾斜的站姿体式，必须靠站姿约束挡住俯卧体式。

    这是「上犬式被三角伸展式认走」那个真实 bug 的回归测试 —— 光靠朝向门槛
    不够，因为上犬式和三角伸展式的躯干竖直分量都在 0.5 附近。
    """
    prone = ["updog", "plank", "chaturanga"]
    standing = ["triangle", "parsvakonasana", "warrior3"]
    for prone_key in prone:
        norm = L.normalize(CANONICAL[prone_key]())
        for standing_key in standing:
            match = score_by_key(norm, standing_key)
            threshold = TEMPLATES_BY_KEY[standing_key].min_score
            assert match.score < threshold, (
                f"{standing_key} 在 {prone_key}（俯卧）的骨架上拿到 {match.score:.2f}，"
                f"超过门槛 {threshold:.2f}"
            )


def test_every_pose_has_cues_and_annotations():
    """每个体式都要有发力要点，且要点锚点必须能解析。

    锚点写错（拼错关键点名、用了不存在的虚拟点）在渲染时才会炸，而渲染只在
    出图时才跑 —— 放到测试里，改数据就能立刻发现。
    """
    from yoga_grid.landmarks import _VIRTUAL
    from yoga_grid.reference import ANNOTATED, CUES

    keys = set(CANONICAL)
    assert not keys - set(CUES), f"缺纯文字要点：{sorted(keys - set(CUES))}"
    assert not keys - set(ANNOTATED), f"缺带锚点要点：{sorted(keys - set(ANNOTATED))}"

    valid = set(L.NAMES) | set(_VIRTUAL)
    for key, cue_list in ANNOTATED.items():
        for cue in cue_list:
            assert cue.anchor in valid, f"{key}：锚点 {cue.anchor!r} 不是有效关键点"
            assert cue.text.strip(), f"{key}：要点文字为空"


def test_restore_landmarks_handles_both_formats():
    """scores.json 的两种 landmarks 格式都要正确还原。

    回归测试：存储格式从「33×2 归一化骨架」换成「33×3 原始关键点」后，旧文件
    被当成新格式读，会把已归一化的坐标再乘一次画面宽高 —— 几何全错而且不报错。
    """
    from yoga_grid.report import _restore_landmarks

    pts = CANONICAL["warrior2"]()
    expected = L.normalize(pts)

    # 当前格式：归一化 x/y + 置信度，还原后的骨架应与直接归一化一致
    raw = [[float(x) / 1000.0, float(y) / 1000.0, 1.0] for x, y in pts]
    lm, norm = _restore_landmarks(raw, 1000, 1000)
    assert lm is not None and lm.shape == (L.N_LANDMARKS, 3)
    assert norm is not None
    import numpy as np

    assert np.allclose(norm, expected, atol=1e-6)

    # 早期格式：本身就是归一化骨架，必须原样采用，不能再缩放一次
    legacy = [[float(x), float(y)] for x, y in expected]
    lm2, norm2 = _restore_landmarks(legacy, 1920, 1080)
    assert lm2 is None, "早期格式没有置信度，lm 应留空以跳过遮脸"
    assert np.allclose(norm2, expected, atol=1e-6)

    assert _restore_landmarks(None, 100, 100) == (None, None)
    assert _restore_landmarks([], 100, 100) == (None, None)

    # 33×2 但内容其实是原始归一化坐标（值全在 [0,1]、髋中点远离原点）必须被拒收：
    # 误当成骨架会让所有姿势的距离塌到极小、聚成一簇 —— 结果全错却不报错。
    bogus = [[float(x) / 1000.0, float(y) / 1000.0] for x, y in pts]
    assert _restore_landmarks(bogus, 1000, 1000) == (None, None)

    # 形状不对的也要拒收
    assert _restore_landmarks([[0.0, 0.0]], 100, 100) == (None, None)


def test_every_pose_renders():
    """每个体式的线稿和对照卡都要能渲染出来，不抛异常。"""
    from yoga_grid.reference import render_pose, render_reference_card

    for key in CANONICAL:
        art = render_pose(key, 160)
        assert art.size == (160, 160)
        card = render_reference_card(key, width=700)
        assert card.size[0] == 700 and card.size[1] > 200


def test_mirrored_skeleton_scores_the_same():
    """左右镜像的同一体式，在自身模板上得分必须一致。

    非对称模板会左右各试一遍取高分，镜像后分数不该变。
    """
    for key, build in CANONICAL.items():
        norm = L.normalize(build())
        mirrored = L.mirror_pose(norm)
        a = score_by_key(norm, key)
        b = score_by_key(mirrored, key)
        assert abs(a.score - b.score) < 1e-6, (
            f"{key} 镜像后得分从 {a.score:.4f} 变为 {b.score:.4f}"
        )


# --------------------------------------------------------------------------
# 负例：图库里**没有**模板的体式。这张网原先只测模板互相之间，于是蹲姿
# （Malasana，没有模板）被鸽子式认走 0.77、排除鸽子式后又被半神猴式认走
# 0.71，两次都溜过去了。没有模板的体式必须回到「未识别体式」。
# --------------------------------------------------------------------------

#: 手搭蹲姿：躯干 100，脚掌落地，膝深屈外开，髋沉到接近脚跟。小腿近乎竖直
#: （膝在踝正上方偏前），大腿向后上，膝角约 64 度 —— 真实视频里量到 38 度，
#: 同为「两腿都折着」，这里取偏保守的 64 度，门槛要连它一起拦住才算过关。
_SQUAT_LEGS = dict(
    left_hip=(-68, 40), right_hip=(-52, 43),
    left_knee=(25, 0), right_knee=(40, 4),
    left_ankle=(20, 100), right_ankle=(36, 104),
    left_heel=(6, 104), right_heel=(22, 108),
    left_foot_index=(48, 106), right_foot_index=(64, 110),
)

#: 躯干直立的蹲姿 —— 朝向门槛能拦住半神猴式，但拦不住鸽子式。
SQUAT_UPRIGHT = skeleton(
    nose=(-58, -70), left_ear=(-48, -60), right_ear=(-40, -58),
    left_eye=(-54, -72), right_eye=(-46, -70),
    mouth_left=(-56, -58), mouth_right=(-48, -56),
    left_shoulder=(-56, -55), right_shoulder=(-40, -52),
    left_elbow=(-40, -14), right_elbow=(-26, -12),
    left_wrist=(-44, 8), right_wrist=(-36, 10),
    **_SQUAT_LEGS,
)

#: 躯干前倾的蹲姿 —— 这一版落进半神猴式的朝向区间，是真实视频里被认走的那种。
SQUAT_FOLDED = skeleton(
    nose=(-30, -46), left_ear=(-24, -36), right_ear=(-16, -34),
    left_eye=(-26, -48), right_eye=(-18, -46),
    mouth_left=(-28, -34), mouth_right=(-20, -32),
    left_shoulder=(-34, -28), right_shoulder=(-18, -25),
    left_elbow=(-26, 6), right_elbow=(-12, 8),
    left_wrist=(-38, 12), right_wrist=(-30, 14),
    **_SQUAT_LEGS,
)

# --------------------------------------------------------------------------
# 2026-08 的第二批负例：两支真实练习视频（38 分钟 + 16 分钟，共 493 个候选帧），
# 用户逐格核对九宫格后点名的误判。每一具骨架都按**实测不变量**手搭，
# 括号里的数值就是从那两支视频量出来的，不是想象的。
#
# 这一批的共同结构和蹲姿完全一样，只是换了七个体式：模板里唯一正确否决它的
# 那一项被其余各项投票淹没。所以修法也一样 —— 给被蒙混的模板补定义性门槛。
#
# 补门槛有个连锁反应，这一批把它暴露得很清楚：拦住一个模板，那些帧会去找
# 几何上次像的模板。压脚背从树式 → 未识别用了 2 轮，站立伸展从站立前屈式
# → 金字塔式 → 下犬式 → 战士三式 → 未识别用了 4 轮。所以**这些测试断言的是
# 「没有任何模板认领它」，而不是「某个模板拒绝了它」** —— 后者会漏掉下一站。
# --------------------------------------------------------------------------

#: 幻椅式（Utkatasana）—— 库里没有。双脚并拢（踝距实测 0.01~0.07）、双膝都屈
#: 117~139°、站姿（踝低于髋 1.05~1.36）、双臂上举、躯干竖直。
#: 曾被新月式认走 0.87~1.00：「前膝屈 90°」和「后膝屈曲跪地」被两条都屈的膝
#: 同时满足，弓步却没要求双脚前后分开。
CHAIR = skeleton(
    nose=(-6, -130), left_ear=(-14, -124), right_ear=(2, -124),
    left_eye=(-10, -132), right_eye=(-2, -132),
    mouth_left=(-10, -122), mouth_right=(-2, -122),
    left_shoulder=(-20, -100), right_shoulder=(20, -100),
    left_elbow=(-18, -150), right_elbow=(18, -150),
    left_wrist=(-14, -195), right_wrist=(14, -195),
    left_hip=(-15, 0), right_hip=(15, 0),
    left_knee=(-48, 60), right_knee=(-42, 60),
    left_ankle=(-18, 130), right_ankle=(-12, 130),
    left_heel=(-8, 134), right_heel=(-2, 134),
    left_foot_index=(-46, 136), right_foot_index=(-40, 136),
)

#: 压脚背（跪坐在脚跟上、脚趾回勾）—— 库里没有。两膝都折到 21~39°、
#: 两踝并在髋下 0.31~0.48、躯干竖直、手在胸前。
#: 曾**把树式整簇占掉**（16 帧，0.72~0.78）：五项里只有「支撑腿伸直」判 0 分，
#: 两膝都折着自然满足「抬起腿屈膝」，两踝并拢自然满足「抬起脚贴支撑腿」。
TOE_SQUAT = skeleton(
    nose=(-8, -130), left_ear=(-16, -124), right_ear=(0, -124),
    left_eye=(-12, -132), right_eye=(-4, -132),
    mouth_left=(-12, -122), mouth_right=(-4, -122),
    left_shoulder=(-20, -100), right_shoulder=(20, -100),
    left_elbow=(-35, -60), right_elbow=(35, -60),
    left_wrist=(-8, -45), right_wrist=(8, -45),
    left_hip=(-15, 0), right_hip=(15, 0),
    left_knee=(-62, 25), right_knee=(-58, 25),
    left_ankle=(-2, 30), right_ankle=(2, 30),
    left_heel=(6, 26), right_heel=(10, 26),
    left_foot_index=(-30, 44), right_foot_index=(-26, 44),
)

#: 四足跪姿/桌面式（Bharmanasana）—— 库里没有，而它是猫牛式、鸟狗式的起始位，
#: 出现频率极高：一支 38 分钟视频里 46 帧。膝压在髋正下方（膝低于髋 0.87~0.98）、
#: 膝角 67~88°、身体成一直线只有 126~146°。
#: 曾被平板式认走 0.62~0.75，补上门槛后依次落到四柱支撑式 0.80、桥式 0.69、
#: 婴儿式 0.62 —— 一个高频体式缺模板，会把整条俯卧支撑家族轮流污染一遍。
TABLE_TOP = skeleton(
    nose=(-140, 10), left_ear=(-128, 4), right_ear=(-126, 12),
    left_eye=(-136, 6), right_eye=(-134, 14),
    mouth_left=(-134, 18), mouth_right=(-132, 22),
    left_shoulder=(-99, -18), right_shoulder=(-100, -2),
    left_elbow=(-101, 32), right_elbow=(-102, 44),
    left_wrist=(-102, 85), right_wrist=(-103, 88),
    left_hip=(-1, -8), right_hip=(1, 8),
    left_knee=(-4, 93), right_knee=(4, 93),
    left_ankle=(53, 88), right_ankle=(57, 88),
    left_heel=(62, 90), right_heel=(66, 90),
    left_foot_index=(74, 94), right_foot_index=(78, 94),
)

#: 站立伸展/展背（半程前屈）—— 库里没有。腿直、双脚并拢，但躯干只折到
#: 略过水平（spine_up 实测 -0.35~-0.49，满程前屈是 -0.85 上下），
#: 头还在髋下 0.50~0.70（满程能到 0.95 以上）。
#: 曾被站立前屈式认走 0.91（六项里只有「躯干向下倒垂」失分）。
HALF_FOLD = skeleton(
    nose=(-120, 55), left_ear=(-108, 46), right_ear=(-106, 54),
    left_eye=(-116, 48), right_eye=(-114, 56),
    mouth_left=(-114, 62), mouth_right=(-112, 66),
    left_shoulder=(-91, 33), right_shoulder=(-91, 49),
    left_elbow=(-66, 72), right_elbow=(-66, 78),
    left_wrist=(-40, 110), right_wrist=(-40, 116),
    left_hip=(-8, -8), right_hip=(8, 8),
    left_knee=(-6, 80), right_knee=(6, 80),
    left_ankle=(-3, 159), right_ankle=(3, 159),
    left_heel=(-3, 164), right_heel=(3, 164),
    left_foot_index=(-16, 170), right_foot_index=(-10, 170),
)

#: 俯卧撑起上身看手机 —— 用户原话。不是体式，但摄像机在录，它就要参与识别。
#: 双腿都伸直贴地（168°/178°）、髋贴地（踝低于髋 0.22）、双肘折到 8°和 57°。
#: 曾被鸽子式认走 0.83（六项里只有「前腿屈膝外旋」判 0 分），
#: 补上鸽子式门槛后落到半神猴式 0.80、再落到上犬式 0.78。
PRONE_PHONE = skeleton(
    nose=(-115, -40), left_ear=(-104, -46), right_ear=(-102, -38),
    left_eye=(-111, -44), right_eye=(-109, -36),
    mouth_left=(-109, -32), mouth_right=(-107, -28),
    left_shoulder=(-79, -69), right_shoulder=(-79, -53),
    left_elbow=(-105, 26), right_elbow=(-105, 34),
    left_wrist=(-95, -45), right_wrist=(-93, -39),
    left_hip=(-1, -8), right_hip=(1, 8),
    left_knee=(85, 8), right_knee=(85, 16),
    left_ankle=(170, 18), right_ankle=(170, 26),
    left_heel=(178, 20), right_heel=(178, 28),
    left_foot_index=(190, 30), right_foot_index=(190, 38),
)

#: 随意站着 —— 用户原话「随意站着手臂没有下垂不是山式」。
#: 两膝 171°/148°、双臂在体侧偏上（腕低于肩 0.66，山式是 0.95）。
#: 曾被山式认走 0.87。它是 CLAUDE.md 那条「站姿在几何上就是一个完美的平板
#: 支撑」的另一副面孔：山式和随意站着的几何差别只有腿直不直、手臂在哪。
#: 两膝均值 159° 落在 178±8 的衰减区里还能拿 0.52 分 —— 见 _worse_leg_extended。
CASUAL_STAND = skeleton(
    nose=(0, -130), left_ear=(-10, -125), right_ear=(10, -125),
    left_eye=(-4, -132), right_eye=(4, -132),
    mouth_left=(-4, -122), mouth_right=(4, -122),
    left_shoulder=(-20, -100), right_shoulder=(20, -100),
    left_elbow=(-26, -60), right_elbow=(26, -60),
    left_wrist=(-30, -34), right_wrist=(30, -34),
    left_hip=(-15, 0), right_hip=(15, 0),
    left_knee=(-20, 72), right_knee=(34, 70),
    left_ankle=(-13, 143), right_ankle=(14, 140),
    left_heel=(-13, 148), right_heel=(14, 145),
    left_foot_index=(-26, 154), right_foot_index=(1, 151),
)

#: 双手撑地的低弓步（体位串联里的过渡）—— 库里没有。前膝屈 79~86°、
#: 后膝 131~139°、双脚前后拉开 2.26~2.37、髋沉到踝上 0.61~0.66、双手撑地。
#: 曾被鸽子式认走 0.92（前膝角正落在「前腿屈膝外旋 85±30」的正中间），
#: 之后依次被半神猴式 0.85、反板式 0.73、侧角伸展式 0.72、上犬式 0.70 接手。
#: 五轮才拦干净，是这一批里最难的一个。
LUNGE_HANDS_DOWN = skeleton(
    nose=(-115, -85), left_ear=(-104, -80), right_ear=(-102, -72),
    left_eye=(-111, -88), right_eye=(-109, -80),
    mouth_left=(-109, -76), mouth_right=(-107, -70),
    left_shoulder=(-85, -59), right_shoulder=(-77, -59),
    left_elbow=(-92, 0), right_elbow=(-84, 0),
    left_wrist=(-95, 57), right_wrist=(-87, 57),
    left_hip=(-8, -8), right_hip=(8, 8),
    left_knee=(-85, -5), right_knee=(70, 0),
    left_ankle=(-85, 64), right_ankle=(130, 62),
    left_heel=(-92, 68), right_heel=(140, 66),
    left_foot_index=(-70, 70), right_foot_index=(112, 70),
)

#: 现在**有**模板的四个体式，骨架按同一批实测不变量手搭。
#:
#: 它们原先在 NON_TEMPLATE_POSES 里，断言「没有模板会认领」；补上模板之后
#: 断言反过来 —— 必须认出、而且认成对的那一个。这比只用 reference.py 的标准
#: 骨架检验强得多：**标准骨架是我按目标值搭的，实测骨架不是。** 目标值取错了
#: 位置，标准骨架照样满分，只有实测骨架会红。
NOW_TEMPLATED = {
    "幻椅式": (CHAIR, "chair"),
    "压脚背": (TOE_SQUAT, "toe_squat"),
    "四足跪姿": (TABLE_TOP, "table_top"),
    "展背式（半程前屈）": (HALF_FOLD, "ardha_uttanasana"),
}

#: 仍然没有模板的。蹲姿和「随意站着」是刻意不补的：前者和压脚背差别很小
#: （都是两腿都折着、躯干竖直、双脚并拢），补上会让两者互抢；后者根本不是
#: 体式。俯卧看手机同理。双手撑地低弓步是**过渡动作**，而且实测那 13 帧
#: 分成两种形状（前膝 79~86 / 后膝 131~139，与前膝 117~120 / 后膝 96~99），
#: 一个模板盖不住；认出来也不会改变选帧，因为簇间排序用的是画质和保持时长、
#: 不是正位分。
#: 四足跪姿单腿后伸（鸟狗式的腿那一半）—— 库里没有。一膝跪地（88°、落地 0.92）、
#: 一腿向后伸直（172°）、双手撑地、身体大致水平。
#:
#: 这一具是整批里最难拦的：它被**六个**模板依次认领过 ——
#: 半神猴式 1.00 → 平板式 0.93 → 侧角伸展式 0.72 → 上犬式 0.72 → 侧板式 0.67 →
#: 未识别。每一轮补的都是那个模板真正的定义性特征，不是为它量身定做的补丁：
#:
#: * 半神猴式：伸直的腿必须伸向**前**方（它伸向后方，投影 -1.55 对 +1.50）；
#: * 平板式：膝盖不能落地 —— 但 ``_knee_drop`` 取两膝**中点**，一膝落地
#:   一膝抬起被抵消成 0.57，恰好钻过 0.60 的上界，所以改取 max；
#: * 侧角伸展式：必须是站姿（髋只在踝上 0.73，真实侧角伸展式 0.96~1.07）；
#: * 上犬式：双腿必须都伸直（跪着那条 88°）；
#: * 侧板式：上手不能撑地（两手都在肩下 0.94）。
#:
#: 它是「一个高频动作缺模板会污染整个几何邻域」最完整的一个例子。
KNEELING_LEG_LIFT = skeleton(
    nose=(-130, -10), left_ear=(-118, -16), right_ear=(-116, -8),
    left_eye=(-126, -14), right_eye=(-124, -6),
    mouth_left=(-124, -2), mouth_right=(-122, 2),
    left_shoulder=(-99, -24), right_shoulder=(-98, -8),
    left_elbow=(-100, 27), right_elbow=(-99, 35),
    left_wrist=(-101, 78), right_wrist=(-100, 81),
    left_hip=(-1, -8), right_hip=(1, 8),
    left_knee=(78, 40), right_knee=(5, 92),
    left_ankle=(155, 56), right_ankle=(60, 78),
    left_heel=(163, 54), right_heel=(52, 82),
    left_foot_index=(175, 60), right_foot_index=(40, 90),
)

NON_TEMPLATE_POSES = {
    "蹲姿·躯干直立": SQUAT_UPRIGHT,
    "蹲姿·躯干前倾": SQUAT_FOLDED,
    "俯卧看手机": PRONE_PHONE,
    "随意站着": CASUAL_STAND,
    "双手撑地低弓步": LUNGE_HANDS_DOWN,
    "跪姿单腿后伸": KNEELING_LEG_LIFT,
}


def test_real_frame_geometry_matches_its_new_template():
    """按实测搭的骨架必须认成对的体式，不只是「认出了什么」。"""
    for name, (pts, key) in NOW_TEMPLATED.items():
        match = match_pose(L.normalize(pts))
        assert match is not None, f"{name} 认不出来了 —— 它现在有模板（{key}）"
        assert match.key == key, (
            f"{name} 被认成 {match.zh}（{match.score:.3f}），应该是 {key}"
        )
        assert match.score > 0.75, (
            f"{name} 认对了但只有 {match.score:.3f} —— 目标值大概取偏了位置，"
            f"实测数据落在容差边缘"
        )


def test_the_new_templates_do_not_poach_the_poses_they_were_confused_with():
    """新模板不能反过来把老体式抢走。

    补模板的风险和补门槛相反：门槛把帧推给邻居，模板把邻居的帧抢过来。
    这里钉住四对具体的相邻关系 —— 每一对都是先有过误判的。
    """
    pairs = [
        ("chair", "anjaneyasana"),          # 幻椅式曾被新月式认走
        ("chair", "mountain"),              # 两者只差腿直不直
        ("toe_squat", "tree"),              # 压脚背曾整簇占掉树式
        ("toe_squat", "child"),             # 两者都深度屈膝，靠躯干区分
        ("table_top", "plank"),             # 四足跪姿曾被平板式认走
        ("table_top", "chaturanga"),
        ("ardha_uttanasana", "uttanasana"),  # 同一动作的两个深度
        ("ardha_uttanasana", "parsvottanasana"),
    ]
    for new_key, old_key in pairs:
        norm = L.normalize(_canonical(old_key))
        match = match_pose(norm)
        assert match is not None and match.key == old_key, (
            f"补了 {new_key} 之后，{old_key} 的标准骨架被 "
            f"{match.zh if match else '未识别'} 抢走了"
        )


def test_a_squat_is_not_claimed_by_any_template():
    for name, pts in NON_TEMPLATE_POSES.items():
        match = match_pose(L.normalize(pts))
        assert match is None, (
            f"{name} 被 {match.zh} 认走（{match.score:.3f}）—— "
            f"库里没有它的模板，它应该回到未识别体式"
        )


def test_the_gate_is_what_rejects_it_not_a_lucky_threshold():
    """区别对待「门槛归零」和「刚好没过 min_score」。

    后者靠的是分数差，容差一放宽就会翻过去；前者是定义性的。

    只查每个负例**曾经被认走的那个**模板 —— 断言它在每一个模板上门槛
    都归零是过分的要求（一个站姿不需要被桥式的门槛拦住，它靠朝向和检查项
    就够远了），而且会把这个测试变成「模板集快照」，加模板就红。
    """
    was_claimed_by = {
        "蹲姿·躯干直立": ("pigeon", "ardha_hanumanasana"),
        "蹲姿·躯干前倾": ("pigeon", "ardha_hanumanasana"),
        # 下面四个现在有自己的模板了，但**曾经抢走它们的那些模板仍然必须
        # 判定它不是自己** —— 有了正确答案不等于错误答案自动消失。
        "幻椅式": ("anjaneyasana",),
        "压脚背": ("tree",),
        "四足跪姿": ("plank", "chaturanga", "bridge", "child"),
        "展背式（半程前屈）": ("uttanasana", "parsvottanasana", "downdog", "warrior3"),
        "俯卧看手机": ("pigeon", "ardha_hanumanasana", "updog"),
        "随意站着": ("tree",),
        "双手撑地低弓步": ("pigeon", "ardha_hanumanasana", "reverse_plank", "updog"),
        "跪姿单腿后伸": (
            "ardha_hanumanasana", "plank", "parsvakonasana", "updog", "side_plank",
        ),
    }
    for name, keys in was_claimed_by.items():
        pts = (
            NON_TEMPLATE_POSES[name]
            if name in NON_TEMPLATE_POSES
            else NOW_TEMPLATED[name][0]
        )
        norm = L.normalize(pts)
        for key in keys:
            m = score_by_key(norm, key)
            factor = m.gate * m.orientation
            assert factor == 0.0, (
                f"{name} 在 {key} 上的门槛系数是 {factor:.3f}（gate {m.gate:.2f} × "
                f"朝向 {m.orientation:.2f}），不是 0 —— 现在挡住它的是分数差而非"
                f"定义性否决，容差一动就会翻过去"
            )


def test_mountain_is_the_one_pose_with_no_definitional_gate():
    """山式的例外，以及为什么它只能靠分数差。

    上面那个测试的规矩是「负例必须被定义性门槛否决，不能只靠没过 min_score」。
    山式过不了这条规矩，而这不是山式的缺陷：**它和「随意站着」之间没有任何
    二值特征可分**。山式就是笔直站好、双臂垂在体侧，两者的差别全在程度上
    （膝多直、手臂多低），而门槛要的是「有 / 没有」那种判据。硬造一条
    只会是把某个容差改写成门槛的样子，判别力一分不多。

    所以山式靠的是：min_score 抬高到 0.70（模板里唯一一个），
    「双腿伸直」取两膝中较屈的那个（``_worse_leg_extended``）并给到最高权重。
    这个测试守住那个差距别缩水 —— 缩到 0.02 就等于没挡。
    """
    template = TEMPLATES_BY_KEY["mountain"]
    m = score_by_key(L.normalize(NON_TEMPLATE_POSES["随意站着"]), "mountain")
    margin = template.min_score - m.score
    assert margin >= 0.05, (
        f"随意站着在山式上拿到 {m.score:.3f}，门槛 {template.min_score} —— "
        f"只差 {margin:.3f}。山式没有定义性门槛可依，全靠这个差距，"
        f"留不到 0.05 就该重新想办法，而不是把 min_score 再抬一点"
    )


def test_gates_do_not_touch_the_poses_they_guard():
    """加门槛不能误伤本来就该匹配的体式。

    遍历**所有**带门槛的模板，而不是手列几个 —— 手列的清单加模板时会忘。
    """
    guarded = [t.key for t in TEMPLATES if t.gates]
    assert guarded, "一个门槛都没有？这个测试就白写了"
    for key in guarded:
        m = score_by_key(L.normalize(_canonical(key)), key)
        assert m.gate == 1.0, f"{key} 的标准骨架被自己的门槛挡住了（gate {m.gate:.2f}）"
        assert m.score > 0.9, f"{key} 标准骨架自身分只有 {m.score:.3f}"


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
