"""体式识别与正位打分：基于关节角度的规则模板。

一个模板是一组几何检查。每项检查测一个量（关节角、肢体与铅垂线的夹角、
两点的相对高低……），和目标值比对后给 0~1 的分：

    误差 <= tol            -> 1.0（完全正位）
    tol < 误差 < tol+slack -> 线性衰减
    误差 >= tol+slack      -> 0.0

模板总分是各项检查的加权平均，超过 ``min_score`` 才认为「这一帧是这个体式」。

局限（重要）
-----------
所有角度都在图像平面（2D 投影）上算。练习者若明显斜对镜头，投影角度会偏离
真实解剖角度，分数会偏低。固定机位、身体大致与镜头平面平行时最准。
识别不出的体式不会被丢弃 —— 它们照样参与九宫格的候选，只是没有正位分，
标签显示为「未识别体式」。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .landmarks import PoseView

# --------------------------------------------------------------------------
# 检查与模板
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """单项几何检查。"""

    label: str
    measure: Callable[[PoseView], float]
    target: float
    tol: float
    slack: float
    weight: float = 1.0
    unit: str = "°"

    def score(self, view: PoseView) -> tuple[float, float]:
        """返回 (得分, 实测值)。实测值为 nan 时得分也是 nan（该项跳过）。"""
        value = self.measure(view)
        if value is None or math.isnan(value):
            return float("nan"), float("nan")
        err = abs(value - self.target)
        if err <= self.tol:
            return 1.0, value
        if err >= self.tol + self.slack:
            return 0.0, value
        return 1.0 - (err - self.tol) / self.slack, value


@dataclass(frozen=True)
class Template:
    key: str
    zh: str
    en: str
    symmetric: bool
    checks: tuple[Check, ...]
    # 允许的躯干朝向区间，取值见 _spine_up()。这是**门槛**而非评分项：
    # 人倒过来了就不是战士二式，膝角再标准也不算。
    spine_up: tuple[float, float]
    min_score: float = 0.55


@dataclass
class CheckResult:
    label: str
    value: float
    target: float
    tol: float
    unit: str
    score: float


@dataclass
class PoseMatch:
    key: str
    zh: str
    en: str
    side: str
    score: float
    checks: list[CheckResult] = field(default_factory=list)
    orientation: float = 1.0  # 躯干朝向门槛的系数，已乘进 score

    @property
    def label(self) -> str:
        return self.zh

    def weak_checks(self, threshold: float = 0.75) -> list[CheckResult]:
        """得分偏低的检查项 —— 这就是「哪里没到正位」。"""
        return sorted(
            (c for c in self.checks if c.score < threshold), key=lambda c: c.score
        )


# --------------------------------------------------------------------------
# 复合测量helper
# --------------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else float("nan")


def _spine_up(view: PoseView) -> float:
    """脊柱方向的竖直分量，有符号：+1 完全直立，0 水平，-1 完全倒置。

    坐标已按躯干长度归一化，而躯干长度**正是**肩中点到髋中点的距离，
    所以这个位移量天然落在 [-1, 1]，等于躯干偏离铅垂线夹角的余弦。

    有符号这件事很关键：``angle_to_vertical`` 取了绝对值，分不出直立和倒立，
    单靠它会把肩倒立判成山式。
    """
    return view.dy("shoulder_mid", "hip_mid")


def _orientation_factor(value: float, lo: float, hi: float, slack: float = 0.35) -> float:
    """躯干朝向落在 [lo, hi] 内给 1.0，超出后线性衰减到 0。

    做成软门槛而不是硬布尔：MediaPipe 在肢体交叠时会有抖动，硬切会让
    临界帧忽有忽无。
    """
    if math.isnan(value):
        return 1.0  # 朝向算不出来时不惩罚，交给其它检查项判断
    if lo <= value <= hi:
        return 1.0
    distance = (lo - value) if value < lo else (value - hi)
    return max(0.0, 1.0 - distance / slack)


def _arms_extended(view: PoseView) -> float:
    """双肘平均伸展角，180 = 完全伸直。"""
    return _mean(
        [
            view.ang("s_shoulder", "s_elbow", "s_wrist"),
            view.ang("o_shoulder", "o_elbow", "o_wrist"),
        ]
    )


def _legs_extended(view: PoseView) -> float:
    """双膝平均伸展角，180 = 完全伸直。"""
    return _mean(
        [
            view.ang("s_hip", "s_knee", "s_ankle"),
            view.ang("o_hip", "o_knee", "o_ankle"),
        ]
    )


def _knees_flexed(view: PoseView) -> float:
    return _mean(
        [
            view.ang("s_hip", "s_knee", "s_ankle"),
            view.ang("o_hip", "o_knee", "o_ankle"),
        ]
    )


def _lowest_wrist_drop(view: PoseView) -> float:
    """两腕相对肩中点向下位移中**较小**的那个，躯干长度为单位。

    取 min 而不是均值，是因为均值会被一高一低抵消：侧板式一手撑地、一手朝天，
    平均下来看起来「像双手撑地」。取 min 则只要有一只手不在身下就立刻失分。

    平板式、四柱式必须双手撑地，这是它们区别于侧板式的硬特征。
    """
    return min(view.dy("shoulder_mid", "s_wrist"), view.dy("shoulder_mid", "o_wrist"))


def _feet_below_hips(view: PoseView) -> float:
    """踝中点低于髋中点的距离，躯干长度为单位。

    「站着」和「趴着」之间最干净的一刀：站姿体式这个值有 1.2~2.4，俯卧/仰卧
    体式（上犬、平板、四柱）接近 0。

    少了它，**上犬式会被三角伸展式的模板认走** —— 两者的躯干倾角（竖直分量
    都在 0.5 附近）和双腿伸直程度太接近，光靠朝向门槛分不开。同理，一个平板式
    能在战士三式模板上拿到 0.8 以上。凡是躯干允许大幅前倾/侧倾的站姿体式，
    都必须显式要求站姿，不能只靠朝向门槛。
    """
    return view.dy("hip_mid", "ankle_mid")


# --------------------------------------------------------------------------
# 体式模板
# --------------------------------------------------------------------------
#
# tol / slack 的取法：tol 是「练习者会觉得这已经到位了」的容差，
# slack 是从到位滑到完全不像这个体式的距离。角度类 unit="°"，
# 距离和 dy 类 unit="" 且单位是躯干长度。

TEMPLATES: tuple[Template, ...] = (
    Template(
        key="downdog",
        zh="下犬式",
        en="Downward-Facing Dog",
        symmetric=True,
        spine_up=(-1.05, -0.20),  # 髋高于肩，脊柱指向斜下方
        checks=(
            Check("髋部为最高点", lambda v: v.dy("hip_mid", "shoulder_mid"), 0.75, 0.35, 0.55, 2.0, ""),
            Check("躯干与手臂成一线", lambda v: v.ang("hip_mid", "shoulder_mid", "wrist_mid"), 172, 15, 35, 2.0),
            Check("髋部折角", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 78, 22, 32, 1.5),
            # 双腿向后斜展是与站立前屈式的区别：前屈时双腿竖直在髋正下方，
            # 而下犬式手脚分开成 V 形，腿与铅垂线有明显夹角。
            Check("双腿向后斜展", lambda v: v.vert("hip_mid", "ankle_mid"), 45, 20, 32, 2.0),
            Check("双腿伸直", _legs_extended, 176, 14, 35, 1.5),
            Check("双臂伸直", _arms_extended, 176, 12, 32, 1.0),
        ),
        min_score=0.62,
    ),
    Template(
        key="plank",
        zh="平板式",
        en="Plank",
        symmetric=True,
        spine_up=(-0.45, 0.45),  # 身体接近水平
        checks=(
            Check("身体成一直线", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 178, 10, 25, 2.0),
            Check("身体接近水平", lambda v: v.horiz("shoulder_mid", "ankle_mid"), 10, 12, 25, 1.5),
            Check("肩在腕正上方", lambda v: v.vert("s_wrist", "s_shoulder"), 0, 15, 30, 1.5),
            # 取双手中较低的那只：只看主侧的话，侧板式那只撑地的手会蒙混过关。
            Check("双手都撑在身体下方", _lowest_wrist_drop, 0.75, 0.40, 0.55, 2.0, ""),
            # 与反板式对称的判据。两者在 2D 剪影上近乎镜像（身体成直线、接近水平、
            # 双臂伸直撑地全都一样），头的朝向是唯一可靠的区别：平板式低头。
            Check("头部朝下（俯卧）", lambda v: v.dy("nose", "shoulder_mid"), -0.30, 0.30, 0.40, 2.5, ""),
            Check("双臂伸直", _arms_extended, 176, 12, 30, 1.0),
        ),
        min_score=0.62,
    ),
    Template(
        key="warrior2",
        zh="战士二式",
        en="Warrior II",
        symmetric=False,
        spine_up=(0.78, 1.06),  # 直立
        checks=(
            Check("前膝屈 90°", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 92, 15, 33, 2.5),
            Check("后腿伸直", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 175, 13, 33, 2.0),
            Check("双臂水平成一线", lambda v: v.horiz("s_wrist", "o_wrist"), 0, 12, 26, 2.0),
            # 「两腕连线水平」不足以区别于战士一式 —— 双臂上举时两腕同样等高。
            # 真正的分水岭是手臂在**肩高**，所以必须显式约束腕与肩的高差。
            Check("双臂与肩同高", lambda v: v.dy("shoulder_mid", "wrist_mid"), 0.0, 0.25, 0.40, 2.5, ""),
            Check("双臂伸直", _arms_extended, 174, 14, 32, 1.0),
            Check("躯干竖直", _spine_up, 0.97, 0.12, 0.30, 1.5, ""),
        ),
        min_score=0.62,
    ),
    Template(
        key="warrior1",
        zh="战士一式",
        en="Warrior I",
        symmetric=False,
        spine_up=(0.72, 1.06),
        checks=(
            Check("前膝屈 90°", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 98, 20, 35, 2.5),
            Check("后腿伸直", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 172, 16, 34, 2.0),
            # 手臂位置是战士一式区别于战士二式的关键，容差收紧、权重给足，
            # 否则双臂平举的战士二式也会拿到不低的战士一式分。
            Check("双臂上举过头", lambda v: v.dy("wrist_mid", "shoulder_mid"), 0.85, 0.30, 0.45, 2.5, ""),
            Check("躯干竖直", _spine_up, 0.95, 0.15, 0.35, 1.5, ""),
        ),
        min_score=0.62,
    ),
    Template(
        key="warrior3",
        zh="战士三式",
        en="Warrior III",
        symmetric=False,
        spine_up=(-0.40, 0.50),  # 躯干前倾到接近水平
        checks=(
            Check("躯干水平", lambda v: v.horiz("hip_mid", "shoulder_mid"), 0, 18, 30, 2.5),
            # 没有这一项，一个平板式能在本模板上拿到 0.8 以上 —— 躯干水平、
            # 双腿伸直、抬起腿水平三项它全满足。
            Check("支撑脚在髋下方（站姿）", lambda v: v.dy("hip_mid", "s_ankle"), 1.60, 0.55, 0.55, 2.5, ""),
            Check("支撑腿伸直", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 176, 12, 30, 2.0),
            # 单腿竖直支撑是战士三式的定义特征，权重要压住「躯干水平 + 双腿伸直」
            # 这几项 —— 平板式恰好也满足那几项。
            Check("支撑腿竖直", lambda v: v.vert("s_hip", "s_ankle"), 0, 15, 30, 2.5),
            Check("抬起腿伸直", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 175, 15, 35, 1.5),
            Check("抬起腿水平", lambda v: v.horiz("o_hip", "o_ankle"), 0, 20, 33, 1.5),
            # 单脚离地才是战士三式的定义特征，权重最高。三角伸展式双脚都在地面，
            # 这个量约为 0；战士三式约为 2（一脚在地、一脚抬到髋高）。
            # 实测真实视频里的三角伸展式：0.20 —— 缺这一条时它在本模板上拿了 0.88。
            # 这个判据几乎不受机位影响，而「躯干水平」「腿伸直」都会被投影带偏。
            Check("抬起脚远高于支撑脚", lambda v: v.dy("o_ankle", "s_ankle"), 1.70, 0.70, 0.70, 3.0, ""),
        ),
        min_score=0.64,
    ),
    Template(
        key="triangle",
        zh="三角伸展式",
        en="Triangle",
        symmetric=False,
        # 上界留给浅版本，下界要低到 0 附近 —— 真实练习里躯干常沉到接近水平，
        # 实测某支视频的三角伸展式是 -0.01。按教科书插画取 (0.25, 0.92) 会把
        # 深度版本判到区间外，于是「正确的模板先把自己排除了」，让战士三式抢走。
        spine_up=(-0.18, 0.92),
        checks=(
            # 站姿约束不可省：上犬式的躯干竖直分量也在 0.5 附近，会钻进上面的门槛。
            # 门槛放宽后这条更关键，slack 收紧到 0.55（上犬式的 0.30 直接归零）。
            Check("双脚在髋下方（站姿）", _feet_below_hips, 1.70, 0.70, 0.55, 2.5, ""),
            # 目标下移、容差放宽，覆盖「教科书 55°」到「深度接近水平」整个区间。
            Check("躯干侧倾", _spine_up, 0.30, 0.40, 0.35, 2.0, ""),
            Check("双腿伸直", _legs_extended, 176, 13, 32, 2.0),
            Check("双臂成一线", lambda v: v.ang("s_wrist", "shoulder_mid", "o_wrist"), 170, 18, 35, 2.0),
            Check("双臂接近竖直", lambda v: v.vert("s_wrist", "o_wrist"), 15, 20, 33, 1.5),
        ),
        min_score=0.64,
    ),
    Template(
        key="parsvakonasana",
        zh="侧角伸展式",
        en="Extended Side Angle",
        symmetric=False,
        # 比三角伸展式还要低：侧角伸展式的躯干本就该贴近前侧大腿、接近水平。
        # 实测某支视频：+0.10。
        spine_up=(-0.18, 0.88),
        checks=(
            # 与三角伸展式的分水岭就是这一项：侧角伸展屈前膝，三角伸展前腿伸直。
            # 权重给足，两个体式才不会互相抢。
            Check("前膝屈 90°", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 95, 18, 35, 2.5),
            Check("后腿伸直", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 175, 14, 33, 2.0),
            # 屈前膝使髋位下沉，目标比三角伸展式低。
            Check("双脚在髋下方（站姿）", _feet_below_hips, 1.35, 0.60, 0.55, 2.5, ""),
            Check("躯干侧倾", _spine_up, 0.28, 0.38, 0.35, 2.0, ""),
            Check("双臂成一线", lambda v: v.ang("s_wrist", "shoulder_mid", "o_wrist"), 166, 20, 38, 1.5),
            Check("下手贴近前脚", lambda v: v.dist("s_wrist", "s_ankle"), 0.55, 0.50, 0.70, 1.5, ""),
        ),
        min_score=0.62,
    ),
    Template(
        key="updog",
        zh="上犬式",
        en="Upward-Facing Dog",
        symmetric=True,
        spine_up=(0.22, 0.88),  # 肩高于髋 —— 与下犬式（髋最高）符号相反
        checks=(
            Check("双臂伸直", _arms_extended, 172, 15, 33, 2.0),
            Check("胸腔上提、肩高于髋", _spine_up, 0.55, 0.25, 0.35, 2.0, ""),
            Check("双腿伸直后展", _legs_extended, 172, 16, 35, 1.5),
            Check("下肢接近水平", lambda v: v.horiz("hip_mid", "ankle_mid"), 12, 16, 30, 1.5),
            Check("手撑在肩下方", lambda v: v.dy("s_shoulder", "s_wrist"), 0.80, 0.40, 0.55, 1.5, ""),
            # 上犬是后弯，髋角明显小于平直；平板式这一项接近 178°，用来区分两者。
            Check("躯干后弯", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 150, 22, 38, 1.5),
        ),
        min_score=0.62,
    ),
    Template(
        key="anjaneyasana",
        zh="新月式",
        en="Low Crescent Lunge",
        symmetric=False,
        spine_up=(0.72, 1.06),
        checks=(
            Check("前膝屈 90°", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 95, 22, 38, 2.0),
            # 与战士一式的分水岭：新月式后膝跪地屈曲，战士一式后腿伸直。
            # 两边权重对称，避免单向误判。
            Check("后膝屈曲跪地", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 105, 35, 50, 2.0),
            Check("后膝低于髋", lambda v: v.dy("hip_mid", "o_knee"), 0.80, 0.50, 0.70, 1.5, ""),
            Check("双臂上举过头", lambda v: v.dy("wrist_mid", "shoulder_mid"), 0.85, 0.35, 0.50, 2.5, ""),
            Check("躯干竖直", _spine_up, 0.95, 0.18, 0.35, 2.0, ""),
        ),
        min_score=0.64,
    ),
    Template(
        key="ardha_hanumanasana",
        zh="半神猴式",
        en="Half Splits",
        symmetric=False,
        spine_up=(-0.35, 0.75),  # 上身前折，脊柱远离铅垂
        checks=(
            # 前腿伸直是与婴儿式的分水岭（婴儿式双膝都深屈），权重给最高。
            Check("前腿伸直", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 175, 15, 32, 2.5),
            Check("后膝屈曲跪地", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 90, 30, 45, 2.0),
            # 跪姿：髋位低，双踝只在髋下 0.4 左右；站姿弓步是 1.0~1.2。
            # 实测真实视频里的侧角伸展式（站姿）：1.05 —— 缺这一条时它在本模板上
            # 拿了 0.92，把真正的侧角伸展式挤掉了。
            Check("髋位低（跪姿）", _feet_below_hips, 0.42, 0.28, 0.30, 2.5, ""),
            Check("躯干前折", _spine_up, 0.25, 0.35, 0.45, 2.0, ""),
            Check("前腿贴地伸展", lambda v: v.horiz("s_hip", "s_ankle"), 15, 20, 35, 1.5),
        ),
        min_score=0.64,
    ),
    Template(
        key="pigeon",
        zh="鸽子式",
        en="Pigeon Pose",
        symmetric=False,
        spine_up=(0.35, 1.06),  # 躯干直立版本（睡鸽式前折不在此模板内）
        checks=(
            Check("前腿屈膝外旋", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 85, 30, 45, 2.0),
            Check("后腿向后伸直", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 172, 18, 35, 2.0),
            Check("后腿贴地水平", lambda v: v.horiz("o_hip", "o_ankle"), 8, 16, 30, 2.0),
            # 髋部落地是鸽子式与新月式的分水岭：新月式髋位明显高于双踝。
            Check("髋部贴近地面", lambda v: abs(v.dy("hip_mid", "ankle_mid")), 0.25, 0.40, 0.60, 2.0, ""),
            Check("髋位低（跪姿）", _feet_below_hips, 0.42, 0.35, 0.35, 2.0, ""),
            Check("躯干直立", _spine_up, 0.85, 0.25, 0.40, 1.5, ""),
        ),
        min_score=0.64,
    ),
    Template(
        key="chaturanga",
        zh="四柱支撑式",
        en="Chaturanga Dandasana",
        symmetric=True,
        spine_up=(-0.45, 0.45),
        checks=(
            # 屈肘 90° 是与平板式唯一的区别（平板式双臂伸直约 176°），
            # 所以这一项权重最高。
            Check("屈肘约 90°", _arms_extended, 95, 25, 40, 2.5),
            Check("身体成一直线", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 178, 12, 28, 2.0),
            Check("身体接近水平", lambda v: v.horiz("shoulder_mid", "ankle_mid"), 8, 12, 25, 2.0),
            Check("双手都撑在身体下方", _lowest_wrist_drop, 0.45, 0.35, 0.50, 2.0, ""),
            Check("头部朝下（俯卧）", lambda v: v.dy("nose", "shoulder_mid"), -0.30, 0.30, 0.45, 1.5, ""),
        ),
        min_score=0.64,
    ),
    Template(
        key="side_plank",
        zh="侧板式",
        en="Side Plank",
        symmetric=False,
        spine_up=(0.00, 0.90),  # 身体呈斜线，肩略高于髋
        checks=(
            Check("身体成一直线", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 176, 14, 30, 2.0),
            Check("支撑臂伸直", lambda v: v.ang("s_shoulder", "s_elbow", "s_wrist"), 176, 14, 32, 2.0),
            Check("支撑臂接近竖直", lambda v: v.vert("s_wrist", "s_shoulder"), 10, 20, 35, 1.5),
            Check("身体呈斜线", lambda v: v.horiz("shoulder_mid", "ankle_mid"), 25, 18, 32, 1.5),
            Check("双腿伸直", _legs_extended, 175, 15, 33, 1.5),
            # 上侧手臂向天空伸展 —— 与平板式（双手都在地面）的区别。
            Check("上臂向上伸展", lambda v: v.dy("o_wrist", "s_shoulder"), 0.70, 0.50, 0.70, 2.0, ""),
        ),
        min_score=0.64,
    ),
    Template(
        key="reverse_plank",
        zh="反板式",
        en="Reverse Plank",
        symmetric=True,
        spine_up=(-0.30, 0.55),
        checks=(
            # 平板式和反板式在 2D 剪影上接近镜像，最可靠的区别是头的朝向：
            # 平板式低头（鼻低于肩），反板式仰头（鼻高于肩）。权重给足。
            Check("头部后仰、胸腔朝上", lambda v: v.dy("nose", "shoulder_mid"), 0.35, 0.25, 0.30, 2.5, ""),
            Check("身体成一直线", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 172, 16, 32, 2.0),
            Check("双臂伸直", _arms_extended, 174, 15, 33, 2.0),
            # 与平板式同理必须双手撑地 —— 侧板式一手朝天，只看主侧会被它蒙混。
            Check("双手都撑在肩后下方", _lowest_wrist_drop, 0.70, 0.45, 0.60, 2.0, ""),
            Check("双腿伸直", _legs_extended, 174, 15, 32, 1.5),
            Check("身体接近水平", lambda v: v.horiz("shoulder_mid", "ankle_mid"), 15, 18, 32, 1.5),
        ),
        min_score=0.66,  # 与平板式易混，门槛调高
    ),
    Template(
        key="uttanasana",
        zh="站立前屈式",
        en="Standing Forward Bend",
        symmetric=True,
        spine_up=(-1.06, -0.30),  # 躯干自髋向下倒垂
        checks=(
            Check("躯干向下倒垂", _spine_up, -0.85, 0.30, 0.40, 2.5, ""),
            Check("双腿伸直", _legs_extended, 176, 14, 32, 2.5),
            # 双腿竖直是与下犬式的分水岭：下犬式手脚分开，腿与铅垂线约 45°。
            Check("双腿竖直", lambda v: v.vert("hip_mid", "ankle_mid"), 0, 12, 26, 2.0),
            Check("双脚在髋下方（站姿）", _feet_below_hips, 1.90, 0.70, 0.90, 2.0, ""),
            Check("头部低于髋部", lambda v: v.dy("hip_mid", "nose"), 1.10, 0.60, 0.80, 1.5, ""),
        ),
        min_score=0.64,
    ),
    Template(
        key="tree",
        zh="树式",
        en="Tree",
        symmetric=False,
        spine_up=(0.82, 1.06),
        checks=(
            Check("支撑腿伸直", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 177, 10, 26, 2.0),
            Check("支撑腿竖直", lambda v: v.vert("s_hip", "s_ankle"), 0, 12, 26, 2.0),
            Check("抬起腿屈膝", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 45, 25, 40, 2.0),
            Check("抬起脚贴支撑腿", lambda v: v.dist("o_ankle", "s_knee"), 0.35, 0.30, 0.45, 1.5, ""),
            Check("躯干竖直", _spine_up, 0.97, 0.10, 0.28, 1.5, ""),
        ),
        min_score=0.64,
    ),
    Template(
        key="bridge",
        zh="桥式",
        en="Bridge",
        symmetric=True,
        spine_up=(-1.05, -0.05),  # 肩在地、髋抬起
        checks=(
            Check("髋部抬高", lambda v: v.dy("hip_mid", "shoulder_mid"), 0.50, 0.35, 0.55, 2.5, ""),
            Check("躯干与大腿成一线", lambda v: v.ang("shoulder_mid", "hip_mid", "knee_mid"), 165, 20, 35, 2.0),
            Check("屈膝约 90°", _knees_flexed, 85, 22, 38, 2.0),
        ),
        min_score=0.62,
    ),
    Template(
        key="child",
        zh="婴儿式",
        en="Child's Pose",
        symmetric=True,
        spine_up=(-1.05, 0.35),  # 上身前折，头低于髋或与髋相当
        checks=(
            Check("深度屈膝", _knees_flexed, 35, 22, 35, 2.0),
            Check("躯干折叠贴腿", lambda v: v.ang("s_shoulder", "s_hip", "s_knee"), 35, 25, 40, 2.0),
            Check("身体重心低", lambda v: abs(v.dy("hip_mid", "nose")), 0.45, 0.45, 0.60, 1.5, ""),
        ),
        min_score=0.62,
    ),
    Template(
        key="mountain",
        zh="山式",
        en="Mountain",
        symmetric=True,
        spine_up=(0.90, 1.06),  # 必须笔直站立
        checks=(
            Check("双腿伸直", _legs_extended, 178, 8, 22, 2.0),
            Check("身体竖直", lambda v: v.vert("hip_mid", "ankle_mid"), 0, 8, 20, 2.0),
            Check("躯干竖直", _spine_up, 1.0, 0.06, 0.20, 2.0, ""),
            Check("双臂垂于体侧", lambda v: v.dy("shoulder_mid", "wrist_mid"), 0.95, 0.35, 0.50, 1.5, ""),
        ),
        min_score=0.70,  # 站姿太常见，门槛调高，免得把过渡站立都算成山式
    ),
)

TEMPLATES_BY_KEY = {t.key: t for t in TEMPLATES}
UNKNOWN_LABEL = "未识别体式"


# --------------------------------------------------------------------------
# 匹配
# --------------------------------------------------------------------------


def _score_template(norm: np.ndarray, template: Template) -> PoseMatch | None:
    """在一个侧别假设下给模板打分；对非对称体式左右各试一次取高分。"""
    sides = ("left",) if template.symmetric else ("left", "right")
    best: PoseMatch | None = None

    for side in sides:
        view = PoseView(norm, side)
        results: list[CheckResult] = []
        total_weight = 0.0
        total_score = 0.0

        for check in template.checks:
            score, value = check.score(view)
            if math.isnan(score):
                continue
            results.append(
                CheckResult(check.label, value, check.target, check.tol, check.unit, score)
            )
            total_score += score * check.weight
            total_weight += check.weight

        if total_weight <= 0.0:
            continue

        orientation = _orientation_factor(
            _spine_up(view), template.spine_up[0], template.spine_up[1]
        )
        match = PoseMatch(
            key=template.key,
            zh=template.zh,
            en=template.en,
            side=side,
            score=(total_score / total_weight) * orientation,
            checks=results,
            orientation=orientation,
        )
        if best is None or match.score > best.score:
            best = match

    return best


def score_by_key(norm: np.ndarray, key: str) -> PoseMatch | None:
    """指定模板打分，**不设门槛**。

    聚类之后用来给同一簇内的每一帧算可比的正位分：簇内主导体式已经定了，
    此时再套 min_score 门槛只会让擦边帧无分可比。
    """
    template = TEMPLATES_BY_KEY.get(key)
    if template is None:
        return None
    return _score_template(norm, template)


def match_pose(
    norm: np.ndarray, exclude: frozenset[str] = frozenset()
) -> PoseMatch | None:
    """返回最匹配且过了门槛的体式，没有就返回 None。"""
    best: PoseMatch | None = None
    for template in TEMPLATES:
        if template.key in exclude:
            continue
        match = _score_template(norm, template)
        if match is None or match.score < template.min_score:
            continue
        if best is None or match.score > best.score:
            best = match
    return best
