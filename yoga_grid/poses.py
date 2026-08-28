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
class Gate:
    """体式的定义性特征：不满足就不是这个体式，其余项做得再标准也不算。

    和 ``Check`` 的区别不是严格程度，而是**作用方式**：Check 按权重扣分，
    一项归零最多损失它那份权重；Gate 是乘在总分上的系数，归零则整个模板
    归零。什么该做成 Gate：这个体式区别于「其它体式恰好也满足的那些检查」
    的那一条。

    实际吃过的教训（见 README「模板互相误判」）：蹲姿两条腿都折着约 64°，
    鸽子式六项里只有「后腿向后伸直」正确地判 0 分，另外五项被低髋位满足，
    于是蹲姿以 0.77 被认成鸽子式。唯一正确否决的那一项被投票淹没了。

    ``spine_up`` 是同一机制的先例，只是它每个模板都有、所以单独成字段。
    和 ``_orientation_factor`` 一样做成软门槛而不是硬布尔：MediaPipe 在
    肢体交叠时会抖，硬切会让临界帧忽有忽无。
    """

    label: str
    measure: Callable[[PoseView], float]
    lo: float
    hi: float
    #: 超出区间多远算完全不是这个体式。单位与 measure 相同。
    slack: float


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
    #: 除朝向之外的定义性门槛。见 :class:`Gate`。
    gates: tuple[Gate, ...] = ()


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
    gate: float = 1.0  # 其它定义性门槛的系数之积，也已乘进 score

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


def _worse_leg_extended(view: PoseView) -> float:
    """两膝中**较屈**的那个的伸展角。

    「双腿伸直」说的是两条腿都直，而均值会被一直一屈抵消：真实视频里一帧
    随意站姿两膝是 171° 和 148°，均值 159° 落在山式 178±8 的衰减区里还能拿
    0.52 分，取 min 则直接 0 分。同 ``_lowest_wrist_drop`` 取 min 的道理。

    只给山式用，别处仍用均值：山式的容差本就收得紧（178±8），职责就是把
    过渡站立挡在外面；而侧面机位下远侧那条腿常被遮挡、膝角估计会抖，
    实测两腿差的 p90 在下犬式能到 55° —— 那些模板里这一项只是六项之一，
    均值对遮挡更稳。
    """
    return min(
        view.ang("s_hip", "s_knee", "s_ankle"),
        view.ang("o_hip", "o_knee", "o_ankle"),
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


def _ankle_spread(view: PoseView) -> float:
    """两踝的水平间距，躯干长度为单位。

    区分「双脚并拢的前屈」和「双脚前后错开的前屈」用的就是这一个量：
    站立前屈式实测 0.02~0.35，金字塔式（双脚前后分开）实测 0.58~1.53。
    少了它，站立前屈式的模板对两者都给满分 —— 真实视频里那一簇 22 帧就混了
    两个体式，其中 20 帧其实是金字塔式。

    取水平分量而不是欧氏距离：两脚是前后错开，竖直方向本来就该几乎等高。
    """
    return abs(float(view.pt("s_ankle")[0] - view.pt("o_ankle")[0]))


def _knee_drop(view: PoseView) -> float:
    """膝中点低于髋中点的距离，躯干长度为单位。

    俯卧支撑类体式和**跪姿**之间的分水岭。平板式的膝在肩踝连线上，身体与
    地面约 25°，所以膝只比髋低 0.27~0.45；四足跪姿/桌面式的膝直接压在髋
    正下方的地面上，实测 0.87~0.98。

    真实视频里 121 帧「平板式」有 46 帧是跪姿（膝角 67~88°、膝落地），
    平板式六项里只有「身体成一直线」正确地判 0 分（实测 126~146°），
    其余五项被跪姿满足，于是以 0.62~0.75 越过门槛 —— 又一次「唯一正确
    否决的那一项被投票淹没」。
    """
    return view.dy("hip_mid", "knee_mid")


def _lower_knee_drop(view: PoseView) -> float:
    """两膝中**更贴近地面**的那个低于髋的距离（取 max）。

    ``_knee_drop`` 取的是两膝中点，而中点会被「一膝落地、一膝抬起」抵消成
    一个中间值 —— 又一次和 ``_lowest_wrist_drop`` / ``_worse_leg_extended``
    同样的毛病，这已经是第三次了。

    实测：**四足跪姿单腿后伸**（一膝跪地读 0.92、一膝伸直抬起读 0.2）中点
    算出来是 0.57，恰好钻过平板式「膝盖不能落地」那道门槛的上界 0.60，
    于是被平板式认走 0.93。取 max 则立刻读到 0.92。

    「膝盖不能落地」问的是**有没有**膝落地，所以要看更低的那个膝。
    """
    return max(view.dy("hip_mid", "s_knee"), view.dy("hip_mid", "o_knee"))


def _higher_knee_drop(view: PoseView) -> float:
    """两膝中**离地更远**的那个低于髋的距离（取 min）。

    和 ``_lower_knee_drop`` 成对：四足跪姿要求**两膝都**落地，所以看抬得
    更高的那个膝够不够低。单腿后伸时它读 0.2，四足跪姿两膝都是 0.87~0.98。
    """
    return min(view.dy("hip_mid", "s_knee"), view.dy("hip_mid", "o_knee"))


def _front_shin_from_horizontal(view: PoseView) -> float:
    """前侧小腿与水平线的夹角，0 = 水平。

    鸽子式的定义性特征：前侧小腿**横过身前**贴地，接近水平。弓步类体式的
    前侧小腿则接近竖直（踝在膝正下方）。实测真实视频里被鸽子式认走的
    「双手撑地的低弓步」是 86~90°，而标准鸽子式骨架是 29°。

    只看膝角分不开这两者 —— 低弓步的前膝也屈到 79~86°，恰好落在鸽子式
    「前腿屈膝外旋 85±30」的正中间。
    """
    return view.horiz("s_knee", "s_ankle")


def _back_ankle_below_knee(view: PoseView) -> float:
    """后侧踝低于后侧膝的距离，躯干长度为单位。负 = 踝比膝高。

    回答一个很具体的问题：**那条腿是躺在地上，还是踩在地上。**跪姿和俯卧
    体式的后侧小腿沿地面向后铺开，踝与膝几乎等高（实测半神猴式 -0.03~+0.07、
    上犬式 +0.03~+0.20）；站姿则是踝在膝正下方一整条小腿的距离（实测 +0.5~+0.8）。

    为什么需要它：非对称模板会左右各试一遍取高分，于是**站姿弓步可以把
    「屈着的前膝」当成「跪着的后膝」**。真实视频里一帧用户确认正确的侧角
    伸展式（前膝 113°、后腿 170°）就这样在半神猴式上拿到 0.99 —— 侧别一翻，
    「前腿必须伸直」看到 170°、「后膝必须屈曲」看到 113°，两条腿角度门槛
    同时被满足。膝角是角度，分不出那条腿在不在地上；这个量能。
    """
    return view.dy("o_knee", "o_ankle")


def _straight_leg_reaches_forward(view: PoseView) -> float:
    """主侧踝相对髋的水平偏移，符号取成「与躯干前倾同侧为正」。

    回答的是用户的原话：**那条伸直的腿有没有伸到前面去。**

    半神猴式是「上身折在伸直的前腿上」，所以躯干和那条腿必然偏向同一侧；
    而**四足跪姿单腿后伸**（鸟狗式的腿那一半）同样是「一条腿伸直、另一条
    跪着、小腿贴地、双手撑地」——四条门槛全满足——但腿伸向躯干的**反**侧。

    实测这一刀切得非常干净：
    用户确认正确的半神猴式 25 帧读 +1.43~+1.66（标准骨架 +1.40），
    单腿后伸的那 7 帧读 -1.55~-1.64。中间空着 3.0，而且符号本身就相反。

    躯干没有明显前倾时（肩在髋正上方）返回 nan —— 这时这个判据说不出话，
    ``_orientation_factor`` 对 nan 不惩罚，正是想要的行为。

    只用水平分量，不用有向角：机位在身体侧面时前后方向就落在图像的 x 轴上，
    而竖直分量会被「折得多深」污染。
    """
    hip = view.pt("hip_mid")
    lean = float(view.pt("shoulder_mid")[0] - hip[0])
    if abs(lean) < 0.05:
        return float("nan")
    reach = float(view.pt("s_ankle")[0] - hip[0])
    return reach if lean > 0 else -reach


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
            #
            # 目标从 45 下调到 35：实测 31 帧落在 20~30°（中位 28），
            # 45±20 让 23% 的真实下犬式在这一项失分。区间仍与站立前屈式
            # 分得开 —— 实测前屈是 0~4°，在 35±20 外还有 11° 余量。
            Check("双腿向后斜展", lambda v: v.vert("hip_mid", "ankle_mid"), 35, 20, 32, 2.0),
            # 目标下调、容差放宽：实测双腿 114~175（中位 167）、双臂 157~172
            # （中位 164）。176±12 时「双臂伸直」只有 52% 的真实下犬式满分，
            # 这一项等于在惩罚所有人 —— 手肘完全锁直既不常见也不是要点。
            Check("双腿伸直", _legs_extended, 172, 16, 35, 1.5),
            Check("双臂伸直", _arms_extended, 170, 16, 32, 1.0),
        ),
        min_score=0.62,
        # 手脚分开成 V 形是下犬式的定义。这一条原先只靠上面那个评分项承担，
        # 而把它的目标从 45 下调到 35（照实测）之后，判别力也跟着掉了：
        # 一帧**站立伸展**（双腿竖直，离铅垂线只有 3.7°）在这一项上从 0 分
        # 变成 0.65 分，总分 0.72 越过门槛。
        #
        # 教训：按实测放宽一项容差，要先问它是不是还兼着判别的职责。
        # 是的话，把判别那一半移到门槛上，别指望一个区间同时干两件事。
        gates=(
            Gate("双腿必须向后斜展", lambda v: v.vert("hip_mid", "ankle_mid"), 15, 75, 10),
        ),
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
        # 「身体成一直线」是平板式的评分项，塌腰的平板式仍然是平板式，所以它
        # 不能当门槛。区别「差的平板式」和「根本不是平板式」的是膝有没有落地。
        gates=(Gate("膝盖不能落地", _lower_knee_drop, -2.0, 0.60, 0.20),),
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
        # 见新月式的同名门槛：幻椅式双膝都屈、双臂上举、躯干竖直，
        # 弓步类体式必须显式要求双脚前后分开。
        gates=(Gate("双脚必须前后分开", _ankle_spread, 0.70, 9.0, 0.40),),
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
        # 单脚离地是战士三式的定义，同树式。它原先只是上面那个评分项
        # （权重 3.0 / 总权重 15.5 = 19%），所以双脚都在地面的**站立伸展**
        # 仍能拿 0.69 —— 躯干水平、支撑腿伸直竖直、抬起腿伸直这五项，
        # 一个前屈全都满足。实测标准骨架抬高 2.00，双脚落地是 0.06。
        gates=(
            Gate("抬起脚必须离地", lambda v: v.dy("o_ankle", "s_ankle"), 0.80, 3.5, 0.40),
        ),
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
            # **刻意保留教科书值，别按实测改。** 用户的决定（2026-08）：
            # 「侧角伸展先按教科书来」。
            #
            # 实测这位练习者做侧角伸展式时这一项**满分率 0%**：双手并拢撑地时
            # 读 0.5~14°，上手举过头时读 87~101°，而教科书那种「下手到上手
            # 一条直线穿过双肩」在真人身上根本到不了 166°（那是插画的画法）。
            # 所以这一项现在是**教学目标**而不是识别判据：它每次都会在
            # report.md 里提示手臂还没打开，这正是用户要的。
            #
            # 这和「按实测校准容差」的原则冲突吗？不冲突 —— 那条原则管的是
            # **识别**（区间取错位置会让正确的模板把自己排除掉，见 README
            # 「局限」）。这一项权重只有 1.5，不参与任何判别，改不改都不影响
            # 认得对不对。识别按实测，教学目标按教科书，两件事。
            Check("双臂成一线", lambda v: v.ang("s_wrist", "shoulder_mid", "o_wrist"), 166, 20, 38, 1.5),
            Check("下手贴近前脚", lambda v: v.dist("s_wrist", "s_ankle"), 0.55, 0.50, 0.70, 1.5, ""),
        ),
        min_score=0.62,
        # 后腿伸直是侧角伸展式的定义（同战士一式、战士二式、三角伸展式）。
        # 它原先只是上面那个评分项，于是**双手撑地的低弓步**（后膝 96~139°）
        # 一路被各模板拒绝后落到这里，仍能拿 0.66~0.72。
        #
        # 界线取得住：真实练习里被用户确认为侧角伸展式的 5 帧后膝读
        # 173.9~177.0°（标准骨架 180），而 8 帧低弓步是 96.3~139.2° ——
        # 中间空着 35°，门槛放在 150 两边都不擦边。
        gates=(
            Gate("后腿必须伸直", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 150, 190, 15),
            # 站姿是定义性的。``_feet_below_hips`` 的注释早就写着「凡是躯干允许
            # 大幅前倾/侧倾的站姿体式，都必须显式要求站姿」，但它一直只是评分项。
            # 实测代价：一帧**四足跪姿单腿后伸**（髋只在踝上 0.66）被各模板依次
            # 拒绝后落到这里 0.72 —— 侧别一翻，伸直的那条腿满足「后腿伸直」，
            # 跪着的那条满足「前膝屈 90°」，站姿那一项判 0 分却被投票淹没。
            # 用户确认正确的侧角伸展式 5 帧读 0.96~1.07（标准骨架 1.16）。
            # slack 取 0.10 而不是 0.15：跪姿那批帧读 0.65~0.70，slack 0.15 只把
            # 它们衰减到 0.13~0.33，靠分数差补足 —— 那正是这条门槛要避免的情形。
            # 收到 0.10 后 0.70 以下直接归零，而真实侧角伸展式最低 0.96，
            # 离边界 0.80 还有 0.16 的余量。
            Gate("必须是站姿", _feet_below_hips, 0.80, 3.0, 0.10),
        ),
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
            # 上犬是后弯，髋角明显小于平直 —— 这是它区别于平板式的**定义**。
            #
            # 容差从 150±22/slack 38 收到 145±18/slack 12（衰减到 0 的位置从
            # 210° 提到 175°）。原来那个区间松得离谱：一个**完全平直**的身体
            # （实测 175.5°）在这一项上还能拿 0.91 分，于是 8 帧斜板/平板式
            # 被认成上犬式 0.98，九宫格里同一个姿势出现了两次（格3 和格6）。
            #
            # 实测依据：整个数据集里唯一一帧真上犬式读 138.7°，标准骨架 156.1°，
            # 而被误认的那批平板式是 175.5~179.1°。中间空着 37°。
            Check("躯干后弯", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 145, 18, 12, 1.5),
        ),
        min_score=0.62,
        # 直臂撑地、胸腔上提是上犬式的定义；屈着肘的俯卧后弯是斯芬克斯式/
        # 眼镜蛇式，库里没有模板。给鸽子式补上前腿门槛后，那帧**趴着看手机**
        # （双肘折到 8°和 57°）就改投上犬式 0.78 —— 六项里只有「双臂伸直」
        # 判 0 分，又一次被投票淹没。
        gates=(
            Gate("双臂必须伸直", _arms_extended, 140, 190, 30),
            # 双腿沿地面向后铺开是「俯卧」的部分。被各模板依次拒绝的那组
            # **双手撑地的低弓步**最后落到上犬式头上 0.70：躯干前倾、双臂
            # 伸直撑地、下肢接近水平这几项它都满足，站着的那条腿却踩在地上
            # （+0.61），而真实上犬式是 +0.03~+0.20。
            Gate("双腿必须贴地后展", _back_ankle_below_knee, -1.0, 0.38, 0.18),
            # 后弯是定义性的，光靠上面那个评分项不够 —— 它权重 1.5 / 总权重
            # 9.5，判 0 分也只扣 16%，平板式照旧能拿 0.83。上界 168° 留着
            # 12° 余量给标准骨架（156.1°），而平板式最小 175.5°。
            Gate("躯干必须后弯", lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"), 100, 168, 6),
            # 双腿都向后伸直是上犬式的定义（两条大腿都离地后展）。取两膝中较屈
            # 的那个，别取均值 —— 已经是第四次踩这个坑了，见 _worse_leg_extended。
            # 实测真上犬式 167.7（两膝 167.7/168.1，标准骨架 180）；而一帧
            # **四足跪姿单腿后伸**（一膝跪着 88°、一腿伸直）在这里拿了 0.72，
            # 「双腿伸直后展」和「胸腔上提」两项都判 0 分仍不够。
            Gate("双腿必须都伸直", _worse_leg_extended, 140, 190, 20),
        ),
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
        # 真实视频里 8 帧**幻椅式**被认成新月式（0.87~1.00）：双膝都屈到
        # 117~139°，「前膝屈 90°」和「后膝屈曲跪地」两项同时满足，双臂上举、
        # 躯干竖直也满足，五项里没有一项否决它。弓步的定义性特征是双脚
        # 前后分开 —— 实测弓步 1.93~2.04，幻椅式 0.01~0.07，差 28 倍。
        gates=(Gate("双脚必须前后分开", _ankle_spread, 0.70, 9.0, 0.40),),
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
            # 目标从 0.42 上调到 0.75：实测 34 帧落在 0.61~0.94（中位 0.84），
            # 0.42±0.28 只覆盖到 0.70，76% 的真实半神猴式在这一项失分。
            #
            # 0.42 是照标准骨架取的（骨架读 0.43），而标准骨架把髋画得比真人
            # 练习时低。容差同时放宽到 0.35，好让标准骨架自己也留在区间内。
            #
            # 这一项原先兼职「区分站姿弓步」（旧注释记的是侧角伸展式实测 1.05）。
            # 现在不再指望它做这件事：实测侧角伸展式 0.74~1.07 与半神猴式
            # 0.61~0.94 已经重叠，区分改由「后膝必须屈曲」门槛承担。
            Check("髋位低（跪姿）", _feet_below_hips, 0.75, 0.35, 0.30, 2.5, ""),
            Check("躯干前折", _spine_up, 0.25, 0.35, 0.45, 2.0, ""),
            Check("前腿贴地伸展", lambda v: v.horiz("s_hip", "s_ankle"), 15, 20, 35, 1.5),
        ),
        min_score=0.64,
        # 同鸽子式：伸直的那条腿是半神猴式的定义。真实误判是躯干前倾的
        # 蹲姿——朝向门槛拦不住它，因为蹲姿确实可以前倾。
        #
        # 「后膝必须屈曲」是另一半定义，两条缺一不可：真实视频里一帧
        # **俯卧撑着上身看手机**（双腿都伸直、髋贴地）先被鸽子式认走 0.83，
        # 给鸽子式补上前腿门槛后就落到半神猴式头上 0.80 —— 补一个门槛把
        # 误判推给邻居，是这类修改的常见后果，要顺着推下去补完。
        #
        # 这一条同时是把「髋位低（跪姿）」重新定心到实测值的前提：那一项
        # 原先兼职区分站姿弓步（见其注释），而实测半神猴式 0.61~0.94 与
        # 侧角伸展式 0.74~1.07 已经重叠，它区分不动了。后膝角能：
        # 半神猴式 68~93°，侧角伸展式 169~177°。
        gates=(
            # 下界从 135 收到 152：那组低弓步左右各试一遍时，139° 那条腿被
            # 当成「伸直的前腿」钻了进来（这一项只判 0.35 分，被投票淹没），
            # 而 139° 显然不是伸直。实测半神猴式最小 161°、标准骨架 180°，
            # 收到 152 还留着 9° 余量。
            Gate("前腿必须伸直", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 152, 190, 18),
            Gate("后膝必须屈曲", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 30, 120, 30),
            # 前腿沿地面伸出去，所以前侧小腿接近水平（标准骨架 13°，实测
            # 17~44°）。给鸽子式补上门槛后，那组**双手撑地的低弓步**改投
            # 半神猴式 0.85：左右各试一遍时，139° 那条腿被当成「伸直的前腿」，
            # 84° 那条被当成「屈曲的后膝」，两条腿角度门槛同时被满足。
            # 小腿朝向拦得住 —— 弓步的小腿接近竖直。
            Gate("前小腿必须接近水平", _front_shin_from_horizontal, 0, 55, 25),
            # 「髋位低（跪姿）」按实测重新定心之后就不再兼职区分站姿了
            # （实测半神猴式 0.80~0.94 与侧角伸展式 0.96~1.07 只差 0.02，
            # 任何区间都分不开），这一条接手：后膝必须真的跪在地面上。
            # 实测半神猴式 -0.03~+0.07，被抢走那帧站姿是 +0.79。
            Gate("后膝必须跪在地面上", _back_ankle_below_knee, -1.0, 0.30, 0.20),
            # 用户原话：「不是半神猴，因为没有腿伸到了前面」。
            #
            # **四足跪姿单腿后伸**（鸟狗式的腿那一半）把上面四条门槛全满足了：
            # 一条腿伸直（172°）、另一条跪着（88°）、小腿贴地（-0.13）、
            # 前小腿接近水平（20°），于是拿到 1.00。它和半神猴式的差别不在
            # 任何一个角度上，而在**那条伸直的腿伸向哪一边** —— 半神猴式的
            # 上身折在腿上（同侧），单腿后伸是腿朝躯干的反侧。
            #
            # 实测：确认正确的半神猴式 25 帧 +1.43~+1.66（标准骨架 +1.40），
            # 单腿后伸 7 帧 -1.55~-1.64。这是这批数据里分得最开的一条判据。
            Gate("伸直的腿必须伸向前方", _straight_leg_reaches_forward, 0.60, 4.0, 0.50),
        ),
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
        # 三条门槛对应鸽子式的三个定义性特征。区间都比同名 Check 宽得多：
        # 门槛只回答「是不是这个体式」，好不好交给 Check。
        #
        # 每一条都是真实视频里的一次误判换来的：
        #
        # * 后腿伸直 —— 蹲姿两腿都约 64°，落到区间外 71°（标准骨架读 180）。
        # * 前腿屈膝 —— 一帧**俯卧撑着上身看手机**双腿都伸直（前腿 168°、
        #   后腿 178°），六项里只有「前腿屈膝外旋」判 0 分，仍拿到 0.83。
        # * 前小腿横过身前 —— 一组**双手撑地的低弓步**前膝屈到 79~86°，
        #   正落在「前腿屈膝外旋 85±30」的正中间，前两条门槛都拦不住；
        #   区别在小腿：鸽子式的小腿贴地横过身前（标准骨架 29°），
        #   弓步的小腿接近竖直（实测 86~90°）。
        gates=(
            Gate("后腿必须伸直", lambda v: v.ang("o_hip", "o_knee", "o_ankle"), 135, 190, 35),
            Gate("前腿必须屈膝", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 20, 130, 30),
            Gate("前小腿必须横过身前", _front_shin_from_horizontal, 0, 50, 25),
        ),
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
        # 与平板式同一条门槛。给平板式补上之后，那 46 帧四足跪姿立刻改投
        # 四柱支撑式（0.80）—— 跪姿屈着的肘正好满足「屈肘约 90°」。
        # 补门槛把误判推给邻居，就得顺着推下去补完。
        gates=(Gate("膝盖不能落地", _lower_knee_drop, -2.0, 0.60, 0.20),),
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
        # 「身体成一直线」是侧板式的定义，不只是打分项。真实视频里一帧髋折到
        # 106° 的侧卧姿势（一手撑地一手抬到肩高）拿了 0.77：这一项判 0 分，
        # 另外五项（支撑臂伸直、接近竖直、身体呈斜线、双腿伸直、上臂向上）
        # 全部满足。
        #
        # 平板式和四柱支撑式**不加**这条同名门槛，是刻意的：那两个体式塌腰
        # 的版本仍然是它们自己，塌腰恰恰是要反馈给练习者的东西，做成门槛会
        # 把这条建议整个静音。它们的定义性门槛是「膝盖不能落地」。侧板式
        # 这里不一样 —— 髋折 106° 没有哪种读法能算「侧板式做差了」。
        gates=(
            Gate(
                "身体必须成一直线",
                lambda v: v.ang("shoulder_mid", "hip_mid", "ankle_mid"),
                145, 200, 25,
            ),
            # 「上手朝天、只有一只手撑地」是侧板式区别于平板式的定义 —— 上面
            # 「上臂向上伸展」那个评分项的注释早就这么写了，但它只有 2.0 权重、
            # 归零也只扣 19%。实测代价：一帧**四足跪姿单腿后伸**（两手都撑地，
            # 较高那只腕仍在肩下 0.91）被前面五个模板依次拒绝后落到这里 0.67，
            # 刚过 0.64 的门槛。
            #
            # 上界 0.45 留给「上手扶髋」这种常见变体（腕约在肩下 0.3），
            # 标准骨架是 -1.29（上手高过肩一整条手臂）。
            Gate("上手不能撑地", _lowest_wrist_drop, -3.0, 0.45, 0.25),
        ),
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
        # 反板式是「仰面的平板」：身体一条直线、双腿伸直。那组**双手撑地的
        # 低弓步**（前膝 84°、双腿均值 112°）躯干直立、双臂撑地、鼻高于肩，
        # 六项里只有「双腿伸直」判 0 分，仍拿 0.73。
        gates=(
            Gate("双腿必须伸直", _legs_extended, 140, 190, 25),
            # 直臂撑地是反板式的定义，同上犬式。那帧**俯卧撑起上身看手机**
            # （双肘折到 29~30°）在这里拿了 0.71~0.75：六项里只有「双臂伸直」
            # 判 0 分，权重 2.0 / 总权重 11.5 只扣掉 17%。
            Gate("双臂必须伸直", _arms_extended, 140, 190, 30),
        ),
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
            # 双脚并拢是与金字塔式的分水岭 —— 后者双脚前后错开。缺这一条时，
            # 真实视频里 20 帧金字塔式全被判成了站立前屈式。
            Check("双脚并拢", _ankle_spread, 0.15, 0.40, 0.45, 2.5, ""),
            Check("头部低于髋部", lambda v: v.dy("hip_mid", "nose"), 1.10, 0.60, 0.80, 1.5, ""),
        ),
        min_score=0.64,
        # 真实视频里 18 帧**站立伸展/展背**（上身只折到略过水平、头还在髋高
        # 附近）被认成站立前屈式，0.91。除「躯干向下倒垂」外五项全满分 ——
        # 腿直、腿竖直、站姿、双脚并拢、头低于髋（0.50~0.70，勉强进 1.10±0.60）
        # 半程折叠这些都满足。
        #
        # 为什么用 Gate 而不是收紧 spine_up：``spine_up`` 字段没有 slack，
        # 衰减距离固定 0.35，把上界收到 -0.55 时实测 -0.49 那些帧仍能拿到
        # 0.83 的系数、越过门槛。Gate 自带 slack，能把衰减收到 0.18。
        gates=(Gate("躯干必须倒垂过半", _spine_up, -1.10, -0.60, 0.18),),
    ),
    Template(
        key="parsvottanasana",
        zh="金字塔式",
        en="Intense Side Stretch",
        symmetric=False,
        spine_up=(-1.06, -0.15),  # 躯干自髋向下折叠
        checks=(
            Check("躯干向下折叠", _spine_up, -0.65, 0.35, 0.40, 2.0, ""),
            Check("双腿伸直", _legs_extended, 175, 15, 33, 2.0),
            # 与站立前屈式的分水岭。容差取得宽：错开量在画面上的投影随机位变化
            # 很大，实测同一支视频里 0.58~1.53 都是这个体式。
            Check("双脚前后错开", _ankle_spread, 1.10, 0.50, 0.35, 2.5, ""),
            Check("双脚在髋下方（站姿）", _feet_below_hips, 1.50, 0.55, 0.55, 2.0, ""),
            Check("头部低于髋部", lambda v: v.dy("hip_mid", "nose"), 1.00, 0.60, 0.70, 1.5, ""),
        ),
        min_score=0.64,
        # 「双脚前后错开」原先只是评分项（权重 2.5 / 总权重 10），给站立前屈式
        # 补上倒垂门槛后，那 18 帧**站立伸展**（双脚并拢，踝距 0.01）就改投
        # 金字塔式 0.75 —— 这一项判了 0 分却被另外四项投票淹没。
        # 双脚前后分开是金字塔式区别于站立前屈式的定义，该做成门槛。
        gates=(Gate("双脚必须前后错开", _ankle_spread, 0.60, 9.0, 0.35),),
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
        # 「单腿站立」是树式的定义，缺了它**跪坐会把树式全占掉**：真实视频里
        # 16 帧压脚背（跪坐在脚跟上、躯干竖直、手在胸前）全被认成树式，
        # 0.72~0.78。五项里只有「支撑腿伸直」判 0 分，另外四项是跪坐白拿的
        # —— 两膝都折着自然满足「抬起腿屈膝」（实测 21~33°，目标 45±25），
        # 两踝并在髋下自然满足「抬起脚贴支撑腿」，躯干竖直也满足。
        #
        # 这就是用户那句「这个视频中我根本就没有练习树式」背后的东西：
        # 一个体式在库里没有模板时，它会去挤几何上最像的那个。
        gates=(
            Gate("支撑腿必须伸直", lambda v: v.ang("s_hip", "s_knee", "s_ankle"), 140, 190, 30),
            # 单脚离地是另一半定义。补上支撑腿门槛后，用户指出的那帧
            # **随意站着**（两膝 171°/148°）改投树式 0.74：148° 那条腿满足
            # 「抬起腿屈膝 45±25」的衰减区，两脚并在一起又满足「抬起脚贴
            # 支撑腿」。标准骨架抬起脚高出 1.25，随意站姿两脚等高、只有 0.04。
            Gate("抬起脚必须离地", lambda v: v.dy("o_ankle", "s_ankle"), 0.35, 3.0, 0.25),
        ),
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
        # 桥式是仰卧：肩在地、髋抬起，膝大致与髋同高（标准骨架 0.11）。
        # 四足跪姿的膝压在髋正下方 0.80。给平板式和四柱式补上膝落地门槛后，
        # 那 46 帧跪姿又改投桥式 0.69 —— 三项里只有「躯干与大腿成一线」
        # 判 0 分，而桥式只有三项，一项归零只扣 31%。
        gates=(Gate("膝不能压在髋下方（仰卧）", _lower_knee_drop, -2.0, 0.45, 0.20),),
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
        # 婴儿式就是「躯干伏在大腿上」，髋角很小（标准骨架 30°）。四足跪姿
        # 的髋是展开的（实测 72°），但婴儿式只有三项检查，一项归零只扣 31%，
        # 于是跪姿刚好压在门槛上（0.62 = min_score）。
        # 靠调高 min_score 也能挡住它，但那是「刚好没过」而不是定义性否决，
        # 容差一动就会翻过去 —— 见 tests 里 test_the_gate_is_what_rejects_it。
        gates=(
            Gate("躯干必须折叠贴腿", lambda v: v.ang("s_shoulder", "s_hip", "s_knee"), 0, 60, 20),
        ),
    ),
    Template(
        key="mountain",
        zh="山式",
        en="Mountain",
        symmetric=True,
        spine_up=(0.90, 1.06),  # 必须笔直站立
        checks=(
            # 取两膝中较屈的那个而不是均值，权重也压过其余各项：山式和「随意
            # 站着」在几何上只差腿直不直，这一项就是山式的全部判别力所在。
            # 真实视频里一帧随意站姿两膝 171°/148°，均值 159° 还能拿 0.52，
            # 取 min 后 148° 直接 0 分；权重 2.0 时总分仍有 0.73（门槛 0.70），
            # 提到 3.0 才落到 0.62 —— 门槛能不能拦住它，取决于这一项的权重
            # 够不够压住另外三项，而不只是它自己判了 0 分。
            # slack 从 22 收到 12（衰减到 0 的位置从 148° 提到 158°）。
            # 用户确认的那一簇山式 31 帧，较屈的那个膝角是 175.4~180.0；
            # 而「随意站着」那一簇是 146~160，越站越松。两组之间空着 15°，
            # 衰减区跨过 148 就等于把 160° 的松腿站姿也收进来（能拿 0.37 分、
            # 总分 0.84）。收到 12 之后真实山式最差那帧离容差边界还有 5.4°。
            Check("双腿伸直", _worse_leg_extended, 178, 8, 12, 3.0),
            Check("身体竖直", lambda v: v.vert("hip_mid", "ankle_mid"), 0, 8, 20, 2.0),
            Check("躯干竖直", _spine_up, 1.0, 0.06, 0.20, 2.0, ""),
            # 容差从 0.35 收到 0.25：0.35 时手臂抬到 0.60 都算「垂于体侧」，
            # 而用户指出的那一帧正是「随意站着手臂没有下垂」（实测 0.66）。
            Check("双臂垂于体侧", lambda v: v.dy("shoulder_mid", "wrist_mid"), 0.95, 0.25, 0.50, 1.5, ""),
        ),
        min_score=0.70,  # 站姿太常见，门槛调高，免得把过渡站立都算成山式
    ),
    # ----------------------------------------------------------------------
    # 2026-08 补的四个体式。目标值取自真实练习的实测中位数，不是插画 ——
    # 每条注释里写着实测区间，标准骨架也照同一批数据搭（见 reference.py）。
    #
    # 这四个此前一直在被邻近模板认走（幻椅式→新月式 0.97、压脚背→树式 0.78
    # 整簇 16 帧、四足跪姿→平板式 46 帧、展背式→站立前屈式 0.91）。给那些
    # 模板补门槛只让这些帧回到「未识别」；补模板才是正解，实测未覆盖率
    # 28% / 14% 的大头就是它们。
    # ----------------------------------------------------------------------
    Template(
        key="chair",
        zh="幻椅式",
        en="Chair",
        symmetric=True,
        spine_up=(0.80, 1.06),  # 直立到略前倾
        checks=(
            # 实测 117~139°（中位 127）。教科书更深（约 100~110°），容差覆盖两端。
            Check("双膝屈曲下坐", _knees_flexed, 118, 28, 42, 2.5),
            # 双脚并拢是幻椅式区别于所有弓步的定义 —— 实测 0.01~0.07，
            # 弓步是 1.93~2.04。缺了它，幻椅式恰好满足新月式全部五项检查。
            Check("双脚并拢", _ankle_spread, 0.10, 0.35, 0.40, 2.5, ""),
            Check("站姿（双脚在髋下方）", _feet_below_hips, 1.25, 0.35, 0.45, 2.0, ""),
            Check("双臂上举过头", lambda v: v.dy("wrist_mid", "shoulder_mid"), 0.90, 0.35, 0.50, 2.0, ""),
            Check("躯干接近竖直", _spine_up, 0.97, 0.14, 0.30, 1.5, ""),
        ),
        min_score=0.64,
        gates=(
            # 与山式的分水岭。山式要求双腿伸直（实测 175~180），幻椅式屈膝
            # 下坐（117~139）—— 两者其余各项（站姿、双脚并拢、躯干竖直）全同。
            Gate("双膝必须屈曲", _knees_flexed, 55, 155, 25),
            # 与弓步类的分水岭，和新月式、战士一式那条门槛方向相反、成对存在。
            Gate("双脚必须并拢", _ankle_spread, -1.0, 0.55, 0.30),
        ),
    ),
    Template(
        key="ardha_uttanasana",
        zh="展背式",
        en="Half Forward Fold",
        symmetric=True,
        # 上界不到 0：躯干确实要过水平。下界 -0.62 与站立前屈式的门槛接在一起。
        spine_up=(-0.62, 0.02),
        checks=(
            # 实测 -0.35~-0.49（中位 -0.41）。满程前屈是 -0.85 上下。
            Check("躯干折到略过水平", _spine_up, -0.42, 0.18, 0.30, 2.5, ""),
            # 头还在髋高附近才是「半程」—— 实测 0.50~0.70，满程能到 0.95 以上。
            # 这是与站立前屈式最直观的一项，权重和躯干那项等高。
            Check("头部仍在髋高附近", lambda v: v.dy("hip_mid", "nose"), 0.60, 0.25, 0.40, 2.5, ""),
            Check("双腿伸直", _legs_extended, 176, 14, 32, 2.0),
            # 双腿竖直是与下犬式的分水岭（下犬式腿与铅垂线约 28°）。
            Check("双腿竖直", lambda v: v.vert("hip_mid", "ankle_mid"), 0, 12, 26, 2.0),
            Check("双脚在髋下方（站姿）", _feet_below_hips, 1.60, 0.45, 0.55, 2.0, ""),
            # 双脚并拢是与金字塔式的分水岭，同站立前屈式。
            Check("双脚并拢", _ankle_spread, 0.12, 0.35, 0.40, 2.0, ""),
        ),
        min_score=0.64,
        gates=(
            # 和站立前屈式的「躯干必须倒垂过半」方向相反、成对存在：
            # 两个模板在竖直分量 -0.60 处一刀两断，同一个动作的两个深度。
            # 缺了它，一个满程前屈（-0.85）在本模板上「躯干」那项虽然判 0，
            # 其余五项照旧满足。
            Gate("躯干不能倒垂过半", _spine_up, -0.60, 0.10, 0.15),
            Gate("双脚必须并拢", _ankle_spread, -1.0, 0.60, 0.35),
        ),
    ),
    Template(
        key="table_top",
        zh="四足跪姿",
        en="Table Top",
        symmetric=True,
        spine_up=(-0.40, 0.45),  # 躯干接近水平
        checks=(
            # 膝压在髋正下方的地面上 —— 实测 0.87~0.98。平板式是 0.27~0.45。
            Check("膝在髋正下方（跪姿）", _knee_drop, 0.92, 0.22, 0.32, 2.5, ""),
            Check("腕在肩正下方", _lowest_wrist_drop, 0.92, 0.25, 0.40, 2.0, ""),
            Check("屈膝约 90°", _knees_flexed, 82, 25, 40, 2.0),
            Check("躯干接近水平", lambda v: v.horiz("hip_mid", "shoulder_mid"), 8, 16, 30, 2.0),
            # 小腿沿地面向后铺开，同 _back_ankle_below_knee 的道理。
            Check("小腿贴地后展", _front_shin_from_horizontal, 8, 20, 32, 1.5),
            Check("手腕不前伸也不后撤", lambda v: v.vert("s_wrist", "s_shoulder"), 0, 20, 35, 1.5),
        ),
        min_score=0.64,
        gates=(
            # 和平板式、四柱支撑式那条「膝盖不能落地」方向相反、成对存在。
            # 实测跪姿 0.87~0.98，平板 0.27~0.45，中间空着一大段。
            # 用「抬得更高的那个膝」：单腿后伸时只有一膝落地，中点算出来 0.57
            # 看着像半跪，取 min 读到 0.2，明确不是四足跪姿。和平板式那条
            # 「膝盖不能落地」取 max 成对 —— 一个问「有没有膝落地」，
            # 一个问「是不是两膝都落地」。
            Gate("两膝必须都落地", _higher_knee_drop, 0.62, 1.45, 0.25),
        ),
    ),
    Template(
        key="toe_squat",
        zh="压脚背",
        en="Toe Squat",
        symmetric=True,
        spine_up=(0.80, 1.06),  # 躯干竖直
        checks=(
            # 折到最紧 —— 实测 21~39°（中位 28）。婴儿式同样深屈，靠躯干区分。
            Check("深度屈膝坐在脚跟上", _knees_flexed, 30, 18, 30, 2.5),
            # 髋直接落在脚跟上，踝离髋很近 —— 实测 0.31~0.48。站姿是 1.0 以上。
            Check("髋落在脚跟上", _feet_below_hips, 0.38, 0.20, 0.30, 2.5, ""),
            Check("双脚并拢", _ankle_spread, 0.08, 0.25, 0.35, 2.0, ""),
            Check("躯干竖直", _spine_up, 0.96, 0.14, 0.30, 2.0, ""),
            Check("膝在髋前下方", _knee_drop, 0.24, 0.30, 0.40, 1.5, ""),
        ),
        min_score=0.64,
        # 手臂不进打分项：实测这一簇里手的位置一直在变（肘 87~172°），
        # 胸前合十、扶膝、举起都有。把它写成检查项等于罚掉一半真实帧。
        gates=(
            # 上界 52° 是为了把**蹲姿**（Malasana，脚掌落地、膝约 64°）挡在外面。
            # 两者都是「两腿都折着、躯干竖直、双脚并拢」，只差折的程度和
            # 髋的高度，所以这条门槛收得比同名 Check 的区间还紧一些。
            Gate("必须深度屈膝", _knees_flexed, 0, 52, 14),
            Gate("髋必须落在脚跟附近", _feet_below_hips, -0.5, 0.62, 0.20),
        ),
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
        gate = 1.0
        for g in template.gates:
            gate *= _orientation_factor(g.measure(view), g.lo, g.hi, g.slack)
        match = PoseMatch(
            key=template.key,
            zh=template.zh,
            en=template.en,
            side=side,
            score=(total_score / total_weight) * orientation * gate,
            checks=results,
            orientation=orientation,
            gate=gate,
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
