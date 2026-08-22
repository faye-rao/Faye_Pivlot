"""The pose library: what "correct" means for each asana.

Every pose is a list of :class:`~yoga_coach.checks.Check` rules.  The target
bands come from the cues a teacher gives in a general class, widened a little
because a webcam sees a flat projection of you: a knee angle read from the
front is a few degrees off the real one.  They are meant as a nudge towards
alignment, not as a clinical measurement -- see the caveats in the README.

Adding a pose means appending one :class:`PoseSpec` to :data:`POSES`; nothing
else in the package needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from . import metrics as m
from .checks import Check, Text
from .metrics import Metric

Side = str


@dataclass(frozen=True)
class PoseSpec:
    """A named posture plus the rules that define good alignment in it."""

    key: str
    name: Text
    sanskrit: str
    #: Where to put the camera.  The checks assume this viewpoint.
    view: Text
    checks: tuple[Check, ...]
    #: Symmetric poses are evaluated once; asymmetric ones are evaluated for
    #: both sides and the better-scoring interpretation wins.
    symmetric: bool = True
    #: One overall cue shown while holding the pose.
    cue: Text = Text("保持均匀呼吸", "Keep the breath even")
    #: Fraction of body landmarks that must be visible to judge this pose.
    min_coverage: float = 0.7

    def sides(self) -> tuple[Side, ...]:
        return ("left",) if self.symmetric else ("left", "right")


def bilateral(
    key: str,
    label: Text,
    metric_of: Callable[[Side], Metric],
    focus: Iterable[str] = (),
    **kwargs,
) -> list[Check]:
    """Build the same check for the left and the right side of the body.

    ``metric_of`` receives ``"left"`` / ``"right"`` and returns the metric for
    that side, e.g. ``lambda s: m.joint_angle(f"{s}_hip", f"{s}_knee", ...)``.
    """
    tags = {"left": ("左", "L"), "right": ("右", "R")}
    out: list[Check] = []
    for side, (tag_zh, tag_en) in tags.items():
        out.append(
            Check(
                key=f"{key}_{side}",
                label=Text(f"{label.zh}·{tag_zh}", f"{label.en} ({tag_en})"),
                metric=metric_of(side),
                focus=tuple(name.format(s=side, o=_other(side)) for name in focus),
                **kwargs,
            )
        )
    return out


def _other(side: Side) -> Side:
    return "right" if side == "left" else "left"


# --------------------------------------------------------------------------
# Rules shared by several poses
# --------------------------------------------------------------------------

def _torso_upright(limit: float = 10.0, weight: float = 1.0) -> Check:
    return Check(
        key="torso_upright",
        label=Text("躯干竖直", "Torso upright"),
        metric=m.from_vertical("mid_hip", "mid_shoulder"),
        high=limit,
        falloff=25.0,
        weight=weight,
        when_high=Text(
            "上身立直，头顶向上延展，不要前倾或后仰",
            "Stack the torso: lengthen up, stop leaning",
        ),
        focus=("mid_shoulder", "mid_hip"),
    )


def _shoulders_level(limit: float = 7.0, weight: float = 0.8) -> Check:
    return Check(
        key="shoulders_level",
        label=Text("双肩等高", "Shoulders level"),
        metric=m.absolute(m.tilt("left_shoulder", "right_shoulder")),
        high=limit,
        falloff=15.0,
        weight=weight,
        when_high=Text(
            "两侧肩膀等高，别一高一低",
            "Level the shoulders, one is riding higher",
        ),
        focus=("left_shoulder", "right_shoulder"),
    )


def _hips_level(limit: float = 8.0, weight: float = 0.9) -> Check:
    return Check(
        key="hips_level",
        label=Text("骨盆水平", "Hips level"),
        metric=m.absolute(m.tilt("left_hip", "right_hip")),
        high=limit,
        falloff=15.0,
        weight=weight,
        when_high=Text(
            "骨盆摆正，不要向一侧顶胯",
            "Square the pelvis, stop hiking one hip",
        ),
        focus=("left_hip", "right_hip"),
    )


def _straight_arms(low: float = 165.0, weight: float = 0.7) -> list[Check]:
    return bilateral(
        key="arm_straight",
        label=Text("手臂伸直", "Arm straight"),
        metric_of=lambda s: m.joint_angle(f"{s}_shoulder", f"{s}_elbow", f"{s}_wrist"),
        low=low,
        falloff=35.0,
        weight=weight,
        when_low=Text("手臂伸直，从肩到指尖主动延展", "Straighten the arm all the way"),
        focus=("{s}_elbow",),
    )


def _stance_width(low: float, weight: float = 0.8) -> Check:
    return Check(
        key="stance_width",
        label=Text("步距", "Stance width"),
        metric=m.span("left_ankle", "right_ankle"),
        low=low,
        falloff=0.6,
        weight=weight,
        unit="×",
        when_low=Text("步子再迈开一些，站得更宽更稳", "Step wider, lengthen the stance"),
        focus=("left_ankle", "right_ankle"),
    )


# --------------------------------------------------------------------------
# 山式 Mountain
# --------------------------------------------------------------------------

MOUNTAIN = PoseSpec(
    key="mountain",
    name=Text("山式", "Mountain"),
    sanskrit="Tadasana",
    view=Text("正面对着摄像头，全身入镜", "Face the camera, whole body in frame"),
    cue=Text("双脚扎根，头顶向上", "Root down through the feet, crown lifts"),
    symmetric=True,
    checks=(
        *bilateral(
            key="knee_straight",
            label=Text("膝盖伸直", "Knee straight"),
            metric_of=lambda s: m.joint_angle(f"{s}_hip", f"{s}_knee", f"{s}_ankle"),
            low=170.0,
            falloff=30.0,
            when_low=Text("双腿伸直，大腿前侧收紧", "Straighten the legs, engage the thighs"),
            focus=("{s}_knee",),
        ),
        _torso_upright(limit=8.0),
        _shoulders_level(),
        _hips_level(),
        Check(
            key="head_centred",
            label=Text("头部居中", "Head centred"),
            metric=m.horizontal_gap("nose", "mid_shoulder"),
            high=0.14,
            falloff=0.25,
            weight=0.6,
            unit="×",
            when_high=Text("头回到两肩正中，下巴微收", "Centre the head over the shoulders"),
            focus=("nose",),
        ),
    ),
)

# --------------------------------------------------------------------------
# 树式 Tree -- side = the standing leg
# --------------------------------------------------------------------------

TREE = PoseSpec(
    key="tree",
    name=Text("树式", "Tree"),
    sanskrit="Vrksasana",
    view=Text("正面对着摄像头，全身入镜", "Face the camera, whole body in frame"),
    cue=Text("目光找一个不动的点，呼吸放慢", "Fix your gaze, slow the breath"),
    symmetric=False,
    checks=(
        Check(
            key="standing_leg_straight",
            label=Text("支撑腿伸直", "Standing leg straight"),
            metric=m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle"),
            low=168.0,
            falloff=30.0,
            weight=1.2,
            when_low=Text("支撑腿伸直但不锁死膝盖", "Straighten the standing leg without locking"),
            focus=("{s}_knee",),
        ),
        Check(
            key="standing_leg_vertical",
            label=Text("支撑腿竖直", "Standing leg vertical"),
            metric=m.from_vertical("{s}_ankle", "{s}_hip"),
            high=12.0,
            falloff=20.0,
            when_high=Text("重心压回支撑脚正上方", "Bring your weight over the standing foot"),
            focus=("{s}_ankle", "{s}_hip"),
        ),
        Check(
            key="foot_lifted",
            label=Text("抬起脚离地", "Foot lifted"),
            metric=m.vertical_gap("{o}_ankle", "{s}_ankle"),
            low=0.45,
            falloff=0.50,
            weight=1.2,
            unit="×",
            when_low=Text(
                "抬起的脚离开地面，踩到支撑腿的小腿或大腿内侧",
                "Lift the foot off the floor onto the standing leg",
            ),
            focus=("{o}_ankle",),
        ),
        Check(
            key="lifted_knee_open",
            label=Text("抬腿膝盖外开", "Lifted knee open"),
            metric=m.horizontal_gap("{o}_knee", "mid_hip"),
            low=0.30,
            falloff=0.45,
            unit="×",
            when_low=Text("抬起的膝盖向侧后方打开，转开髋", "Open the lifted knee out to the side"),
            focus=("{o}_knee",),
        ),
        Check(
            key="foot_off_knee",
            label=Text("脚掌避开膝关节", "Foot clear of knee"),
            metric=m.span("{o}_ankle", "{s}_knee"),
            low=0.28,
            falloff=0.30,
            weight=1.3,
            unit="×",
            when_low=Text(
                "脚掌别踩在支撑腿膝盖上，移到大腿内侧或小腿",
                "Move the foot off the knee joint -- thigh or calf, never the knee",
            ),
            focus=("{o}_ankle", "{s}_knee"),
        ),
        _hips_level(limit=8.0, weight=1.0),
        _torso_upright(limit=9.0),
    ),
)

# --------------------------------------------------------------------------
# 战士二 Warrior II -- side = the front leg
# --------------------------------------------------------------------------

WARRIOR_II = PoseSpec(
    key="warrior2",
    name=Text("战士二式", "Warrior II"),
    sanskrit="Virabhadrasana II",
    view=Text("摄像头放在正前方，侧身入镜", "Camera in front of you, body side-on"),
    cue=Text("目光越过前手中指", "Gaze past the front middle finger"),
    symmetric=False,
    checks=(
        Check(
            key="front_knee_bend",
            label=Text("前膝屈度", "Front knee bend"),
            metric=m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle"),
            low=80.0,
            high=110.0,
            falloff=35.0,
            weight=1.4,
            when_low=Text("前膝屈得太深了，小腿回到垂直", "Too deep -- shin back to vertical"),
            when_high=Text("前腿再屈深一点，大腿趋向平行地面", "Bend the front knee deeper"),
            focus=("{s}_knee",),
        ),
        Check(
            key="front_knee_over_ankle",
            label=Text("前膝对准脚踝", "Knee over ankle"),
            metric=m.horizontal_gap("{s}_knee", "{s}_ankle"),
            high=0.22,
            falloff=0.35,
            weight=1.4,
            unit="×",
            when_high=Text(
                "前膝移回脚踝正上方，别超过脚尖",
                "Stack the front knee over the ankle, not past the toes",
            ),
            focus=("{s}_knee", "{s}_ankle"),
        ),
        Check(
            key="back_leg_straight",
            label=Text("后腿伸直", "Back leg straight"),
            metric=m.joint_angle("{o}_hip", "{o}_knee", "{o}_ankle"),
            low=165.0,
            falloff=35.0,
            weight=1.1,
            when_low=Text("后腿蹬直，后脚外缘压实地面", "Straighten the back leg, press the outer edge down"),
            focus=("{o}_knee",),
        ),
        Check(
            key="arms_level",
            label=Text("双臂成一条水平线", "Arms level"),
            metric=m.from_horizontal("left_wrist", "right_wrist"),
            high=12.0,
            falloff=25.0,
            weight=1.1,
            when_high=Text("双臂展平到与地面平行，两侧等高", "Bring both arms level with the floor"),
            focus=("left_wrist", "right_wrist"),
        ),
        *_straight_arms(),
        _torso_upright(limit=12.0, weight=1.2),
        _stance_width(low=1.15),
    ),
)

# --------------------------------------------------------------------------
# 战士一 Warrior I -- side = the front leg
# --------------------------------------------------------------------------

WARRIOR_I = PoseSpec(
    key="warrior1",
    name=Text("战士一式", "Warrior I"),
    sanskrit="Virabhadrasana I",
    view=Text("摄像头放在侧前方，全身入镜", "Camera at your front-side, whole body in frame"),
    cue=Text("后脚跟压地，胸腔上提", "Back heel down, chest lifts"),
    symmetric=False,
    checks=(
        Check(
            key="front_knee_bend",
            label=Text("前膝屈度", "Front knee bend"),
            metric=m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle"),
            low=80.0,
            high=115.0,
            falloff=35.0,
            weight=1.4,
            when_low=Text("前膝屈得太深了，小腿回到垂直", "Too deep -- shin back to vertical"),
            when_high=Text("前腿再屈深一点", "Bend the front knee deeper"),
            focus=("{s}_knee",),
        ),
        Check(
            key="front_knee_over_ankle",
            label=Text("前膝对准脚踝", "Knee over ankle"),
            metric=m.horizontal_gap("{s}_knee", "{s}_ankle"),
            high=0.25,
            falloff=0.35,
            weight=1.3,
            unit="×",
            when_high=Text("前膝退回脚踝正上方", "Stack the front knee over the ankle"),
            focus=("{s}_knee", "{s}_ankle"),
        ),
        Check(
            key="back_leg_straight",
            label=Text("后腿伸直", "Back leg straight"),
            metric=m.joint_angle("{o}_hip", "{o}_knee", "{o}_ankle"),
            low=160.0,
            falloff=35.0,
            when_low=Text("后腿蹬直", "Straighten the back leg"),
            focus=("{o}_knee",),
        ),
        *bilateral(
            key="arm_overhead",
            label=Text("手臂上举", "Arm overhead"),
            metric_of=lambda s: m.from_vertical(f"{s}_shoulder", f"{s}_wrist"),
            high=28.0,
            falloff=40.0,
            weight=1.0,
            when_high=Text("双臂向上延展，指尖指向天空", "Reach both arms straight overhead"),
            focus=("{s}_wrist",),
        ),
        *_straight_arms(low=160.0, weight=0.6),
        _torso_upright(limit=18.0, weight=1.1),
        _stance_width(low=1.0),
    ),
)

# --------------------------------------------------------------------------
# 三角式 Triangle -- side = the front leg (the one you bend towards)
# --------------------------------------------------------------------------

TRIANGLE = PoseSpec(
    key="triangle",
    name=Text("三角式", "Triangle"),
    sanskrit="Trikonasana",
    view=Text("摄像头放在正前方，侧身入镜", "Camera in front of you, body side-on"),
    cue=Text("从髋部折叠，脊柱保持长", "Hinge from the hip, keep the spine long"),
    symmetric=False,
    checks=(
        Check(
            key="front_leg_straight",
            label=Text("前腿伸直", "Front leg straight"),
            metric=m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle"),
            low=163.0,
            falloff=35.0,
            weight=1.2,
            when_low=Text("前腿伸直，膝盖别塌向内侧", "Straighten the front leg"),
            focus=("{s}_knee",),
        ),
        Check(
            key="back_leg_straight",
            label=Text("后腿伸直", "Back leg straight"),
            metric=m.joint_angle("{o}_hip", "{o}_knee", "{o}_ankle"),
            low=163.0,
            falloff=35.0,
            when_low=Text("后腿也蹬直，两腿同时发力", "Straighten the back leg too"),
            focus=("{o}_knee",),
        ),
        Check(
            key="torso_side_bend",
            label=Text("躯干侧倾", "Side bend"),
            metric=m.from_vertical("mid_hip", "mid_shoulder"),
            low=35.0,
            high=75.0,
            falloff=30.0,
            weight=1.3,
            when_low=Text("上身继续向侧下方延展", "Reach further out over the front leg"),
            when_high=Text("别塌下去，胸腔向上旋开", "Stop collapsing -- rotate the chest open"),
            focus=("mid_shoulder", "mid_hip"),
        ),
        Check(
            key="arms_in_line",
            label=Text("双臂成一条直线", "Arms in one line"),
            metric=m.from_vertical("{s}_wrist", "{o}_wrist"),
            high=22.0,
            falloff=35.0,
            weight=1.2,
            when_high=Text("上方手臂垂直向上，与下方手成一条线", "Stack the top arm over the bottom one"),
            focus=("{s}_wrist", "{o}_wrist"),
        ),
        *_straight_arms(),
        _stance_width(low=1.15),
    ),
)

# --------------------------------------------------------------------------
# 下犬式 Downward-Facing Dog -- side = the side facing the camera
# --------------------------------------------------------------------------

DOWN_DOG = PoseSpec(
    key="downdog",
    name=Text("下犬式", "Downward Dog"),
    sanskrit="Adho Mukha Svanasana",
    view=Text("摄像头放在身体侧面", "Camera to the side of your mat"),
    cue=Text("身体成倒 V，坐骨指向天花板", "Make an upside-down V, sit bones to the ceiling"),
    symmetric=False,
    min_coverage=0.6,
    checks=(
        Check(
            key="hip_angle",
            label=Text("髋部折叠角", "Hip angle"),
            metric=m.joint_angle("{s}_shoulder", "{s}_hip", "{s}_knee"),
            # No lower bound on purpose.  Pushing the hips higher and further
            # back closes this angle, and that *is* the pose -- a practitioner
            # folding to 45 degrees was being told to ease off while their
            # spine measured longer than at 90.  Folding deep is only a
            # problem when the back rounds, and `back_long` below is the check
            # that actually watches for that.
            high=100.0,
            falloff=35.0,
            weight=1.4,
            when_high=Text("臀部向上向后推高，做出倒 V", "Push the hips up and back into an inverted V"),
            focus=("{s}_hip",),
        ),
        Check(
            key="back_long",
            label=Text("背部延展", "Long back"),
            metric=m.joint_angle("{s}_wrist", "{s}_shoulder", "{s}_hip"),
            low=155.0,
            falloff=40.0,
            weight=1.3,
            when_low=Text("耳朵回到两臂之间，从手到髋拉成一条线", "Ears between the arms, one line from hands to hips"),
            focus=("{s}_shoulder",),
        ),
        *bilateral(
            key="leg_straight",
            label=Text("腿部伸展", "Leg extended"),
            metric_of=lambda s: m.joint_angle(f"{s}_hip", f"{s}_knee", f"{s}_ankle"),
            low=155.0,
            falloff=40.0,
            weight=0.9,
            when_low=Text("腿后侧紧就微屈膝，但主动向上推坐骨", "Micro-bend is fine, but keep lifting the sit bones"),
            focus=("{s}_knee",),
        ),
        *_straight_arms(low=163.0, weight=1.0),
    ),
)

# --------------------------------------------------------------------------
# 平板支撑 Plank -- side = the side facing the camera
# --------------------------------------------------------------------------

PLANK = PoseSpec(
    key="plank",
    name=Text("平板支撑", "Plank"),
    sanskrit="Phalakasana",
    view=Text("摄像头放在身体侧面，与地面同高", "Camera to your side, at floor height"),
    cue=Text("核心收紧，脚跟向后蹬", "Draw the belly in, press the heels back"),
    symmetric=False,
    min_coverage=0.6,
    checks=(
        Check(
            key="body_horizontal",
            label=Text("身体接近水平", "Body horizontal"),
            metric=m.from_horizontal("{s}_shoulder", "{s}_ankle"),
            high=18.0,
            falloff=30.0,
            weight=1.3,
            when_high=Text(
                "肩、髋、脚踝拉成接近水平的一条线",
                "Bring shoulders, hips and ankles onto one near-level line",
            ),
            focus=("{s}_shoulder", "{s}_ankle"),
        ),
        Check(
            key="body_line",
            label=Text("身体一条线", "Body in one line"),
            metric=m.line_offset("{s}_shoulder", "{s}_hip", "{s}_ankle"),
            low=-0.07,
            high=0.10,
            falloff=0.20,
            weight=1.6,
            unit="×",
            when_low=Text("髋部下沉了，收紧核心把臀部抬回一条线", "Hips are sagging -- engage the core and lift them"),
            when_high=Text("臀部翘太高了，放低到肩踝一条线", "Hips are too high -- lower them into line"),
            focus=("{s}_hip",),
        ),
        Check(
            key="shoulder_over_wrist",
            label=Text("肩在腕正上方", "Shoulder over wrist"),
            metric=m.horizontal_gap("{s}_shoulder", "{s}_wrist"),
            high=0.20,
            falloff=0.35,
            weight=1.2,
            unit="×",
            when_high=Text("肩膀推到手腕正上方", "Bring the shoulders over the wrists"),
            focus=("{s}_shoulder", "{s}_wrist"),
        ),
        *_straight_arms(low=160.0, weight=1.0),
        *bilateral(
            key="leg_straight",
            label=Text("腿伸直", "Leg straight"),
            metric_of=lambda s: m.joint_angle(f"{s}_hip", f"{s}_knee", f"{s}_ankle"),
            low=165.0,
            falloff=30.0,
            weight=0.9,
            when_low=Text("双腿蹬直，脚跟向后推", "Straighten the legs, press the heels back"),
            focus=("{s}_knee",),
        ),
    ),
)

# --------------------------------------------------------------------------
# 椅子式 Chair -- side = the side facing the camera
# --------------------------------------------------------------------------

CHAIR = PoseSpec(
    key="chair",
    name=Text("椅子式", "Chair"),
    sanskrit="Utkatasana",
    view=Text("摄像头放在身体侧面", "Camera to the side of your mat"),
    cue=Text("重心落在脚跟，像坐进一把椅子", "Weight in the heels, sit back into a chair"),
    symmetric=False,
    checks=(
        Check(
            key="knee_bend",
            label=Text("屈膝深度", "Knee bend"),
            metric=m.joint_angle("{s}_hip", "{s}_knee", "{s}_ankle"),
            low=95.0,
            high=145.0,
            falloff=35.0,
            weight=1.4,
            when_low=Text("蹲得有点低了，膝盖压力大，稍微起来一些", "A bit too low for the knees -- come up slightly"),
            when_high=Text("再向下坐深一些", "Sit down deeper"),
            focus=("{s}_knee",),
        ),
        Check(
            key="knee_behind_toes",
            label=Text("膝不过脚尖", "Knees behind toes"),
            metric=m.horizontal_gap("{s}_knee", "{s}_ankle"),
            high=0.35,
            falloff=0.40,
            weight=1.2,
            unit="×",
            when_high=Text("重心后移到脚跟，膝盖别过多超过脚尖", "Shift back into the heels, knees behind the toes"),
            focus=("{s}_knee", "{s}_ankle"),
        ),
        Check(
            key="torso_lean",
            label=Text("上身前倾角", "Torso lean"),
            metric=m.from_vertical("mid_hip", "mid_shoulder"),
            low=12.0,
            high=45.0,
            falloff=25.0,
            weight=1.1,
            when_low=Text("上身可以再向前倾一点，与大腿形成夹角", "Hinge the torso forward a little more"),
            when_high=Text("胸腔上提，别过度前倾塌腰", "Lift the chest, you are folding too far"),
            focus=("mid_shoulder", "mid_hip"),
        ),
        *bilateral(
            key="arm_overhead",
            label=Text("手臂上举", "Arm overhead"),
            metric_of=lambda s: m.from_vertical(f"{s}_shoulder", f"{s}_wrist"),
            high=38.0,
            falloff=45.0,
            weight=0.8,
            when_high=Text("双臂向斜上方延展，肩膀放松下沉", "Reach the arms up, shoulders soft"),
            focus=("{s}_wrist",),
        ),
    ),
)


POSES: tuple[PoseSpec, ...] = (
    MOUNTAIN,
    TREE,
    WARRIOR_II,
    WARRIOR_I,
    TRIANGLE,
    DOWN_DOG,
    PLANK,
    CHAIR,
)

POSES_BY_KEY: dict[str, PoseSpec] = {p.key: p for p in POSES}


def get_pose(key: str) -> PoseSpec:
    try:
        return POSES_BY_KEY[key]
    except KeyError:
        known = ", ".join(POSES_BY_KEY)
        raise KeyError(f"unknown pose {key!r}; available: {known}") from None
