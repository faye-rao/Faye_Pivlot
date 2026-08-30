"""对一批静态体式图片跑姿态估计，看模板会把它们认成什么。

**先说清楚这不是训练。** `yoga_grid` 的体式识别是写死的几何规则
（`poses.py`），没有从图片里学参数的环节，多放几百张图进来也不会让它变准。
图片能起的作用只有两个，都要人来做决定：

1. **决定该加哪些模板** —— 看图库里有、模板里没有的体式。
2. **校准已有模板的目标值与容差** —— 只有**真人照片或视频帧**才算数。
   插画、线稿、剪影、3D 渲染会把已经修好的 bug 装回来，原因写在
   `reference_images/README.md`：最初的标准骨架照插画搭，三角伸展式的躯干
   竖直分量取到 0.57，而真实练习实测是 -0.01，结果正确的模板先把自己排除在
   朝向门槛之外。

这个模块做的是**取证**，不是判决：把每张图的检出情况、关键点置信度、当前
模板的打分排名摆出来，判断留给人。唯一能自动下结论的是「检测不到人体」——
那张图确定没用。
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import compat, heads
from . import landmarks as L
from .poses import TEMPLATES, PoseMatch, score_by_key

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

# 文件名/目录名里出现这些字样就认为标注的是对应模板。值为 None 表示
# **这个体式故意没有模板** —— 报出「库里没有」比硬套一个近似的模板有用。
#
# 匹配规则是「最长别名优先」，这样 `Ardha Uttanasana - Half Forward Fold`
# 会先命中 "ardha uttanasana"（→ None，半前屈没有模板）而不是更短的
# "uttanasana"。`reference_images/README.md` 记的两对陷阱同理：
# Hanumanasana 是全劈叉不是半神猴，Viparita Virabhadrasana 是反战士不是反板式。
ALIASES: dict[str, str | None] = {
    # —— 有模板的 ——
    "downward facing dog": "downdog", "downwardfacing dog": "downdog",
    "adho mukha svanasana": "downdog", "down dog": "downdog",
    "downdog": "downdog", "下犬": "downdog",
    "phalakasana": "plank", "plank": "plank", "平板": "plank", "板式": "plank",
    "virabhadrasana ii": "warrior2", "warrior ii": "warrior2",
    "warrior 2": "warrior2", "战士二": "warrior2",
    "virabhadrasana i": "warrior1", "warrior i": "warrior1",
    "warrior 1": "warrior1", "战士一": "warrior1",
    "virabhadrasana iii": "warrior3", "warrior iii": "warrior3",
    "warrior 3": "warrior3", "战士三": "warrior3",
    "trikonasana": "triangle", "triangle": "triangle", "三角": "triangle",
    "utthita parsvakonasana": "parsvakonasana", "parsvakonasana": "parsvakonasana",
    "side angle": "parsvakonasana", "侧角": "parsvakonasana",
    "urdhva mukha svanasana": "updog", "upward facing dog": "updog",
    "upwards facing dog": "updog", "up dog": "updog", "updog": "updog",
    "上犬": "updog",
    "anjaneyasana": "anjaneyasana", "low lunge": "anjaneyasana",
    "ashta chandrasana": "anjaneyasana", "crescent moon": "anjaneyasana",
    "新月": "anjaneyasana",
    "ardha hanumanasana": "ardha_hanumanasana", "half splits": "ardha_hanumanasana",
    "half split": "ardha_hanumanasana", "半神猴": "ardha_hanumanasana",
    "eka pada rajakapotasana": "pigeon", "pigeon": "pigeon", "鸽子": "pigeon",
    "chaturanga dandasana": "chaturanga", "chaturanga": "chaturanga",
    "four limb staff": "chaturanga", "四柱": "chaturanga",
    "vasisthasana": "side_plank", "side plank": "side_plank", "侧板": "side_plank",
    "purvottanasana": "reverse_plank", "reverse plank": "reverse_plank",
    "反板": "reverse_plank",
    "uttanasana": "uttanasana", "forward fold": "uttanasana",
    "standing forward bend": "uttanasana", "站立前屈": "uttanasana",
    "前屈": "uttanasana",
    "parsvottanasana": "parsvottanasana", "intense side stretch": "parsvottanasana",
    "pyramid": "parsvottanasana", "金字塔": "parsvottanasana",
    "vrksasana": "tree", "tree": "tree", "树式": "tree",
    "setu bandha sarvangasana": "bridge", "setu bandhasana": "bridge",
    "setu bandha": "bridge", "bridge": "bridge", "桥式": "bridge",
    "balasana": "child", "childs pose": "child", "child pose": "child",
    "child": "child", "婴儿": "child",
    "tadasana": "mountain", "mountain": "mountain", "山式": "mountain",
    # —— 故意没有模板，别被更短的别名认走 ——
    "ardha uttanasana": None, "half forward fold": None,
    "hanumanasana": None, "monkey or splits": None, "splits": None,
    "viparita virabhadrasana": None, "reverse warrior": None,
    "ardha chandrasana": None, "half moon": None,
    # Kapotasana 是鸽王式（深后弯），和 pigeon 模板对应的
    # Eka Pada Rajakapotasana 不是一个体式。
    "kapotasana": None, "king pigeon": None,
}

_ALIAS_ORDER = sorted(ALIASES, key=len, reverse=True)

# 梵文体式名靠限定词区分体式，而不是修饰同一个体式：
#
#     Vrksasana                 树式
#     Adho Mukha Vrksasana      手倒立      —— 词根相同，两个体式
#     Chaturanga Dandasana      四柱支撑式
#     Utthita Chaturanga Dandasana  直臂斜板
#
# 别名表列过的组合由「最长优先」挡住；没列过的由 _unconsumed_qualifier 挡住。
# 收录标准是**这个词一出现就换了体式**。「extended」不在其列：
# Extended Triangle 就是三角伸展式本身，收进来会误伤。
QUALIFIERS: tuple[str, ...] = (
    "adho mukha", "urdhva", "parivrtta", "parivritta", "revolved",
    "utthita", "upavistha", "supta", "ardha", "eka pada", "viparita",
    "salamba", "niralamba", "prasarita", "parsva", "baddha",
)

MATCHED = "matched"        # 认出体式，且它有模板
UNMODELLED = "unmodelled"  # 认出体式，但**故意**没有模板
QUALIFIED = "qualified"    # 名字里有没消化掉的限定词，不敢下判断
UNKNOWN = "unknown"        # 名字没对上任何东西


@dataclass(frozen=True)
class Label:
    """文件名认出了什么。

    四种状态刻意分开，因为**对使用者的意义完全不同**：认对了、库里没有这个
    体式、名字含糊不敢认、根本没认出来 —— 混成一个「没有 key」会把「我不知道」
    伪装成「库里没有」。
    """

    key: str | None
    status: str
    detail: str = ""

    @property
    def known(self) -> bool:
        """名字对上了某个体式（不管有没有模板）。"""
        return self.status in (MATCHED, UNMODELLED)


@dataclass
class ImageProbe:
    """一张图跑完姿态估计后能说的全部事情。"""

    path: Path
    width: int = 0
    height: int = 0
    lm: np.ndarray | None = None
    norm: np.ndarray | None = None
    visibility: float = 0.0
    face_confidence: float = 0.0
    spine_up: float = float("nan")
    labeled: str | None = None          # 从文件名猜到的模板 key
    label_status: str = UNKNOWN         # 见 Label.status
    label_detail: str = ""
    ranked: list[tuple[str, float]] = field(default_factory=list)
    error: str = ""

    @property
    def detected(self) -> bool:
        return self.lm is not None

    @property
    def top(self) -> tuple[str, float] | None:
        return self.ranked[0] if self.ranked else None

    @property
    def agrees(self) -> bool | None:
        """标注体式是否就是模板打分最高的那个。无标注/未检出时为 None。"""
        if not self.labeled or not self.ranked:
            return None
        return self.ranked[0][0] == self.labeled

    def score_of(self, key: str) -> float:
        for k, s in self.ranked:
            if k == key:
                return s
        return float("nan")


def _fold(text: str) -> str:
    """折叠成可比对的形式：小写、去重音、非字母数字与中文一律变空格。"""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.lower()
    text = re.sub(r"[^0-9a-z一-鿿]+", " ", text)
    return f" {' '.join(text.split())} "


def label_from_path(path: Path, root: Path | None = None) -> Label:
    """从文件名和上级目录名猜这张图标的是哪个体式。"""
    parts = [path.stem]
    parent = path.parent
    if root is not None:
        try:
            parts.extend(p.name for p in path.relative_to(root).parents if p.name)
        except ValueError:
            parts.append(parent.name)
    else:
        parts.append(parent.name)

    haystack = _fold(" ".join(parts))
    for alias in _ALIAS_ORDER:
        folded = _fold(alias).strip()
        if not _contains(haystack, folded):
            continue
        key = ALIASES[alias]
        if key is None:
            # 已经标着「没有模板」了，限定词再挡一次没有意义 —— 这道闸防的是
            # **认错成某个模板**，而这里根本没给出模板。
            return Label(None, UNMODELLED, alias)
        leftover = _unconsumed_qualifier(haystack, folded)
        if leftover:
            return Label(None, QUALIFIED, leftover)
        return Label(key, MATCHED, alias)
    return Label(None, UNKNOWN, "")


def _unconsumed_qualifier(haystack: str, matched: str) -> str:
    """名字里有没有**没被命中的别名吃掉**的限定词。

    梵文体式名靠限定词区分：`Vrksasana` 是树式，`Adho Mukha Vrksasana` 是
    手倒立 —— 两个体式，共用一个词根。别名表按最长优先匹配，只能挡住**表里
    列过**的那些组合；没列过的限定词会安静地被忽略，于是手倒立被标成树式。

    实际发生过，就在用户第一次跑自己的素材时（`Adho Mukha Vrksasana.jpg`
    标成树式、`Utthita Chaturanga Dandasana.jpg` 标成四柱支撑式，而模板判它
    是直臂斜板 —— 模板是对的，我的标注是错的）。穷举所有组合是补不完的，
    所以反过来做：**命中的别名没吃掉限定词，就不下判断。**
    """
    for q in QUALIFIERS:
        if _contains(haystack, q) and not _contains(f" {matched} ", q):
            return q
    return ""


def _has_cjk(text: str) -> bool:
    return any("一" <= c <= "鿿" for c in text)


def _contains(haystack: str, needle: str) -> bool:
    """别名是否出现在名字里。

    英文按**整词**匹配，否则 `treetop` 会命中 `tree`、`Warrior III` 会命中
    `Warrior I`（罗马数字是前缀关系，这个坑是真的）。中文不能这么办 ——
    「下犬式」里的「下犬」两边根本没有空格，按整词匹配会一个都认不出来。
    """
    if not needle:
        return False
    if _has_cjk(needle):
        return needle in haystack
    return f" {needle} " in haystack


def iter_images(root: Path) -> list[Path]:
    """递归找出目录里的图片，按路径排序。"""
    if root.is_file():
        return [root] if root.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _rank_all(norm: np.ndarray) -> list[tuple[str, float]]:
    """所有模板的原始得分，从高到低，**不设 min_score 门槛**。

    看的是「差多少才够得着」，所以擦边和差很远必须能区分开；套上门槛就
    全归成「没匹配上」，正好把要找的信息抹掉。
    """
    scores: list[tuple[str, float]] = []
    for template in TEMPLATES:
        match: PoseMatch | None = score_by_key(norm, template.key)
        if match is not None:
            scores.append((template.key, match.score))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores


def probe_images(
    paths: list[Path],
    model_path: Path,
    root: Path | None = None,
    work_size: int = 720,
    min_confidence: float = 0.5,
    progress: bool = True,
) -> list[ImageProbe]:
    """逐张跑姿态估计并按当前模板打分。"""
    import mediapipe as mp

    vision = mp.tasks.vision
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
    )

    probes: list[ImageProbe] = []
    landmarker = vision.PoseLandmarker.create_from_options(options)
    try:
        for n, path in enumerate(paths, 1):
            label = label_from_path(path, root)
            probe = ImageProbe(
                path=path, labeled=label.key,
                label_status=label.status, label_detail=label.detail,
            )

            bgr = compat.imread(path)
            if bgr is None:
                probe.error = "读不出来（文件损坏或格式不受支持）"
                probes.append(probe)
                continue

            small = _resize_long_side(bgr, work_size)
            h, w = small.shape[:2]
            probe.width, probe.height = w, h
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            image = mp.Image(
                image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)
            )
            result = landmarker.detect(image)

            if result.pose_landmarks:
                person = result.pose_landmarks[0]
                lm = np.array(
                    [(p.x, p.y, min(p.visibility, p.presence)) for p in person],
                    dtype=np.float64,
                )
                probe.lm = lm
                probe.norm = L.normalize(L.to_pixels(lm, w, h))
                probe.visibility = L.mean_visibility(lm)
                probe.face_confidence = heads.face_confidence(lm)
                probe.spine_up = _spine_up_of(probe.norm)
                probe.ranked = _rank_all(probe.norm)

            probes.append(probe)
            if progress and n % 10 == 0:
                print(f"  已处理 {n}/{len(paths)}", file=sys.stderr)
    finally:
        landmarker.close()
    return probes


def _spine_up_of(norm: np.ndarray) -> float:
    """躯干竖直分量：+1 直立，0 水平，-1 倒立。和 ``poses._spine_up`` 同定义。"""
    from .landmarks import PoseView

    return PoseView(norm, "left").dy("shoulder_mid", "hip_mid")


def _resize_long_side(image: np.ndarray, long_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    longest = max(h, w)
    if long_side <= 0 or longest <= long_side:
        return image
    scale = long_side / longest
    return cv2.resize(
        image, (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def _display_width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _zh(key: str) -> str:
    for t in TEMPLATES:
        if t.key == key:
            return t.zh
    return key


def format_table(probes: list[ImageProbe], root: Path) -> str:
    rows: list[list[str]] = []
    for p in probes:
        try:
            name = str(p.path.relative_to(root))
        except ValueError:
            name = p.path.name
        if p.error:
            rows.append([name, "读失败", "", "", "", "", p.error])
            continue
        if not p.detected:
            rows.append([name, "未检出", "", "", "", "", "确定不可用"])
            continue

        top = p.top
        top_text = f"{_zh(top[0])} {top[1]:.2f}" if top else "—"
        if p.labeled:
            mine = p.score_of(p.labeled)
            label_text = _zh(p.labeled)
            verdict = "一致" if p.agrees else f"不一致（标注体式得 {mine:.2f}）"
        elif p.label_status == UNMODELLED:
            label_text = "无对应模板"
            verdict = "只能用来决定要不要加模板"
        elif p.label_status == QUALIFIED:
            label_text = f"名字含「{p.label_detail}」"
            verdict = "限定词换了体式，没敢认 —— 请你确认这是什么"
        else:
            label_text = "认不出名字"
            verdict = "文件名没对上任何体式"
        rows.append([
            name, "检出", f"{p.visibility:.2f}", f"{p.face_confidence:.2f}",
            f"{p.spine_up:+.2f}", f"{label_text} → {top_text}", verdict,
        ])

    header = ["文件", "检出", "可见度", "人脸置信", "躯干竖直", "标注 → 模板判定", "结论"]
    widths = [
        max(_display_width(header[i]), max((_display_width(r[i]) for r in rows), default=0))
        for i in range(len(header))
    ]
    lines = ["  ".join(_pad(header[i], widths[i]) for i in range(len(header))).rstrip()]
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append("  ".join(_pad(r[i], widths[i]) for i in range(len(header))).rstrip())
    return "\n".join(lines)


def summarize(probes: list[ImageProbe]) -> str:
    total = len(probes)
    detected = [p for p in probes if p.detected]
    unusable = total - len(detected)
    labeled = [p for p in detected if p.labeled]
    agree = [p for p in labeled if p.agrees]
    suspicious = [p for p in detected if p.face_confidence >= 0.90]

    lines = [
        f"共 {total} 张，检出人体 {len(detected)} 张，检不出 {unusable} 张。",
    ]
    if unusable:
        lines.append(
            f"  检不出的那 {unusable} 张确定不可用 —— 连人都找不到，别的量也没有理由相信。"
        )
    if labeled:
        lines.append(
            f"  文件名认得出体式的 {len(labeled)} 张里，模板判定与标注一致 {len(agree)} 张，"
            f"不一致 {len(labeled) - len(agree)} 张。"
        )
        if len(agree) < len(labeled):
            lines.append(
                "  不一致的既可能是模板容差不对，也可能是这张图本身不适合校准，"
                "先按下面这条把图源分清楚再说。"
            )
    if suspicious:
        lines.append(
            f"  有 {len(suspicious)} 张的人脸关键点置信度 ≥ 0.90 —— **这个数不能当证据**。"
        )
        lines.append(
            "  visibility/presence 回答的是「这个部位在画面内且没被遮挡」，不是「这里真的有」。"
        )
        lines.append(
            "  自己去看一眼那几张图有没有画五官：如果没画脸还给 1.00，那么它给别的"
            "关键点的坐标同样没有理由相信。"
        )
    lines.append("")
    lines.append(
        "能不能用来校准容差，判据是**是不是真人照片或视频帧**，不是画得准不准 ——"
    )
    lines.append(
        "  插画画的是「体式应该长什么样」的理想化印象，和真人做同一个体式时的关节角度"
        "常常差很远。"
    )
    lines.append("  详见 reference_images/README.md。")
    return "\n".join(lines)


def dump(probes: list[ImageProbe], path: Path, root: Path) -> None:
    """把骨架和打分写成 JSON，供离线复算和分享。

    存的是**归一化坐标**（33×2，髋心为原点、躯干长为单位）加原始关键点
    （33×3，含置信度），不含像素 —— 既够复算模板，也不用把图片本身传出去。
    """
    entries = []
    for p in probes:
        try:
            name = str(p.path.relative_to(root))
        except ValueError:
            name = p.path.name
        entry: dict = {
            "file": name,
            "labeled": p.labeled,
            "label_status": p.label_status,
            "label_detail": p.label_detail,
            "detected": p.detected,
        }
        if p.error:
            entry["error"] = p.error
        if p.detected:
            entry.update({
                "width": p.width,
                "height": p.height,
                "visibility": round(p.visibility, 4),
                "face_confidence": round(p.face_confidence, 4),
                "spine_up": round(p.spine_up, 4),
                "ranked": [[k, round(s, 4)] for k, s in p.ranked],
                "landmarks": [[round(float(v), 5) for v in row] for row in p.lm],
                "norm": [[round(float(v), 5) for v in row] for row in p.norm],
            })
        entries.append(entry)

    payload = {
        "version": 2,
        "root": str(root),
        "templates": [t.key for t in TEMPLATES],
        "images": entries,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
