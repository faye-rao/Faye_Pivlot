"""练习改进点的跟踪：把该改的地方记进一个文件，连续做到就关掉。

和 ``calibrate.py`` 分工不同，别混：

* ``calibrate.py`` 问的是**模板对不对** —— 容差该不该收紧、目标值有没有取偏。
  结论是改代码。
* 这个工具问的是**人做到了没有** —— 同一项检查，今天的实测有没有落进
  ``target ± tol``。结论是继续练。

测量本身只有一份实现，在 ``calibrate.measure()``，两边都用它。

判定一天的三种结果
------------------
``达标``
    这一项今天的实测中位数落在 ``target ± tol`` 里。连续 ``--streak`` 天达标
    就关闭这一条。
``未达标``
    落在区间外。**连续计数清零。**
``跳过``
    今天没有足够的判据。**连续计数保持不变，不清零也不推进。**

第三种是刻意的，它盖住三件事：

* **这个体式今天没练到。** 清零会逼着人每天把所有体式过一遍，那不是练习该有
  的样子。
* **读数是被遮挡的推测值。** 侧面机位下远侧半身每一帧都被挡住，MediaPipe 给的
  推测值看着完全合理（猜出来的膝角读 175~180「伸直」）。一条「双腿伸直」如果
  靠这种读数连着三天「达标」，它是被蒙混关掉的，不是被练好的。
* **同一天之内实测就摆动得比容差还大。** 实测过一例：侧角伸展式的「双臂成一线」
  在一天之内从 104° 摆到 179°（容差 20°）。中位数概括不了这种分布，拿它判
  今天达标没达标等于抛硬币 —— 而抛三次正面就会关掉一条改进点。

文件是给人看也给人改的
----------------------
``改进点.md`` 是普通 markdown：手动加一条、删一条、或者把某条挪到「已关闭」
都行。解析器对格式是宽容的（认得的字段读走，认不得的原样留着）。

用法::

    python tools/progress.py out/20260830/scores.json            # 更新（不存在就先建）
    python tools/progress.py out/*/scores.json                   # 一次补多天
    python tools/progress.py out/20260830/scores.json --date 2026-08-30
    python tools/progress.py --show                              # 只看，不动文件
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibrate import (  # noqa: E402
    ALWAYS_FAIL_RATIO,
    OFFSET_RATIO,
    Frame,
    load,
    measure,
    population,
)

from yoga_grid.poses import MIN_TRUSTED_VISIBILITY, TEMPLATES_BY_KEY  # noqa: E402

DEFAULT_FILE = Path("改进点.md")
DEFAULT_STREAK = 3
#: 一个体式当天至少要有这么多帧，才够给出一个判定。
MIN_FRAMES = 3
#: 当天实测散布（p90-p10）超过 tol 的这个倍数，就认为中位数不足以概括这一天。
UNSTABLE_SPREAD = 2.0
#: 判「达标」除了中位数落在区间里，还要求这个比例的帧真的拿到满分。
MET_FULL_RATE = 0.5

MET, MISSED, SKIPPED = "达标", "未达标", "跳过"
_MARK = {MET: "✓", MISSED: "✗", SKIPPED: "—"}


@dataclass
class Day:
    """某一天某一项的实测结果。"""

    date: str
    n: int
    median: float
    full_rate: float
    verdict: str
    note: str = ""

    @property
    def unreliable(self) -> bool:
        return self.verdict == SKIPPED and self.note.startswith("遮挡")


@dataclass
class Item:
    """一条改进点。"""

    pose_key: str
    pose_zh: str
    label: str
    target: float
    tol: float
    unit: str = "°"
    history: list[Day] = field(default_factory=list)
    closed_on: str = ""
    manual_note: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.pose_key, self.label)

    @property
    def streak(self) -> int:
        """连续达标天数。跳过的日子不打断，也不计入。"""
        n = 0
        for day in reversed(self.history):
            if day.verdict == MET:
                n += 1
            elif day.verdict == MISSED:
                break
        return n

    @property
    def latest(self) -> Day | None:
        return self.history[-1] if self.history else None

    def fmt(self, value: float) -> str:
        return f"{value:.1f}{self.unit}" if self.unit == "°" else f"{value:.2f}"


# --------------------------------------------------------------------------
# 测量
# --------------------------------------------------------------------------


def evaluate(frames: list[Frame], date: str) -> dict[tuple[str, str], Day]:
    """把一天的帧算成「每个体式每一项今天怎么样」。"""
    out: dict[tuple[str, str], Day] = {}
    for key, template in TEMPLATES_BY_KEY.items():
        sample = population(frames, template)
        if len(sample) < MIN_FRAMES:
            continue
        measured = measure(sample, template)
        for check in template.checks:
            rows = measured.get(check.label) or []
            if not rows:
                continue
            values = np.array([r[0] for r in rows], dtype=np.float64)
            scores = np.array([r[1] for r in rows], dtype=np.float64)
            conf = np.array([r[2] for r in rows], dtype=np.float64)

            median = float(np.median(values))
            full = float((scores >= 0.999).mean())
            verdict, note = decide(
                median=median,
                full_rate=full,
                occluded=float((conf < MIN_TRUSTED_VISIBILITY).mean()),
                spread=float(np.percentile(values, 90) - np.percentile(values, 10)),
                target=check.target,
                tol=check.tol,
            )
            out[(key, check.label)] = Day(date, len(rows), median, full, verdict, note)
    return out


def decide(
    *, median: float, full_rate: float, occluded: float, spread: float,
    target: float, tol: float,
) -> tuple[str, str]:
    """一天一项的判定。纯函数，四个入参就是全部依据。

    顺序有意义：两条「今天不算数」排在前面，因为它们说的是**这个读数没资格
    下判断**，而不是「做得好不好」。
    """
    if occluded >= 0.5:
        return SKIPPED, f"遮挡 {occluded:.0%}，读数不可信"
    if tol > 0 and spread > tol * UNSTABLE_SPREAD:
        # 同一天之内就摆动这么大，中位数概括不了它，按它判达标等于抛硬币。
        return SKIPPED, f"当天散布 {spread:.1f}，中位数说明不了问题"
    if abs(median - target) > tol:
        return MISSED, f"偏 {abs(median - target):.1f}"
    if full_rate < MET_FULL_RATE:
        # 中位数擦着区间边缘过去时会出现这种：一半的帧其实在界外。实测见过
        # 金字塔式「躯干向下折叠」中位 -0.30、目标 -0.65±0.35，正好压在边上，
        # 满分率只有 52%。压线不算做到。
        return MISSED, f"中位在区间内，但只有 {full_rate:.0%} 的帧达标"
    return MET, ""


def worth_tracking(day: Day, target: float, tol: float) -> bool:
    """值不值得开一条改进点。

    只收**当天确实没做到、而且读数可信**的项。饱和项（区间白给）是模板的
    问题，归 ``calibrate.py``，不该出现在人的练习清单上。
    """
    if day.verdict != MISSED:
        return False
    offset = abs(day.median - target)
    return offset > tol * OFFSET_RATIO or (1.0 - day.full_rate) >= ALWAYS_FAIL_RATIO


def seed(results: dict[tuple[str, str], Day]) -> list[Item]:
    items: list[Item] = []
    for (pose_key, label), day in results.items():
        template = TEMPLATES_BY_KEY[pose_key]
        check = next(c for c in template.checks if c.label == label)
        if not worth_tracking(day, check.target, check.tol):
            continue
        items.append(
            Item(pose_key, template.zh, label, check.target, check.tol, check.unit,
                 history=[day])
        )
    items.sort(key=lambda i: (i.pose_zh, i.label))
    return items


def apply(
    items: list[Item], results: dict[tuple[str, str], Day], streak: int
) -> tuple[list[Item], list[str]]:
    """把一天的结果并进现有条目，返回 (条目, 变化说明)。

    同一天重复跑只更新那一天，不追加 —— 否则改一次模板重跑一遍，连续天数
    就凭空涨了。
    """
    changes: list[str] = []
    known = {i.key for i in items}

    for item in items:
        if item.closed_on:
            continue
        day = results.get(item.key)
        if day is None:
            continue
        item.history = [d for d in item.history if d.date != day.date]
        item.history.append(day)
        item.history.sort(key=lambda d: d.date)
        if item.streak >= streak:
            item.closed_on = day.date
            changes.append(f"关闭：{item.pose_zh} · {item.label}（连续 {item.streak} 天达标）")
        elif day.verdict == MET:
            changes.append(
                f"达标：{item.pose_zh} · {item.label}（{item.streak}/{streak}）")
        elif day.verdict == MISSED:
            changes.append(
                f"未达标：{item.pose_zh} · {item.label}，实测 {item.fmt(day.median)}"
                f"（目标 {item.fmt(item.target)} ± {item.tol}），连续计数归零")

    for fresh in seed(results):
        if fresh.key in known:
            continue
        items.append(fresh)
        changes.append(f"新增：{fresh.pose_zh} · {fresh.label}")

    items.sort(key=lambda i: (bool(i.closed_on), i.pose_zh, i.label))
    return items, changes


# --------------------------------------------------------------------------
# 文件格式
# --------------------------------------------------------------------------

_HEAD = """# 练习改进点

<!-- 这个文件由 `python tools/progress.py <输出目录>/scores.json` 更新，
     也可以直接手改：加一条、删一条、把某条挪进「已关闭」都行。

     连续 {streak} 天达标自动关闭。没练到那个体式、或者读数被遮挡不可信的日子
     记「跳过」，连续计数保持不变 —— 不清零也不推进。

     「目标 ± 容差」来自 yoga_grid/poses.py，改那里才是改标准；这个文件只记
     你做到了没有。 -->
"""

_ITEM_RE = re.compile(r"^### +(?P<title>.+?)\s*$", re.M)
_FIELD_RE = re.compile(r"^- +(?P<k>[^：:]+)[：:] *(?P<v>.*?)\s*$", re.M)
_ROW_RE = re.compile(
    r"^\| *(?P<date>\d{4}-\d{2}-\d{2}) *\| *(?P<n>\d+) *\| *(?P<median>[-\d.]+)[^|]*\|"
    r" *(?P<full>[\d.]+)% *\| *(?P<verdict>[^|]*?) *\| *(?P<note>[^|]*?) *\|",
    re.M,
)


def parse(text: str) -> list[Item]:
    """读回条目。认不出的段落忽略，不报错 —— 这个文件是给人改的。"""
    items: list[Item] = []
    starts = [(m.start(), m.group("title")) for m in _ITEM_RE.finditer(text)]
    for n, (pos, title) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(text)
        block = text[pos:end]
        if "·" not in title:
            continue
        pose_zh, label = (p.strip() for p in title.split("·", 1))

        fields = {m.group("k").strip(): m.group("v").strip()
                  for m in _FIELD_RE.finditer(block)}
        pose_key = fields.get("模板", "")
        template = TEMPLATES_BY_KEY.get(pose_key)
        if template is None:
            continue
        check = next((c for c in template.checks if c.label == label), None)
        if check is None:
            continue

        history = [
            Day(m.group("date"), int(m.group("n")), float(m.group("median")),
                float(m.group("full")) / 100.0, m.group("verdict").strip("✓✗— ").strip(),
                m.group("note").strip())
            for m in _ROW_RE.finditer(block)
        ]
        items.append(Item(
            pose_key, template.zh, label, check.target, check.tol, check.unit,
            history=history,
            closed_on=fields.get("已关闭", ""),
            manual_note=fields.get("备注", ""),
        ))
    return items


def _render_item(item: Item, streak: int) -> str:
    lines = [f"### {item.pose_zh} · {item.label}", ""]
    lines.append(f"- 模板：{item.pose_key}")
    lines.append(f"- 目标：{item.fmt(item.target)} ± {item.tol}")
    if item.closed_on:
        lines.append(f"- 已关闭：{item.closed_on}")
    else:
        lines.append(f"- 连续达标：{item.streak} / {streak}")
    if item.manual_note:
        lines.append(f"- 备注：{item.manual_note}")
    lines.append("")
    if item.history:
        lines.append("| 日期 | 帧数 | 实测中位 | 满分率 | 判定 | 说明 |")
        lines.append("|------|------|---------|--------|------|------|")
        for d in item.history:
            lines.append(
                f"| {d.date} | {d.n} | {item.fmt(d.median)} | {d.full_rate:.0%} "
                f"| {_MARK.get(d.verdict, '')} {d.verdict} | {d.note} |"
            )
        lines.append("")
    return "\n".join(lines)


def render(items: list[Item], streak: int = DEFAULT_STREAK) -> str:
    live = [i for i in items if not i.closed_on]
    done = [i for i in items if i.closed_on]
    out = [_HEAD.format(streak=streak), ""]
    out.append(f"## 进行中（{len(live)}）")
    out.append("")
    out.extend(_render_item(i, streak) for i in live) if live else out.append("暂无。\n")
    out.append(f"## 已关闭（{len(done)}）")
    out.append("")
    if done:
        out.extend(_render_item(i, streak) for i in done)
    else:
        out.append("暂无。\n")
    return "\n".join(out).rstrip() + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def date_of(path: Path, frames: list[Frame]) -> str:
    """练习日期：优先从视频文件名里的 8 位数字取，取不到就用今天。

    用视频名而不是跑这个脚本的日子 —— 隔天才复盘是常事，记错日期会让
    「连续三天」变成假的。
    """
    for candidate in (frames[0].source if frames else "", str(path)):
        m = re.search(r"(20\d{2})(\d{2})(\d{2})", candidate)
        if m:
            return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return dt.date.today().isoformat()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="跟踪练习改进点，连续达标就关闭。")
    ap.add_argument("scores", nargs="*", type=Path, help="一个或多个 scores.json")
    ap.add_argument("-f", "--file", type=Path, default=DEFAULT_FILE,
                    help=f"改进点文件（默认 {DEFAULT_FILE}）")
    ap.add_argument("--date", help="练习日期 YYYY-MM-DD，默认从视频文件名推断")
    ap.add_argument("--streak", type=int, default=DEFAULT_STREAK, help="连续几天达标就关闭")
    ap.add_argument("--show", action="store_true", help="只打印当前状态，不写文件")
    args = ap.parse_args(argv)

    items = parse(args.file.read_text(encoding="utf-8")) if args.file.is_file() else []

    if args.show or not args.scores:
        live = [i for i in items if not i.closed_on]
        if not items:
            print(f"{args.file} 还没有内容。跑一次："
                  f"\n  python tools/progress.py out/<日期>/scores.json")
            return 0
        print(f"# {args.file} —— 进行中 {len(live)}，已关闭 {len(items) - len(live)}\n")
        for i in live:
            last = i.latest
            tail = f"，最近 {last.date} {i.fmt(last.median)} {last.verdict}" if last else ""
            print(f"  [{i.streak}/{args.streak}] {i.pose_zh} · {i.label}"
                  f"（目标 {i.fmt(i.target)} ± {i.tol}）{tail}")
        return 0

    for path in args.scores:
        frames = load(path)
        date = args.date or date_of(path, frames)
        items, changes = apply(items, evaluate(frames, date), args.streak)
        print(f"\n{date}（{path}，{len(frames)} 帧）")
        for line in changes or ["  没有变化。"]:
            print(f"  {line}")

    args.file.write_text(render(items, args.streak), encoding="utf-8")
    live = sum(1 for i in items if not i.closed_on)
    print(f"\n已写入 {args.file}：进行中 {live}，已关闭 {len(items) - live}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
