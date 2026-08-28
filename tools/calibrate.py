"""把若干 scores.json 里的实测值和模板的 target ± tol 并排列出来。

为什么需要这个工具
------------------
``report.md`` 只写入选那九张里**偏离较多**的检查项。九张 × 一两项，样本太少，
而且只看得到失分的那一头 —— 恰好看不到「全部满分」这种同样需要改的情况。
容差是统计量，要按分布调：

* **饱和**（几乎全部满分，且实测散布远窄于 tol）—— 区间比数据需要的宽，
  这一项其实没在打分，白占权重。
* **系统性偏移**（中位数离 target 超过半个 tol）—— 区间取在了错误的位置，
  真实练习稳定地落在别处。教科书插画取的目标值最常犯这个（见
  ``reference_images/README.md``）。
* **常否**（大量 0 分）—— 要么这一项写错了，要么样本被别的体式污染了。
  后者更常见，也更危险：它意味着模板在认领不属于它的帧。

只读关键点，不碰 cv2 / mediapipe
--------------------------------
``scores.json`` 里存着每一帧的原始关键点，几何层重算是毫秒级的。所以这个
工具和测试一样跑在纯 numpy 上，不需要视频、不需要装完整依赖。

**逐帧重新识别，不信文件里的标签。** ``scores.json`` 的 ``pose`` 字段是
*簇的结论*：``select._assign_alignment`` 会把簇内主导体式强加给簇里每一帧。
拿它当样本，等于用别的体式的帧去校准这个体式的容差 —— 实测某支 38 分钟的
视频，标着「直臂斜板」的 159 帧里有 28% 逐帧识别是四柱式、上犬式、婴儿式。
所以这里一律重跑 ``match_pose``。

用法::

    python tools/calibrate.py out/v1/scores.json out/v2/scores.json
    python tools/calibrate.py out/*/scores.json --key pigeon    # 只看一个体式
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid.landmarks import PoseView, normalize, to_pixels  # noqa: E402
from yoga_grid.poses import (  # noqa: E402
    TEMPLATES,
    TEMPLATES_BY_KEY,
    Template,
    _score_template,
    _spine_up,
    match_pose,
)

#: 判「饱和」：满分比例的下限，以及实测散布相对 tol 的上限。
SATURATED_RATIO = 0.95
SATURATED_SPREAD = 0.6
#: 判「系统性偏移」：中位数偏离 target 超过 tol 的这个倍数。
OFFSET_RATIO = 0.5
#: 判「常否」：0 分比例的下限。
ALWAYS_FAIL_RATIO = 0.25


@dataclass
class Frame:
    """一帧的最小信息：时间、来源、归一化骨架。"""

    source: str
    t: float
    norm: np.ndarray


def load(path: Path) -> list[Frame]:
    """读 scores.json，重建归一化骨架。

    ``landmarks`` 是 33×3 的原始关键点（归一化 x、y 加置信度）。算 norm 之前
    必须先乘回画面宽高 —— 直接拿归一化坐标算角度会被画面长宽比拉歪。
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    width = int(payload["video"].get("width") or 0)
    height = int(payload["video"].get("height") or 0)
    if not width or not height:
        raise SystemExit(f"{path}：video 段里没有画面尺寸，无法还原骨架。")

    # 按视频文件名分组，而不是按 scores.json 所在目录 —— 「样本只来自一支视频」
    # 这个警告是这个工具最要紧的一句话，不能被目录布局搞错。
    tag = Path(payload["video"].get("path") or path.stem).stem
    frames: list[Frame] = []
    for entry in payload["candidates"]:
        raw = entry.get("landmarks")
        if not raw:
            continue
        lm = np.asarray(raw, dtype=np.float64)
        if lm.shape != (33, 3):
            # 33×2 的早期格式已经是 norm 了，不能再乘一次宽高。
            frames.append(Frame(tag, float(entry["t"]), lm[:, :2]))
            continue
        frames.append(Frame(tag, float(entry["t"]), normalize(to_pixels(lm, width, height))))

    if not frames:
        raise SystemExit(
            f"{path}：没有可用关键点 —— 当初可能用了 --no-landmarks。"
            "重跑一次完整流水线才能校准。"
        )
    return frames


def population(frames: list[Frame], template: Template) -> list[Frame]:
    """逐帧重新识别，挑出**本模板确实赢下**的帧。

    只要它赢，不要求它赢得多漂亮 —— 擦边帧正是容差该覆盖的那一头，
    把它们剔掉会让分布假性收窄，越校越紧。
    """
    keep = []
    for frame in frames:
        match = match_pose(frame.norm)
        if match is not None and match.key == template.key:
            keep.append(frame)
    return keep


def _measure(frames: list[Frame], template: Template) -> dict[str, list[tuple[float, float]]]:
    """每项检查在样本上的 (实测值, 得分) 列表。侧别取模板自己选的那一侧。"""
    out: dict[str, list[tuple[float, float]]] = {c.label: [] for c in template.checks}
    for frame in frames:
        match = _score_template(frame.norm, template)
        if match is None:
            continue
        view = PoseView(frame.norm, match.side)
        for check in template.checks:
            score, value = check.score(view)
            if not np.isnan(value):
                out[check.label].append((value, score))
    return out


def _verdict(values: np.ndarray, scores: np.ndarray, target: float, tol: float) -> str:
    """给一项检查下判断。空字符串表示没发现问题。"""
    full = float((scores >= 0.999).mean())
    zero = float((scores <= 0.001).mean())
    spread = float(np.percentile(values, 90) - np.percentile(values, 10))
    offset = abs(float(np.median(values)) - target)

    if zero >= ALWAYS_FAIL_RATIO:
        return f"常否（{zero:.0%} 得 0 分）"
    if offset > tol * OFFSET_RATIO:
        return f"系统性偏移（中位数偏 {offset:.2f}，tol {tol}）"
    if full >= SATURATED_RATIO and tol > 0 and spread < tol * SATURATED_SPREAD:
        return f"饱和（{full:.0%} 满分，实测散布 {spread:.2f} < tol {tol}）"
    return ""


def report_template(frames: list[Frame], template: Template, min_n: int) -> None:
    sample = population(frames, template)
    if len(sample) < min_n:
        print(f"\n## {template.zh}（{template.key}）—— 样本 {len(sample)} 帧，不足 {min_n}，跳过")
        return

    sources: dict[str, int] = {}
    for frame in sample:
        sources[frame.source] = sources.get(frame.source, 0) + 1
    by_source = "、".join(f"{k} {v}" for k, v in sorted(sources.items()))
    print(f"\n## {template.zh}（{template.key}）—— 样本 {len(sample)} 帧（{by_source}）")

    if len(sources) == 1:
        print("   注意：样本只来自一支视频，容易把某一天的状态当成标准。")

    spine = np.array([_spine_up(PoseView(f.norm)) for f in sample])
    print(
        f"   朝向门槛 spine_up {template.spine_up}："
        f"实测 {np.percentile(spine, 10):+.2f} ~ {np.percentile(spine, 90):+.2f}"
        f"（中位 {np.median(spine):+.2f}）"
    )

    print(f"   {'检查项':<22}{'目标':>14}  {'实测 p10~p90（中位）':>26}  {'满分':>5}  判断")
    measured = _measure(sample, template)
    for check in template.checks:
        rows = measured[check.label]
        if not rows:
            print(f"   {check.label:<22}{'—— 全部无法测量':>14}")
            continue
        values = np.array([v for v, _ in rows])
        scores = np.array([s for _, s in rows])
        target = f"{check.target}±{check.tol}{check.unit}"
        band = (
            f"{np.percentile(values, 10):.2f} ~ {np.percentile(values, 90):.2f}"
            f"（{np.median(values):.2f}）"
        )
        full = f"{(scores >= 0.999).mean():.0%}"
        print(
            f"   {check.label:<22}{target:>14}  {band:>26}  {full:>5}  "
            f"{_verdict(values, scores, check.target, check.tol)}"
        )


def report_contamination(frames: list[Frame], min_n: int) -> None:
    """每个模板认领的帧里，第二名是谁、差多少。

    差距小说明两个模板在真实数据上贴得很近，改任何一边的容差都会把帧推来推去
    —— 这种情况下该加判别性检查或门槛，而不是调容差。
    """
    print("\n\n# 模板之间的余量")
    print("（第二名和第一名的差距。差距 < 0.05 的，调容差会让归属反复摇摆）\n")
    print(f"  {'体式':<10}{'n':>4}  {'第一名中位':>10}  {'第二名':<10}{'第二名中位':>10}  {'余量':>7}")
    rows: list[tuple[float, str]] = []
    for template in TEMPLATES:
        margins: list[float] = []
        firsts: list[float] = []
        seconds: dict[str, list[float]] = {}
        for frame in frames:
            match = match_pose(frame.norm)
            if match is None or match.key != template.key:
                continue
            scored = sorted(
                (
                    (m.score, t.key)
                    for t, m in (
                        (t, _score_template(frame.norm, t)) for t in TEMPLATES
                    )
                    if m is not None and t.key != template.key
                ),
                reverse=True,
            )
            if not scored:
                continue
            firsts.append(match.score)
            margins.append(match.score - scored[0][0])
            seconds.setdefault(scored[0][1], []).append(scored[0][0])
        if len(margins) < min_n:
            continue
        runner = max(seconds, key=lambda k: len(seconds[k]))
        rows.append(
            (
                float(np.median(margins)),
                f"  {template.zh:<10}{len(margins):>4}  {np.median(firsts):>10.2f}  "
                f"{TEMPLATES_BY_KEY[runner].zh:<10}{np.median(seconds[runner]):>10.2f}  "
                f"{np.median(margins):>7.2f}",
            )
        )
    for _, line in sorted(rows):
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="按实测分布检查 yoga_grid 模板的容差。"
    )
    parser.add_argument("scores", nargs="+", type=Path, help="一个或多个 scores.json")
    parser.add_argument("--key", action="append", default=None, help="只看这些体式 key")
    parser.add_argument(
        "--min-n", type=int, default=5, help="样本少于这么多帧就跳过（默认 5）"
    )
    args = parser.parse_args(argv)

    frames: list[Frame] = []
    for path in args.scores:
        frames.extend(load(path))

    print(f"# 容差校准 —— {len(args.scores)} 支视频，{len(frames)} 帧候选")
    print("\n实测值一律按当前代码逐帧重新识别，不采用文件里的簇标签。")

    keys = args.key or [t.key for t in TEMPLATES]
    unknown = [k for k in keys if k not in TEMPLATES_BY_KEY]
    if unknown:
        raise SystemExit(
            f"没有这些体式：{', '.join(unknown)}\n"
            f"可用的是：{', '.join(t.key for t in TEMPLATES)}"
        )

    for key in keys:
        report_template(frames, TEMPLATES_BY_KEY[key], args.min_n)

    if not args.key:
        report_contamination(frames, args.min_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
