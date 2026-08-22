#!/usr/bin/env python3
"""Regenerate the parts of README.md that are derived from the code.

The pose table, the per-pose check tables and the module list all describe
things that live in the source.  Written by hand they go stale the first time
someone widens a target band and forgets the docs; generated from
``yoga_coach.POSES`` and the package directory they cannot.

Usage::

    python tools/update_readme.py            # rewrite the generated blocks
    python tools/update_readme.py --check    # exit 1 if README is out of date

``tests/test_readme.py`` runs the ``--check`` mode, so a code change that
should have updated the documentation fails the test suite instead of
shipping silently.

Generated regions are delimited in the Markdown by::

    <!-- BEGIN GENERATED: name -->
    ...replaced wholesale...
    <!-- END GENERATED: name -->

Everything outside those markers is hand-written prose and is never touched.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from yoga_coach.poses import POSES  # noqa: E402

README = ROOT / "README.md"
PACKAGE = ROOT / "yoga_coach"
TESTS = ROOT / "tests"

#: One line of Chinese for every module in the package, plus whether it is
#: allowed to touch the outside world.  Adding a module without adding it here
#: is an error: the architecture section would silently omit it.
MODULE_NOTES: dict[str, tuple[str, str]] = {
    "geometry.py": ("角度、距离、EMA 平滑等纯几何计算", "无依赖"),
    "landmarks.py": ("33 个关键点命名、单帧骨架、可见度门控、mid_* 虚拟点", "无依赖"),
    "metrics.py": ("能测什么：关节角、与竖直/水平夹角、左右倾斜、间距、点到直线偏移", "无依赖"),
    "checks.py": ("一条规则的定义与打分：测量 + 目标区间 + 两句建议", "无依赖"),
    "poses.py": ("体式库，每个体式是一组 Check", "无依赖"),
    "evaluator.py": ("单帧打分、左右侧判定、体式识别排序、置信度", "无依赖"),
    "session.py": ("跨帧状态：关键点平滑、体式跟踪迟滞、保持计时、建议节流", "无依赖"),
    "detector.py": ("MediaPipe 封装、模型下载与缓存", "MediaPipe"),
    "render.py": ("画面叠加：骨架、评分面板、中文文字", "OpenCV + Pillow"),
    "console.py": ("终端输出：实时单行播报、单张照片完整报告", "无依赖"),
    "voice.py": ("可选语音播报，缺少 pyttsx3 时静默降级", "pyttsx3（可选）"),
    "cli.py": ("命令行入口、采集循环、四种运行模式", "OpenCV"),
    "__init__.py": ("对外导出", "无依赖"),
    "__main__.py": ("`python -m yoga_coach` 入口", "无依赖"),
}


def _module_rows() -> list[str]:
    found = sorted(p.name for p in PACKAGE.glob("*.py"))
    missing = [name for name in found if name not in MODULE_NOTES]
    if missing:
        raise SystemExit(
            f"新模块没有中文说明：{', '.join(missing)}\n"
            f"请在 {Path(__file__).relative_to(ROOT)} 的 MODULE_NOTES 里补上，再重新生成。"
        )
    stale = [name for name in MODULE_NOTES if name not in found]
    if stale:
        raise SystemExit(
            f"MODULE_NOTES 里的模块已不存在：{', '.join(stale)}\n"
            f"请从 {Path(__file__).relative_to(ROOT)} 里删掉，再重新生成。"
        )
    # Interesting files first, dunder plumbing last.
    ordered = [n for n in found if not n.startswith("__")] + [
        n for n in found if n.startswith("__")
    ]
    rows = ["| 模块 | 职责 | 外部依赖 |", "| --- | --- | --- |"]
    for name in ordered:
        note, deps = MODULE_NOTES[name]
        rows.append(f"| `{name}` | {note} | {deps} |")
    return rows


def _pose_rows() -> list[str]:
    rows = [
        "| key | 体式 | Sanskrit | 摄像头位置 | 检查项 | 左右 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for pose in POSES:
        sides = "对称" if pose.symmetric else "自动判别"
        rows.append(
            f"| `{pose.key}` | {pose.name.zh} | {pose.sanskrit} | "
            f"{pose.view.zh} | {len(pose.checks)} | {sides} |"
        )
    return rows


def _check_sections() -> list[str]:
    lines: list[str] = []
    for pose in POSES:
        lines.append(f"#### {pose.name.zh} `{pose.key}`")
        lines.append("")
        lines.append(f"> {pose.view.zh}。{pose.cue.zh}。")
        lines.append("")
        # Deliberately phrased as "below / above the band" rather than
        # "not enough / too much": for a knee angle, *below* the band means
        # bent more deeply, which is the opposite of "not enough".
        lines.append("| 检查项 | 目标区间 | 低于区间时提示 | 高于区间时提示 | 权重 |")
        lines.append("| --- | --- | --- | --- | --- |")
        for check in pose.checks:
            low = check.when_low.zh if check.when_low else "—"
            high = check.when_high.zh if check.when_high else "—"
            lines.append(
                f"| {check.label.zh} | {check.target_text()} | {low} | {high} "
                f"| {check.weight:g} |"
            )
        lines.append("")
    return lines[:-1]  # drop the trailing blank line


def _count_test_functions() -> int:
    total = 0
    for path in sorted(TESTS.glob("test_*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name.startswith("test_"):
                    total += 1
    return total


def _stats_lines() -> list[str]:
    checks = sum(len(pose.checks) for pose in POSES)
    return [
        f"- 体式 **{len(POSES)}** 个，检查规则 **{checks}** 条",
        f"- 测试函数 **{_count_test_functions()}** 个"
        "（部分带参数化，实际用例数更多），不需要摄像头和 MediaPipe",
    ]


BLOCKS = {
    "poses": _pose_rows,
    "checks": _check_sections,
    "modules": _module_rows,
    "stats": _stats_lines,
}


def render(text: str) -> str:
    """Replace every generated block in ``text`` with freshly built content."""
    for name, build in BLOCKS.items():
        begin = f"<!-- BEGIN GENERATED: {name} -->"
        end = f"<!-- END GENERATED: {name} -->"
        pattern = re.compile(
            re.escape(begin) + r".*?" + re.escape(end),
            re.DOTALL,
        )
        if not pattern.search(text):
            raise SystemExit(f"README.md 里找不到 {name} 的生成区块标记（{begin}）。")
        body = "\n".join(build())
        text = pattern.sub(f"{begin}\n{body}\n{end}", text, count=1)
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="只检查不写入；README 过期时打印 diff 并以 1 退出",
    )
    args = parser.parse_args(argv)

    current = README.read_text(encoding="utf-8")
    updated = render(current)

    if current == updated:
        if not args.check:
            print("README.md 已是最新。")
        return 0

    if args.check:
        diff = difflib.unified_diff(
            current.splitlines(keepends=True),
            updated.splitlines(keepends=True),
            fromfile="README.md (当前)",
            tofile="README.md (应有)",
        )
        sys.stdout.writelines(diff)
        print("\nREADME.md 已过期，请运行：python tools/update_readme.py", file=sys.stderr)
        return 1

    README.write_text(updated, encoding="utf-8")
    print("README.md 已更新。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
