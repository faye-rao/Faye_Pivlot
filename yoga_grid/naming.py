"""输出文件名的日期 + 序号戳，让一天里多次运行不互相覆盖。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def today_stamp() -> str:
    """本地日期，形如 20260828。"""
    return datetime.now().strftime("%Y%m%d")


def next_sequence(out_dir: Path, base: str, suffix: str, stamp: str) -> int:
    """扫目录里同名同日期的文件，返回下一个序号（从 1 开始）。

    序号按**当天已有文件**推算而不是维护一个计数器文件：目录本身就是事实来源，
    手动删掉几张、或换个目录跑，序号都自然跟上，不会和某个游标文件失去同步。
    """
    if not out_dir.is_dir():
        return 1
    pattern = re.compile(
        rf"^{re.escape(base)}_{re.escape(stamp)}_(\d+){re.escape(suffix)}$"
    )
    used = [
        int(m.group(1))
        for entry in out_dir.iterdir()
        if (m := pattern.match(entry.name))
    ]
    return max(used) + 1 if used else 1


def stamped_path(out_dir: Path, base: str, suffix: str, stamp: str | None = None,
                 sequence: int | None = None) -> Path:
    """拼出 ``base_YYYYMMDD_NN.suffix`` 的完整路径。

    传入 ``sequence`` 可以让同一次运行的多个产物共用一个序号 —— 九宫格和它的
    对照图、复盘必须是同一个号，否则事后对不上是哪次跑的。
    """
    stamp = stamp or today_stamp()
    if sequence is None:
        sequence = next_sequence(out_dir, base, suffix, stamp)
    return out_dir / f"{base}_{stamp}_{sequence:02d}{suffix}"


def run_sequence(out_dir: Path, bases: list[tuple[str, str]], stamp: str | None = None) -> int:
    """给一次运行定一个序号：取所有产物里已用序号的最大值 + 1。

    统一取最大值，是为了让本次运行的所有产物共用同一个号。若各自算，
    上次没生成对照图（默认就不生成）时，两者的序号会从此错开。
    """
    stamp = stamp or today_stamp()
    return max(
        (next_sequence(out_dir, base, suffix, stamp) for base, suffix in bases),
        default=1,
    )


def clear_generated(directory: Path, pattern: str) -> int:
    """删掉目录里由本程序生成的旧文件，返回删除数量。

    只删**匹配自己命名规则**的文件，不清空整个目录 —— 用户可能往里放了别的东西。

    为什么必须清：``frames/`` 和 ``candidates/`` 的文件名带时间戳和体式名，
    换一次模板或改一次聚类参数，新文件就换了名字，旧文件留在原地不会被覆盖。
    于是目录里同时存在两代结果，看文件名根本分不出哪个是本次的 —— 这已经真实
    导致过误判：一个上一轮留下的 `02_unknown_0098.72s.jpg` 让人以为本轮没识别出
    那个体式，而本轮其实识别对了。
    """
    if not directory.is_dir():
        return 0
    removed = 0
    for entry in directory.glob(pattern):
        if entry.is_file():
            entry.unlink()
            removed += 1
    return removed
