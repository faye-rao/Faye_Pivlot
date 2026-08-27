"""输出文件名的日期 + 序号戳。

    python tests/test_naming.py      或      python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid import naming  # noqa: E402

STAMP = "20260828"


def _tmp() -> Path:
    return Path(tempfile.mkdtemp(prefix="yoga_grid_naming_"))


def test_first_sequence_is_one():
    root = _tmp()
    assert naming.next_sequence(root, "九宫格", ".jpg", STAMP) == 1


def test_missing_directory_is_not_an_error():
    """目录还没建就先算序号是正常调用顺序，不该抛异常。"""
    root = _tmp() / "还没建"
    assert naming.next_sequence(root, "九宫格", ".jpg", STAMP) == 1


def test_sequence_counts_existing_files():
    root = _tmp()
    for n in (1, 2):
        (root / f"九宫格_{STAMP}_{n:02d}.jpg").touch()
    assert naming.next_sequence(root, "九宫格", ".jpg", STAMP) == 3


def test_sequence_takes_max_not_count():
    """有空洞时取最大值 + 1，不填空洞。

    填空洞会让新文件占用某个已删文件的号，事后按文件名排序就不再等于时间顺序。
    """
    root = _tmp()
    for n in (1, 5):
        (root / f"九宫格_{STAMP}_{n:02d}.jpg").touch()
    assert naming.next_sequence(root, "九宫格", ".jpg", STAMP) == 6


def test_other_dates_and_names_are_ignored():
    root = _tmp()
    (root / f"九宫格_20260101_07.jpg").touch()      # 别的日期
    (root / f"标准对照图_{STAMP}_09.jpg").touch()     # 别的前缀
    (root / f"九宫格_{STAMP}_03.png").touch()        # 别的扩展名
    (root / "九宫格.jpg").touch()                    # 无戳的旧文件
    (root / f"九宫格_{STAMP}_x.jpg").touch()         # 序号不是数字
    assert naming.next_sequence(root, "九宫格", ".jpg", STAMP) == 1


def test_sequence_beyond_two_digits():
    """超过 99 也要能继续，位数自然增长而不是回绕。"""
    root = _tmp()
    (root / f"九宫格_{STAMP}_99.jpg").touch()
    assert naming.next_sequence(root, "九宫格", ".jpg", STAMP) == 100
    path = naming.stamped_path(root, "九宫格", ".jpg", STAMP)
    assert path.name == f"九宫格_{STAMP}_100.jpg"


def test_stamped_path_format():
    root = _tmp()
    path = naming.stamped_path(root, "九宫格", ".jpg", STAMP, sequence=7)
    assert path == root / f"九宫格_{STAMP}_07.jpg"


def test_run_sequence_shares_one_number_across_outputs():
    """一次运行的多个产物必须同号，即使上次只生成了其中一部分。

    默认不出对照图，所以对照图的已用序号会落后于九宫格；各自算的话两者会
    从此错开，事后对不上是哪次跑的。
    """
    root = _tmp()
    bases = [("九宫格", ".jpg"), ("标准对照图", ".jpg"), ("report", ".md")]
    for n in (1, 2, 3):
        (root / f"九宫格_{STAMP}_{n:02d}.jpg").touch()
        (root / f"report_{STAMP}_{n:02d}.md").touch()
    (root / f"标准对照图_{STAMP}_01.jpg").touch()   # 只有第一次出过对照图

    seq = naming.run_sequence(root, bases, STAMP)
    assert seq == 4, f"应取所有产物的最大已用序号 + 1，实际 {seq}"
    for base, suffix in bases:
        assert naming.stamped_path(root, base, suffix, STAMP, sequence=seq).name.endswith(
            f"_{STAMP}_04{suffix}"
        )


def test_today_stamp_shape():
    stamp = naming.today_stamp()
    assert len(stamp) == 8 and stamp.isdigit()


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
