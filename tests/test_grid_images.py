"""``yoga_grid.images`` 的文件名 → 模板映射。

这一层的价值全在**别认错**：认错的后果不是少一条信息，而是把一张别的体式的
骨架当成某个模板的真值去校准容差，把模板改坏。所以下面的用例几乎都是陷阱。

    python -m pytest tests/test_grid_images.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid.images import ALIASES, label_from_path  # noqa: E402
from yoga_grid.poses import TEMPLATES  # noqa: E402


def _label(name: str, parent: str = "素材") -> tuple[str | None, bool]:
    root = Path("/tmp/lib")
    return label_from_path(root / parent / name, root)


def test_aliases_point_at_real_templates():
    """别名表里的 key 必须都存在 —— 改模板 key 时这条会先炸。"""
    keys = {t.key for t in TEMPLATES}
    bad = sorted({v for v in ALIASES.values() if v is not None} - keys)
    assert not bad, f"别名指向不存在的模板 key：{bad}"


def test_library_naming_convention():
    """`reference_images/` 用的是 `梵文 - English.png`。"""
    assert _label("Virabhadrasana II - Warrior II.png") == ("warrior2", True)
    assert _label("Adho Mukha Svanasana - Downward Facing Dog.png") == ("downdog", True)
    assert _label("Purvottanasana - Reverse Plank.png.jpg") == ("reverse_plank", True)


def test_chinese_names_work_too():
    assert _label("下犬式_01.jpg") == ("downdog", True)
    assert _label("战士二 侧面.png") == ("warrior2", True)


def test_folder_name_counts_as_a_label():
    """按体式分子目录是另一种常见组织方式。"""
    assert _label("IMG_2043.jpg", parent="三角伸展") == ("triangle", True)


def test_longest_alias_wins_over_substring():
    """`Ardha Uttanasana` 是半前屈，不是站立前屈 —— 库里没有对应模板。

    短别名优先的话它会被 "uttanasana" 认走，于是半前屈的骨架被当成站立前屈的
    真值，正好是最难发现的那种错。
    """
    assert _label("Ardha Uttanasana - Half Forward Fold.png") == (None, True)
    assert _label("Uttanasana - Forward Fold.png") == ("uttanasana", True)


def test_the_two_traps_the_library_readme_records():
    """差一个词就是另一个体式，`reference_images/README.md` 专门记了这两对。"""
    # 全劈叉，不是半神猴式
    assert _label("Hanumanasana - Monkey or Splits.png") == (None, True)
    assert _label("Ardha Hanumanasana - Half Splits.png.jpg") == (
        "ardha_hanumanasana", True
    )
    # 反战士，不是反板式
    assert _label("Viparita Virabhadrasana - Reverse Warrior.png") == (None, True)
    assert _label("Purvottanasana - Reverse Plank.png.jpg") == ("reverse_plank", True)


def test_warrior_numerals_do_not_collide():
    """罗马数字是前缀关系：I 是 II 和 III 的前缀，按短的匹配全会认成战士一。"""
    assert _label("Virabhadrasana I - Warrior I.png") == ("warrior1", True)
    assert _label("Virabhadrasana II - Warrior II.png") == ("warrior2", True)
    assert _label("Virabhadrasana III - Warrior III.png") == ("warrior3", True)


def test_pyramid_and_side_angle_are_different_poses():
    """Parsvottanasana / Parsvakonasana 只差中间一段。"""
    assert _label("Parsvottanasana - Intense Side Stretch.png") == (
        "parsvottanasana", True
    )
    assert _label("Parsvakonasana - Side Angle.png") == ("parsvakonasana", True)


def test_unrecognised_name_is_distinguishable_from_unmodelled_pose():
    """两种「没有 key」必须分得开。

    `(None, True)` = 认得这个体式，但故意没做模板 —— 素材有用，用来决定要不要加模板。
    `(None, False)` = 名字没对上任何东西 —— 得人去看这是什么。
    """
    assert _label("IMG_0031.jpg", parent="随手拍") == (None, False)
    assert _label("Bakasana - Crow.png") == (None, False)   # 库里有、模板没有，也没进别名表
    assert _label("Half Moon.png") == (None, True)          # 进了别名表，明确标为无模板


def test_alias_must_match_whole_words():
    """别名是按词匹配的，不能被更长的词裹进去误命中。"""
    assert _label("treetop-yoga-retreat.jpg") == (None, False)
    assert _label("Vrksasana - Tree.png") == ("tree", True)


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
