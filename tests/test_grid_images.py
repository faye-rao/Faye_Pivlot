"""``yoga_grid.images`` 的文件名 → 模板映射。

这一层的价值全在**别认错**：认错的后果不是少一条信息，而是把一张别的体式的
骨架当成某个模板的真值去校准容差，把模板改坏。所以下面的用例几乎都是陷阱。

    python -m pytest tests/test_grid_images.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid.images import (  # noqa: E402
    ALIASES, MATCHED, QUALIFIED, QUALIFIERS, UNKNOWN, UNMODELLED, label_from_path,
)
from yoga_grid.poses import TEMPLATES  # noqa: E402


def _label(name: str, parent: str = "素材") -> tuple[str | None, str]:
    root = Path("/tmp/lib")
    label = label_from_path(root / parent / name, root)
    return label.key, label.status


def test_aliases_point_at_real_templates():
    """别名表里的 key 必须都存在 —— 改模板 key 时这条会先炸。"""
    keys = {t.key for t in TEMPLATES}
    bad = sorted({v for v in ALIASES.values() if v is not None} - keys)
    assert not bad, f"别名指向不存在的模板 key：{bad}"


def test_every_template_is_reachable_by_some_alias():
    """**加了模板就必须加别名。**

    别名表是手写的，而模板集在长。漏掉一个的后果不是「少认出一张图」，是
    那个体式的图会被报成「认不出名字」或者被邻近体式的别名认走，而报告看上去
    一切正常 —— 恰恰是这个模块存在的意义所在的那类错。
    """
    covered = {v for v in ALIASES.values() if v is not None}
    missing = sorted(t.key for t in TEMPLATES if t.key not in covered)
    assert not missing, (
        f"这些模板在别名表里没有入口：{missing}。"
        "在 yoga_grid/images.py 的 ALIASES 里给每个补上梵文名、英文名和中文名。"
    )


def _fold_plain(text: str) -> str:
    return " ".join(text.replace("_", " ").replace("-", " ").lower().split())


def test_no_alias_denies_a_pose_that_now_has_a_template():
    """标成「故意没有模板」的别名，不能是某个真存在的模板。

    这条防的是**表和模板集脱节**：`ardha uttanasana`（展背式）原本确实没有模板，
    所以被标成 None 以免被更短的 `uttanasana` 认走。等哪天补上了这个模板，
    那条 None 就从「保护」变成「否认」—— 图明明有模板可对，却被报成库里没有。
    脱节是安静的，所以这里用模板的 key / 中文名 / 英文名三路去撞它。
    """
    names: set[str] = set()
    for t in TEMPLATES:
        names.update({_fold_plain(t.key), _fold_plain(t.zh), _fold_plain(t.en)})

    denied = sorted(
        alias for alias, key in ALIASES.items()
        if key is None and _fold_plain(alias) in names
    )
    assert not denied, (
        f"这些别名标着「没有模板」，但同名的模板已经存在：{denied}。"
        "把它们改成指向对应的模板 key。"
    )


def test_library_naming_convention():
    """`reference_images/` 用的是 `梵文 - English.png`。"""
    assert _label("Virabhadrasana II - Warrior II.png") == ("warrior2", MATCHED)
    assert _label("Adho Mukha Svanasana - Downward Facing Dog.png") == ("downdog", MATCHED)
    assert _label("Purvottanasana - Reverse Plank.png.jpg") == ("reverse_plank", MATCHED)


def test_chinese_names_work_too():
    assert _label("下犬式_01.jpg") == ("downdog", MATCHED)
    assert _label("战士二 侧面.png") == ("warrior2", MATCHED)


def test_folder_name_counts_as_a_label():
    """按体式分子目录是另一种常见组织方式。"""
    assert _label("IMG_2043.jpg", parent="三角伸展") == ("triangle", MATCHED)


def test_longest_alias_wins_over_substring():
    """`Ardha Uttanasana` 是半前屈，不是站立前屈 —— 库里没有对应模板。

    短别名优先的话它会被 "uttanasana" 认走，于是半前屈的骨架被当成站立前屈的
    真值，正好是最难发现的那种错。
    """
    assert _label("Ardha Uttanasana - Half Forward Fold.png") == (None, UNMODELLED)
    assert _label("Uttanasana - Forward Fold.png") == ("uttanasana", MATCHED)


def test_the_two_traps_the_library_readme_records():
    """差一个词就是另一个体式，`reference_images/README.md` 专门记了这两对。"""
    # 全劈叉，不是半神猴式
    assert _label("Hanumanasana - Monkey or Splits.png") == (None, UNMODELLED)
    assert _label("Ardha Hanumanasana - Half Splits.png.jpg") == (
        "ardha_hanumanasana", MATCHED
    )
    # 反战士，不是反板式
    assert _label("Viparita Virabhadrasana - Reverse Warrior.png") == (None, UNMODELLED)
    assert _label("Purvottanasana - Reverse Plank.png.jpg") == ("reverse_plank", MATCHED)


def test_warrior_numerals_do_not_collide():
    """罗马数字是前缀关系：I 是 II 和 III 的前缀，按短的匹配全会认成战士一。"""
    assert _label("Virabhadrasana I - Warrior I.png") == ("warrior1", MATCHED)
    assert _label("Virabhadrasana II - Warrior II.png") == ("warrior2", MATCHED)
    assert _label("Virabhadrasana III - Warrior III.png") == ("warrior3", MATCHED)


def test_pyramid_and_side_angle_are_different_poses():
    """Parsvottanasana / Parsvakonasana 只差中间一段。"""
    assert _label("Parsvottanasana - Intense Side Stretch.png") == (
        "parsvottanasana", MATCHED
    )
    assert _label("Parsvakonasana - Side Angle.png") == ("parsvakonasana", MATCHED)


def test_unrecognised_name_is_distinguishable_from_unmodelled_pose():
    """两种「没有 key」必须分得开。

    `(None, UNMODELLED)` = 认得这个体式，但故意没做模板 —— 素材有用，用来决定要不要加模板。
    `(None, UNKNOWN)` = 名字没对上任何东西 —— 得人去看这是什么。
    """
    assert _label("IMG_0031.jpg", parent="随手拍") == (None, UNKNOWN)
    assert _label("Bakasana - Crow.png") == (None, UNKNOWN)   # 库里有、模板没有，也没进别名表
    assert _label("Half Moon.png") == (None, UNMODELLED)          # 进了别名表，明确标为无模板


def test_qualifiers_that_change_the_pose_are_not_swallowed():
    """**这四条是用户第一批素材上真错过的**，不是假想的。

    梵文体式名靠限定词区分体式，不是修饰同一个体式。别名表只挡得住列过的
    组合，没列过的会被安静忽略：

        Adho Mukha Vrksasana            手倒立   被标成了树式
        Utthita Chaturanga Dandasana    直臂斜板 被标成了四柱支撑式
        Upavistha Parivritta ...        坐姿扭转 被标成了侧角伸展式
        Parivrtta Ashta Chandrasana     扭转新月 被标成了新月式

    第二条尤其说明问题：模板判它是直臂斜板 0.91、四柱支撑式只有 0.64 ——
    **模板是对的，标注是错的**。要是照这条标注去「校准」，会拿一张直臂斜板
    把四柱支撑式的容差撑开。

    穷举所有组合补不完，所以反过来：命中的别名没吃掉限定词就不下判断。
    """
    assert _label("Adho Mukha Vrksasana.jpg") == (None, QUALIFIED)
    assert _label("Utthita Chaturanga Dandasana.jpg") == (None, QUALIFIED)
    assert _label("Upavistha Parivritta Parsvakonasana.jpg") == (None, QUALIFIED)
    assert _label("Parivrtta Ashta Chandrasana - Revolved Crescent Moon.png") == (
        None, QUALIFIED
    )


def test_qualifiers_the_alias_itself_consumes_are_fine():
    """限定词是别名的一部分时不该触发 —— 否则半数体式都认不出来。"""
    assert _label("Adho Mukha Svanasana.jpg") == ("downdog", MATCHED)
    assert _label("Urdhva Mukha Svanasana.jpg") == ("updog", MATCHED)
    assert _label("Utthita Parsvakonasana.jpg") == ("parsvakonasana", MATCHED)
    assert _label("Ardha Hanumanasana.jpg") == ("ardha_hanumanasana", MATCHED)
    assert _label("Eka Pada Rajakapotasana.jpg") == ("pigeon", MATCHED)
    assert _label("Viparita Virabhadrasana.jpg") == (None, UNMODELLED)


def test_qualifier_list_does_not_shadow_any_alias_root():
    """限定词不能等于某个别名本身，否则那个别名永远认不出来。

    比如把 "side" 收进限定词，`side plank` 会被自己的限定词否掉。
    """
    clashes = sorted(q for q in QUALIFIERS if q in ALIASES)
    assert not clashes, f"这些限定词同时也是别名，会把自己否掉：{clashes}"


def test_sanskrit_spelling_variants_of_the_same_pose():
    """同一个体式的梵文写法不止一种，整词匹配对拼写变体不宽容。

    `Setu Bandhasana` 和 `Setu Bandha Sarvangasana` 是同一个桥式，但
    "bandhasana" 是一个词，别名 "setu bandha" 按整词匹配进不去 —— 用户素材里
    那张就因此报了「认不出名字」。
    """
    assert _label("Setu Bandhasana.jpg") == ("bridge", MATCHED)
    assert _label("Setu Bandha Sarvangasana - Bridge.png") == ("bridge", MATCHED)


def test_alias_must_match_whole_words():
    """别名是按词匹配的，不能被更长的词裹进去误命中。"""
    assert _label("treetop-yoga-retreat.jpg") == (None, UNKNOWN)
    assert _label("Vrksasana - Tree.png") == ("tree", MATCHED)


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
