"""体式模板之间的区分度测试。

模板集越大，最容易出的问题不是「认不出」，而是**A 体式被 B 的模板认走**。
这类错误在真实视频里已经发生过好几次：

* 旋转 180° 的身体被判成山式（缺有符号的朝向门槛）；
* 上犬式被三角伸展式认走（三角式缺站姿约束，两者躯干倾角都在 0.5 附近）；
* 平板式在战士三式模板上拿到 0.8（战士三式缺站姿约束）；
* 双臂上举的战士一式被判成战士二式（战士二式只查两腕连线水平，
  而上举时两腕同样等高）；
* 侧板式被平板式认走（平板式只查主侧那只手，侧板式撑地的手蒙混过关）；
* 反板式被平板式认走（2D 剪影近乎镜像，缺头部朝向这一对称判据）。

每一条都是先在真实数据上出错、再回头补判据。这个测试把补上的判据钉死：
遍历所有标准骨架，断言每具骨架的最高分模板就是它自己。加模板或调容差时
若破坏了任何一对区分，这里立刻报错，并指出被谁抢走了。

    python tests/test_templates.py      或      python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yoga_grid.reference import CANONICAL  # noqa: E402
from yoga_grid import landmarks as L  # noqa: E402
from yoga_grid.poses import TEMPLATES, TEMPLATES_BY_KEY, match_pose, score_by_key  # noqa: E402


def test_every_template_has_a_canonical_skeleton():
    """新加模板必须同时提供标准骨架，否则它的区分度无人把关。"""
    missing = sorted({t.key for t in TEMPLATES} - set(CANONICAL))
    assert not missing, f"以下模板缺标准骨架：{missing}（请在 yoga_grid/reference.py 里补上）"

    unknown = sorted(set(CANONICAL) - {t.key for t in TEMPLATES})
    assert not unknown, f"以下骨架没有对应模板：{unknown}"


def test_each_skeleton_matches_its_own_template():
    """每具标准骨架的最高分模板必须是它自己。"""
    problems: list[str] = []
    for key, build in CANONICAL.items():
        norm = L.normalize(build())
        match = match_pose(norm)
        if match is None:
            own = score_by_key(norm, key)
            problems.append(
                f"{key}: 没有任何模板过线"
                f"（自身模板得分 {own.score:.2f}，门槛 {TEMPLATES_BY_KEY[key].min_score:.2f}）"
            )
        elif match.key != key:
            own = score_by_key(norm, key)
            problems.append(
                f"{key}: 被 {match.key} 认走"
                f"（{match.key} {match.score:.2f} vs 自身 {own.score:.2f}）"
            )
    assert not problems, "模板区分度回归：\n  " + "\n  ".join(problems)


def test_own_template_clears_its_threshold():
    """标准骨架在自身模板上要明显高于门槛，留出真实数据的抖动余量。"""
    problems: list[str] = []
    for key, build in CANONICAL.items():
        template = TEMPLATES_BY_KEY[key]
        match = score_by_key(L.normalize(build()), key)
        assert match is not None
        if match.score < template.min_score + 0.15:
            problems.append(
                f"{key}: {match.score:.2f}，门槛 {template.min_score:.2f}，余量不足 0.15"
            )
    assert not problems, "自身得分余量不足：\n  " + "\n  ".join(problems)


def test_orientation_gate_rejects_wrong_facing():
    """朝向门槛必须挡住方向明显不符的体式。

    抽查三对方向相反的体式：站姿 vs 倒垂、俯卧 vs 仰卧、髋高 vs 肩高。
    """
    pairs = [
        ("mountain", "uttanasana"),      # 直立 vs 向下倒垂
        ("downdog", "updog"),            # 髋最高 vs 肩最高
        ("plank", "reverse_plank"),      # 俯卧 vs 仰卧
    ]
    for a, b in pairs:
        norm_a = L.normalize(CANONICAL[a]())
        norm_b = L.normalize(CANONICAL[b]())
        score_a_on_b = score_by_key(norm_b, a)
        score_b_on_a = score_by_key(norm_a, b)
        own_a = score_by_key(norm_a, a)
        own_b = score_by_key(norm_b, b)
        assert score_a_on_b.score < own_b.score, f"{a} 在 {b} 的骨架上分数过高"
        assert score_b_on_a.score < own_a.score, f"{b} 在 {a} 的骨架上分数过高"


def test_standing_templates_reject_prone_bodies():
    """允许躯干大幅倾斜的站姿体式，必须靠站姿约束挡住俯卧体式。

    这是「上犬式被三角伸展式认走」那个真实 bug 的回归测试 —— 光靠朝向门槛
    不够，因为上犬式和三角伸展式的躯干竖直分量都在 0.5 附近。
    """
    prone = ["updog", "plank", "chaturanga"]
    standing = ["triangle", "parsvakonasana", "warrior3"]
    for prone_key in prone:
        norm = L.normalize(CANONICAL[prone_key]())
        for standing_key in standing:
            match = score_by_key(norm, standing_key)
            threshold = TEMPLATES_BY_KEY[standing_key].min_score
            assert match.score < threshold, (
                f"{standing_key} 在 {prone_key}（俯卧）的骨架上拿到 {match.score:.2f}，"
                f"超过门槛 {threshold:.2f}"
            )


def test_every_pose_has_cues_and_annotations():
    """每个体式都要有发力要点，且要点锚点必须能解析。

    锚点写错（拼错关键点名、用了不存在的虚拟点）在渲染时才会炸，而渲染只在
    出图时才跑 —— 放到测试里，改数据就能立刻发现。
    """
    from yoga_grid.landmarks import _VIRTUAL
    from yoga_grid.reference import ANNOTATED, CUES

    keys = set(CANONICAL)
    assert not keys - set(CUES), f"缺纯文字要点：{sorted(keys - set(CUES))}"
    assert not keys - set(ANNOTATED), f"缺带锚点要点：{sorted(keys - set(ANNOTATED))}"

    valid = set(L.NAMES) | set(_VIRTUAL)
    for key, cue_list in ANNOTATED.items():
        for cue in cue_list:
            assert cue.anchor in valid, f"{key}：锚点 {cue.anchor!r} 不是有效关键点"
            assert cue.text.strip(), f"{key}：要点文字为空"


def test_restore_landmarks_handles_both_formats():
    """scores.json 的两种 landmarks 格式都要正确还原。

    回归测试：存储格式从「33×2 归一化骨架」换成「33×3 原始关键点」后，旧文件
    被当成新格式读，会把已归一化的坐标再乘一次画面宽高 —— 几何全错而且不报错。
    """
    from yoga_grid.report import _restore_landmarks

    pts = CANONICAL["warrior2"]()
    expected = L.normalize(pts)

    # 当前格式：归一化 x/y + 置信度，还原后的骨架应与直接归一化一致
    raw = [[float(x) / 1000.0, float(y) / 1000.0, 1.0] for x, y in pts]
    lm, norm = _restore_landmarks(raw, 1000, 1000)
    assert lm is not None and lm.shape == (L.N_LANDMARKS, 3)
    assert norm is not None
    import numpy as np

    assert np.allclose(norm, expected, atol=1e-6)

    # 早期格式：本身就是归一化骨架，必须原样采用，不能再缩放一次
    legacy = [[float(x), float(y)] for x, y in expected]
    lm2, norm2 = _restore_landmarks(legacy, 1920, 1080)
    assert lm2 is None, "早期格式没有置信度，lm 应留空以跳过遮脸"
    assert np.allclose(norm2, expected, atol=1e-6)

    assert _restore_landmarks(None, 100, 100) == (None, None)
    assert _restore_landmarks([], 100, 100) == (None, None)


def test_every_pose_renders():
    """19 个体式的线稿和对照卡都要能渲染出来，不抛异常。"""
    from yoga_grid.reference import render_pose, render_reference_card

    for key in CANONICAL:
        art = render_pose(key, 160)
        assert art.size == (160, 160)
        card = render_reference_card(key, width=700)
        assert card.size[0] == 700 and card.size[1] > 200


def test_mirrored_skeleton_scores_the_same():
    """左右镜像的同一体式，在自身模板上得分必须一致。

    非对称模板会左右各试一遍取高分，镜像后分数不该变。
    """
    for key, build in CANONICAL.items():
        norm = L.normalize(build())
        mirrored = L.mirror_pose(norm)
        a = score_by_key(norm, key)
        b = score_by_key(mirrored, key)
        assert abs(a.score - b.score) < 1e-6, (
            f"{key} 镜像后得分从 {a.score:.4f} 变为 {b.score:.4f}"
        )


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
