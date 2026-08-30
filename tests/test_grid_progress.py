"""``tools/progress.py`` —— 改进点的连续达标与关闭。

这个文件的价值集中在**什么情况下不该关闭一条改进点**。关错了没有任何提示：
文件上写着「连续 3 天达标，已关闭」，看起来正是想要的结果，而人根本没练到
那个地方。所以下面一多半用例是反面的。

    python -m pytest tests/test_grid_progress.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import progress as P  # noqa: E402
from calibrate import Frame  # noqa: E402

from yoga_grid import landmarks as L  # noqa: E402
from yoga_grid.reference import CANONICAL  # noqa: E402


def _norm(key: str) -> np.ndarray:
    return L.normalize(CANONICAL[key]())


def _bend_front_knee(norm: np.ndarray, degrees: float) -> np.ndarray:
    """把战士二式的前膝改成另一个角度，用来造「同一天两种做法」。"""
    out = norm.copy()
    hip, knee = out[L.IDX["left_hip"]], out[L.IDX["left_knee"]]
    shin = float(np.hypot(*(out[L.IDX["left_ankle"]] - knee)))
    thigh_dir = (knee - hip) / max(np.hypot(*(knee - hip)), 1e-9)
    theta = np.radians(180.0 - degrees)
    c, s_ = np.cos(theta), np.sin(theta)
    d = np.array([thigh_dir[0] * c - thigh_dir[1] * s_,
                  thigh_dir[0] * s_ + thigh_dir[1] * c])
    out[L.IDX["left_ankle"]] = knee + d * shin
    return out


def _day(verdict: str, date: str, median: float = 100.0, full: float = 1.0) -> P.Day:
    return P.Day(date, 8, median, full, verdict)


def _item(*days: P.Day) -> P.Item:
    return P.Item("warrior2", "战士二式", "前膝屈 90°", 92.0, 15.0, "°",
                  history=list(days))


# --------------------------------------------------------------------------
# 连续计数
# --------------------------------------------------------------------------


def test_streak_counts_consecutive_days():
    item = _item(_day(P.MET, "2026-08-28"), _day(P.MET, "2026-08-29"))
    assert item.streak == 2


def test_a_miss_resets_the_streak():
    item = _item(_day(P.MET, "2026-08-28"), _day(P.MISSED, "2026-08-29"),
                 _day(P.MET, "2026-08-30"))
    assert item.streak == 1, "未达标之后要从头数"


def test_skipped_days_neither_advance_nor_reset():
    """没练到那个体式不算做错，也不算做对。

    清零会逼着人每天把所有体式过一遍；计入则等于没练也能关闭。
    """
    item = _item(_day(P.MET, "2026-08-28"), _day(P.SKIPPED, "2026-08-29"),
                 _day(P.MET, "2026-08-30"))
    assert item.streak == 2


def test_a_skip_cannot_by_itself_close_an_item():
    item = _item(*(_day(P.SKIPPED, f"2026-08-2{n}") for n in range(1, 8)))
    assert item.streak == 0


# --------------------------------------------------------------------------
# 关闭
# --------------------------------------------------------------------------


def test_closes_after_the_required_streak():
    items = [_item(_day(P.MET, "2026-08-28"), _day(P.MET, "2026-08-29"))]
    results = {("warrior2", "前膝屈 90°"): _day(P.MET, "2026-08-30")}
    items, changes = P.apply(items, results, streak=3)
    assert items[0].closed_on == "2026-08-30"
    assert any("关闭" in c for c in changes)


def test_does_not_close_one_day_early():
    items = [_item(_day(P.MET, "2026-08-29"))]
    items, _ = P.apply(items, {("warrior2", "前膝屈 90°"): _day(P.MET, "2026-08-30")},
                       streak=3)
    assert not items[0].closed_on and items[0].streak == 2


def test_rerunning_the_same_day_does_not_inflate_the_streak():
    """改一次模板重跑一遍是常事，那不是多练了一天。"""
    items = [_item(_day(P.MET, "2026-08-29"))]
    key = ("warrior2", "前膝屈 90°")
    for _ in range(5):
        items, _ = P.apply(items, {key: _day(P.MET, "2026-08-30")}, streak=3)
    assert items[0].streak == 2, "同一天重复跑只该更新那一天"
    assert not items[0].closed_on


def test_closed_items_are_left_alone():
    item = _item(_day(P.MET, "2026-08-30"))
    item.closed_on = "2026-08-30"
    items, changes = P.apply([item], {("warrior2", "前膝屈 90°"): _day(P.MISSED, "2026-09-01")},
                             streak=3)
    assert items[0].closed_on == "2026-08-30" and not changes


# --------------------------------------------------------------------------
# 判定：三种「今天不算数」
# --------------------------------------------------------------------------


def _frames(norm: np.ndarray, n: int, vis: np.ndarray | None = None) -> list[Frame]:
    return [Frame("v", float(i), norm, vis) for i in range(n)]


def test_occluded_readings_are_skipped_not_met():
    """**最要紧的一条。** 被遮挡的关键点上，MediaPipe 的推测值看着完全合理。

    一条「双腿伸直」如果靠推测值连着三天达标，它是被蒙混关掉的。
    """
    norm = _norm("warrior2")
    vis = np.ones(33)
    vis[[25, 26, 27, 28]] = 0.1          # 双膝双踝全判为猜的
    day = P.evaluate(_frames(norm, 8, vis), "2026-08-30")
    knee = day.get(("warrior2", "前膝屈 90°"))
    assert knee is not None and knee.verdict == P.SKIPPED
    assert "遮挡" in knee.note


def test_a_day_that_swings_wider_than_the_tolerance_is_skipped():
    """同一天之内摆动比容差还大时，中位数概括不了它。

    实测过一例：侧角伸展式的「双臂成一线」一天内从 104° 摆到 179°（容差 20°）。
    按中位数判达标等于抛硬币，而抛三次正面就关掉一条改进点。
    """
    a = _norm("warrior2")
    b = _bend_front_knee(a, 175.0)
    frames = _frames(a, 6) + _frames(b, 6)
    day = P.evaluate(frames, "2026-08-30")
    knee = day[("warrior2", "前膝屈 90°")]
    assert knee.verdict == P.SKIPPED and "散布" in knee.note


def test_too_few_frames_gives_no_verdict_at_all():
    day = P.evaluate(_frames(_norm("warrior2"), P.MIN_FRAMES - 1), "2026-08-30")
    assert ("warrior2", "前膝屈 90°") not in day


def _decide(**kw) -> str:
    base = dict(median=92.0, full_rate=1.0, occluded=0.0, spread=2.0,
                target=92.0, tol=15.0)
    return P.decide(**{**base, **kw})[0]


def test_decide_met_is_the_ordinary_case():
    assert _decide() == P.MET


def test_decide_misses_when_the_median_is_outside():
    assert _decide(median=116.0) == P.MISSED


def test_median_inside_tolerance_but_most_frames_failing_is_not_met():
    """中位数擦着区间边缘过去时，一半的帧其实在界外。压线不算做到。

    实测见过：金字塔式「躯干向下折叠」中位 -0.30、目标 -0.65 ± 0.35，
    正好压在边上，满分率只有 52%。
    """
    assert _decide(median=107.0, full_rate=0.45) == P.MISSED
    assert _decide(median=107.0, full_rate=0.55) == P.MET


def test_the_two_no_verdict_gates_outrank_everything():
    """遮挡和摆动说的是「这个读数没资格下判断」，不是「做得好不好」。

    所以即使中位数正正好好落在目标上，它们也要压过 MET —— 否则一条改进点
    可以靠推测值或者靠抛硬币被关掉。
    """
    assert _decide(occluded=0.8) == P.SKIPPED
    assert _decide(spread=40.0) == P.SKIPPED
    assert _decide(median=116.0, occluded=0.8) == P.SKIPPED


# --------------------------------------------------------------------------
# 开条目
# --------------------------------------------------------------------------


def test_only_missed_days_open_an_item():
    """达标的、跳过的都不该开条目。跳过尤其危险 —— 那是「不知道」。"""
    for verdict in (P.MET, P.SKIPPED):
        assert not P.worth_tracking(_day(verdict, "2026-08-30", 120.0), 92.0, 15.0)
    assert P.worth_tracking(_day(P.MISSED, "2026-08-30", 120.0, 0.0), 92.0, 15.0)


def test_a_near_miss_does_not_open_an_item():
    """刚出界一点点、而且大部分帧仍达标的，不值得占一条。"""
    assert not P.worth_tracking(_day(P.MISSED, "2026-08-30", 99.0, 0.9), 92.0, 15.0)


# --------------------------------------------------------------------------
# 文件往返
# --------------------------------------------------------------------------


def test_round_trip_preserves_everything_that_matters():
    item = _item(_day(P.MET, "2026-08-28", 95.0, 0.9),
                 _day(P.MISSED, "2026-08-29", 118.5, 0.0),
                 _day(P.SKIPPED, "2026-08-30", 100.0, 0.5))
    item.manual_note = "手写的备注要留住"
    back = P.parse(P.render([item]))
    assert len(back) == 1
    got = back[0]
    assert got.key == item.key
    assert got.manual_note == item.manual_note
    assert [(d.date, d.verdict) for d in got.history] == [
        (d.date, d.verdict) for d in item.history
    ]
    assert got.history[1].median == 118.5


def test_round_trip_keeps_closed_items_closed():
    item = _item(_day(P.MET, "2026-08-30"))
    item.closed_on = "2026-08-30"
    assert P.parse(P.render([item]))[0].closed_on == "2026-08-30"


def test_parsing_ignores_sections_it_does_not_understand():
    """文件是给人改的，手写的段落不能让它崩。"""
    text = P.render([_item(_day(P.MET, "2026-08-30"))])
    text += "\n### 我自己加的一条 · 随手写的\n\n- 想到什么写什么\n"
    text += "\n### 没有分隔符的标题\n\n- 模板：warrior2\n"
    items = P.parse(text)
    assert [i.key for i in items] == [("warrior2", "前膝屈 90°")]


def test_renamed_template_or_check_drops_the_item_rather_than_guessing():
    """模板改了名字，旧条目对不上就丢掉 —— 不要挂到一个相近的检查项上。"""
    text = P.render([_item(_day(P.MET, "2026-08-30"))])
    assert not P.parse(text.replace("- 模板：warrior2", "- 模板：不存在的体式"))
    assert not P.parse(text.replace("前膝屈 90°", "已经改过名的检查项"))


def test_empty_file_renders_and_reparses():
    assert P.parse(P.render([])) == []


def _run_all() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
