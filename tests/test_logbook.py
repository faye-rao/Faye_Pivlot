"""The practice log: what gets sampled, and what the summary says.

The point of the summary is to answer one question without the practitioner
taking any notes mid-pose: which target bands are wrong for this body?
"""

import csv

import pytest
from figures import PLANK_SIDE, WARRIOR_II_RIGHT, figure

from yoga_coach import get_pose
from yoga_coach.logbook import Logbook, _display_width
from yoga_coach.session import CoachSession

SHALLOW = {**WARRIOR_II_RIGHT, "right_knee": (0.610, 0.680)}
CROPPED = (
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
)


def practise(logbook, skeleton, seconds=20.0, step=0.1, start=0.0, pose="warrior2", rows=False):
    session = CoachSession(pose=get_pose(pose))
    now = start
    while now < start + seconds:
        logbook.record(session.update(skeleton, now), now, collect_rows=rows)
        now += step
    return logbook


class TestSampling:
    def test_records_measured_checks(self):
        book = practise(Logbook(), figure(WARRIOR_II_RIGHT))
        assert not book.empty
        assert "warrior2" in book.poses
        assert "front_knee_bend" in book.poses["warrior2"]

    def test_ignores_the_settling_period(self):
        """The first seconds in a pose are you arranging yourself; counting
        them would drag every median towards 'wrong'."""
        book = Logbook(settle=5.0)
        practise(book, figure(WARRIOR_II_RIGHT), seconds=4.0)
        assert book.empty

    def test_samples_at_the_interval_not_every_frame(self):
        book = practise(Logbook(interval=1.0, settle=0.0), figure(WARRIOR_II_RIGHT), seconds=10.0)
        samples = book.poses["warrior2"]["front_knee_bend"].samples
        assert 9 <= samples <= 11  # ~one per second, not 100

    def test_skips_frames_where_the_body_is_not_fully_visible(self):
        book = practise(Logbook(settle=0.0), figure(WARRIOR_II_RIGHT, hidden=CROPPED))
        assert book.empty

    def test_skips_frames_with_no_body_at_all(self):
        book = practise(Logbook(settle=0.0), None)
        assert book.empty

    def test_a_pose_change_restarts_the_settling_clock(self):
        book = Logbook(settle=2.0)
        now = 0.0
        for key, skeleton in (
            ("warrior2", figure(WARRIOR_II_RIGHT)),
            ("plank", figure(PLANK_SIDE)),
        ):
            session = CoachSession(pose=get_pose(key))
            for _ in range(15):  # 1.5s each -- neither reaches the settle time
                book.record(session.update(skeleton, now), now)
                now += 0.1
        assert book.empty


class TestSummary:
    def test_flags_a_check_whose_median_sits_outside_the_band(self):
        book = practise(Logbook(settle=1.0), figure(SHALLOW))
        record = book.poses["warrior2"]["front_knee_bend"]
        assert record.verdict() == "高"  # knee not bent enough
        text = "\n".join(book.summary_lines())
        assert "前膝屈度" in text
        assert "长期偏高" in text

    def test_does_not_flag_a_check_inside_its_band(self):
        book = practise(Logbook(settle=1.0), figure(WARRIOR_II_RIGHT))
        assert book.poses["warrior2"]["front_knee_bend"].verdict() == ""
        assert "← 长期偏" not in "\n".join(book.summary_lines())

    def test_worst_pass_rate_comes_first(self):
        book = practise(Logbook(settle=1.0), figure(SHALLOW))
        records = sorted(book.poses["warrior2"].values(), key=lambda r: r.pass_rate)
        lines = book.summary_lines()
        body = [line for line in lines if line.startswith("  ") and "检查项" not in line]
        assert records[0].check.label.zh in body[0]

    def test_pass_rate_and_median_are_reported(self):
        book = practise(Logbook(settle=1.0), figure(SHALLOW))
        record = book.poses["warrior2"]["front_knee_bend"]
        assert 0.0 <= record.pass_rate <= 1.0
        assert record.median == pytest.approx(record.values[0], abs=2.0)

    def test_summary_says_so_when_nothing_was_captured(self):
        assert "没有采集到有效数据" in "\n".join(Logbook().summary_lines())

    def test_column_alignment_accounts_for_wide_glyphs(self):
        assert _display_width("前膝屈度") == 8
        assert _display_width("abcd") == 4
        book = practise(Logbook(settle=1.0), figure(SHALLOW))
        rows = [
            line
            for line in book.summary_lines()
            if line.startswith("  ") and "%" in line
        ]
        widths = {_display_width(line.split("%")[0]) for line in rows}
        assert len(widths) == 1, "百分比列没有对齐"


class TestCsv:
    def test_writes_a_row_per_check_sample(self, tmp_path):
        book = practise(
            Logbook(interval=1.0, settle=0.0), figure(SHALLOW), seconds=5.0, rows=True
        )
        path = tmp_path / "practice.csv"
        book.write_csv(path)
        with path.open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert set(rows[0]) == {"秒", "体式", "侧", "总分", "检查项", "实测", "目标", "通过"}
        assert any(r["检查项"] == "前膝屈度" and r["通过"] == "否" for r in rows)

    def test_no_rows_collected_writes_nothing(self, tmp_path):
        path = tmp_path / "empty.csv"
        practise(Logbook(settle=1.0), figure(WARRIOR_II_RIGHT), rows=False).write_csv(path)
        assert not path.exists()

    def test_uses_a_bom_so_excel_reads_the_chinese(self, tmp_path):
        book = practise(
            Logbook(interval=1.0, settle=0.0), figure(SHALLOW), seconds=3.0, rows=True
        )
        path = tmp_path / "practice.csv"
        book.write_csv(path)
        assert path.read_bytes().startswith(b"\xef\xbb\xbf")
