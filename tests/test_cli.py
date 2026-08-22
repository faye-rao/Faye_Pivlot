import pytest
from figures import WARRIOR_II_RIGHT, figure

from yoga_coach import POSES, evaluate, get_pose
from yoga_coach.cli import build_parser, list_poses, main
from yoga_coach.console import ConsoleReporter, report
from yoga_coach.session import CoachSession


class TestParser:
    def test_defaults_to_the_first_camera_in_auto_mode(self):
        args = build_parser().parse_args([])
        assert args.source == "0"
        assert args.pose == "auto"

    def test_rejects_an_unknown_model(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--model", "gigantic"])


class TestListPoses:
    def test_lists_every_pose_with_its_camera_hint(self, capsys):
        assert main(["--list-poses"]) == 0
        out = capsys.readouterr().out
        for pose in POSES:
            assert pose.key in out
            assert pose.name.zh in out
            assert pose.view.zh in out

    def test_english_listing(self, capsys):
        list_poses("en")
        out = capsys.readouterr().out
        assert "Warrior II" in out


def test_unknown_pose_exits_with_a_helpful_message(capsys):
    assert main(["--pose", "headstand"]) == 2
    err = capsys.readouterr().err
    assert "warrior2" in err
    assert "--list-poses" in err


class TestConsoleOutput:
    def test_report_lists_every_correction(self, capsys):
        bad = figure({**WARRIOR_II_RIGHT, "right_knee": (0.800, 0.660)})
        report(evaluate(bad, get_pose("warrior2")), lang="zh")
        out = capsys.readouterr().out
        assert "战士二式" in out
        assert "需要调整" in out

    def test_report_congratulates_a_clean_pose(self, capsys):
        report(evaluate(figure(WARRIOR_II_RIGHT), get_pose("warrior2")), lang="zh")
        assert "都通过了" in capsys.readouterr().out

    def test_report_names_the_checks_it_could_not_measure(self, capsys):
        cropped = figure(WARRIOR_II_RIGHT, hidden=("left_wrist", "left_elbow"))
        report(evaluate(cropped, get_pose("warrior2")), lang="en")
        out = capsys.readouterr().out
        assert "Not measured" in out

    def test_reporter_prints_once_per_change(self, capsys):
        session = CoachSession(pose=get_pose("warrior2"))
        reporter = ConsoleReporter(lang="zh", interval=10.0)
        skeleton = figure(WARRIOR_II_RIGHT)
        for i in range(5):
            reporter.update(session.update(skeleton, i * 0.1), i * 0.1)
        assert len(capsys.readouterr().out.strip().splitlines()) == 1

    def test_reporter_prints_again_when_the_advice_changes(self, capsys):
        session = CoachSession(pose=get_pose("warrior2"))
        reporter = ConsoleReporter(lang="zh", interval=10.0)
        good = figure(WARRIOR_II_RIGHT)
        bad = figure({**WARRIOR_II_RIGHT, "right_knee": (0.800, 0.660)})
        for i in range(5):
            reporter.update(session.update(good, i * 0.1), i * 0.1)
        for i in range(5, 20):
            reporter.update(session.update(bad, i * 0.1), i * 0.1)
        assert len(capsys.readouterr().out.strip().splitlines()) > 1

    def test_reporter_reports_a_missing_body(self, capsys):
        session = CoachSession(pose=get_pose("mountain"))
        reporter = ConsoleReporter(lang="zh")
        reporter.update(session.update(None, 0.0), 0.0)
        assert "没有检测到人" in capsys.readouterr().out
