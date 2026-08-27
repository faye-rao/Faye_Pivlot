"""Keep ``docs/yoga_coach.md`` honest about the code.

The pose tables, per-pose check tables and module list in that document are
generated from the source by ``tools/update_readme.py``.  This test runs the
generator in check mode, so widening a target band or adding a pose without
regenerating the docs fails here instead of shipping a document that quietly
disagrees with the program.

Note this is the *coach's* document, not the repo README -- that one is
yoga_grid's front page and nothing here covers it.

Fix a failure with::

    python tools/update_readme.py
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "update_readme.py"

sys.path.insert(0, str(ROOT / "tools"))

import update_readme  # noqa: E402

# Taken from the generator rather than spelled out again, so moving the
# document cannot leave this test pointing at the old path.
DOC = update_readme.DOC


def test_readme_is_up_to_date():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        f"{update_readme.LABEL} 与代码不同步，请运行 python tools/update_readme.py\n\n"
        + result.stdout
        + result.stderr
    )


def test_generator_is_idempotent():
    once = update_readme.render(DOC.read_text(encoding="utf-8"))
    assert update_readme.render(once) == once


@pytest.fixture(scope="module")
def readme():
    return DOC.read_text(encoding="utf-8")


class TestGeneratedContent:
    def test_every_block_marker_is_present_and_filled(self, readme):
        for name in update_readme.BLOCKS:
            begin = f"<!-- BEGIN GENERATED: {name} -->"
            end = f"<!-- END GENERATED: {name} -->"
            assert begin in readme and end in readme, name
            body = readme.split(begin)[1].split(end)[0]
            assert body.strip(), f"{name} 区块是空的"

    def test_every_pose_and_check_is_documented(self, readme):
        from yoga_coach import POSES

        for pose in POSES:
            assert f"`{pose.key}`" in readme
            assert pose.name.zh in readme
            for check in pose.checks:
                assert check.label.zh in readme, f"{pose.key}.{check.key}"
                assert check.target_text() in readme, f"{pose.key}.{check.key}"

    def test_every_module_is_documented(self, readme):
        for path in (ROOT / "yoga_coach").glob("*.py"):
            assert f"`{path.name}`" in readme


class TestSharedTestDirectory:
    """``tests/`` holds both tools' tests, told apart by filename prefix.

    The convention is load-bearing rather than cosmetic: the generated stats
    line counts the coach's tests by skipping the prefix, so a yoga_grid test
    filed under the wrong name would quietly inflate the coach's documented
    test count -- exactly the kind of drift the generator exists to prevent.

    Keyed on real import statements, not the mere appearance of the name:
    this very file mentions yoga_grid in prose without importing it.
    """

    IMPORTS_GRID = re.compile(r"^\s*(?:from|import)\s+yoga_grid\b", re.MULTILINE)

    def test_the_prefix_marks_exactly_the_grid_tests(self):
        for path in sorted((ROOT / "tests").glob("test_*.py")):
            source = path.read_text(encoding="utf-8")
            imports_grid = bool(self.IMPORTS_GRID.search(source))
            named_grid = path.name.startswith(update_readme.GRID_TEST_PREFIX)
            assert imports_grid == named_grid, (
                f"{path.name}: import yoga_grid={imports_grid} 但文件名前缀"
                f"={named_grid}。属于 yoga_grid 的测试请用 "
                f"{update_readme.GRID_TEST_PREFIX} 前缀命名。"
            )

    def test_the_counted_tests_are_a_strict_subset_of_the_directory(self):
        counted = update_readme._count_test_functions()
        everything = sum(
            len(re.findall(r"^\s*def test_", p.read_text(encoding="utf-8"), re.MULTILINE))
            for p in (ROOT / "tests").glob("test_*.py")
        )
        assert 0 < counted < everything, (counted, everything)


class TestModuleNotes:
    def test_notes_cover_exactly_the_package(self):
        actual = {p.name for p in (ROOT / "yoga_coach").glob("*.py")}
        assert set(update_readme.MODULE_NOTES) == actual

    def test_a_missing_marker_is_an_error(self):
        with pytest.raises(SystemExit, match="poses"):
            update_readme.render("# 没有任何生成标记的 README")
