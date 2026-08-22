"""Keep README.md honest about the code.

The pose tables, per-pose check tables and module list in the README are
generated from the source by ``tools/update_readme.py``.  This test runs the
generator in check mode, so widening a target band or adding a pose without
regenerating the docs fails here instead of shipping a README that quietly
disagrees with the program.

Fix a failure with::

    python tools/update_readme.py
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "tools" / "update_readme.py"
README = ROOT / "README.md"

sys.path.insert(0, str(ROOT / "tools"))

import update_readme  # noqa: E402


def test_readme_is_up_to_date():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, (
        "README.md 与代码不同步，请运行 python tools/update_readme.py\n\n"
        + result.stdout
        + result.stderr
    )


def test_generator_is_idempotent():
    once = update_readme.render(README.read_text(encoding="utf-8"))
    assert update_readme.render(once) == once


@pytest.fixture(scope="module")
def readme():
    return README.read_text(encoding="utf-8")


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


class TestModuleNotes:
    def test_notes_cover_exactly_the_package(self):
        actual = {p.name for p in (ROOT / "yoga_coach").glob("*.py")}
        assert set(update_readme.MODULE_NOTES) == actual

    def test_a_missing_marker_is_an_error(self):
        with pytest.raises(SystemExit, match="poses"):
            update_readme.render("# 没有任何生成标记的 README")
