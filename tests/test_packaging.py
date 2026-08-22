"""Guards on the files a user touches before any of our code runs.

A mistake in a requirements file fails at ``pip install`` time, before there
is a program to test.  These checks are cheap and catch the class of problem
that only shows up on someone else's machine.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REQUIREMENT_FILES = sorted(ROOT.glob("requirements*.txt"))


def test_there_are_requirement_files_to_check():
    assert REQUIREMENT_FILES, "找不到 requirements*.txt"


@pytest.mark.parametrize("path", REQUIREMENT_FILES, ids=lambda p: p.name)
def test_requirements_are_pure_ascii(path):
    """pip reads these with the system locale encoding, not UTF-8.

    A Chinese comment here raised ``UnicodeDecodeError: 'charmap' codec can't
    decode byte 0x81`` for a Windows user on a cp1252 locale -- the install
    failed before a single line of the program ran, while Linux and CI (both
    UTF-8) saw nothing wrong.  Keep these files ASCII; the Chinese belongs in
    the README.
    """
    raw = path.read_bytes()
    offenders = [(i, byte) for i, byte in enumerate(raw) if byte > 127]
    assert not offenders, (
        f"{path.name} 含 {len(offenders)} 个非 ASCII 字节，"
        f"首个在偏移 {offenders[0][0]} (0x{offenders[0][1]:02x})。"
        "pip 用系统本地编码读这个文件，非 UTF-8 环境（Windows）会装不上。"
    )


@pytest.mark.parametrize("path", REQUIREMENT_FILES, ids=lambda p: p.name)
def test_requirements_parse_as_requirement_lines(path):
    for number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        # Everything left should look like `name<spec>`, optionally with a
        # trailing comment.
        requirement = stripped.split("#")[0].strip()
        assert requirement[0].isalpha(), f"{path.name}:{number} 看起来不像依赖行：{line!r}"


def test_declared_dependencies_match_requirements():
    """pyproject.toml and requirements.txt must not drift apart."""
    import tomllib

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = {
        dep.split(">")[0].split("<")[0].split("=")[0].strip().lower()
        for dep in pyproject["project"]["dependencies"]
    }
    pinned = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            pinned.add(stripped.split(">")[0].split("<")[0].split("=")[0].strip().lower())
    assert declared == pinned, f"pyproject 声明 {declared}，requirements.txt 是 {pinned}"
