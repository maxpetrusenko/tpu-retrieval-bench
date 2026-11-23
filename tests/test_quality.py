"""Repo hygiene checks that docs/QUALITY.md cites as evidence: size, typing, packaging, CI pins."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_LOC = 500  # CONTRIBUTING.md: files under 500 LOC
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_python_files_stay_under_500_lines():
    files = [p for d in ("src", "bench", "tests") for p in (ROOT / d).rglob("*.py")]
    assert files
    too_long = {
        str(p.relative_to(ROOT)): n
        for p in files
        if (n := len(p.read_text().splitlines())) >= MAX_LOC
    }
    assert too_long == {}


def test_py_typed_marker_ships_in_the_wheel():
    assert (ROOT / "src" / "trbench" / "py.typed").is_file()
    wheel = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/trbench"]
    assert "src/trbench/py.typed" in wheel["include"]
    assert "Typing :: Typed" in PYPROJECT["project"]["classifiers"]


def test_mypy_is_strict_for_the_library_and_bench():
    cfg = PYPROJECT["tool"]["mypy"]
    assert cfg["strict"] is True
    assert "src/trbench" in cfg["files"] and "bench" in cfg["files"]
    assert not cfg.get("overrides"), "no per-module relaxations or exemptions"


def test_type_ignores_are_few_targeted_and_commented():
    for tree, limit in (("src", 4), ("bench", 4)):
        hits = [
            ln
            for p in (ROOT / tree).rglob("*.py")
            for ln in p.read_text().splitlines()
            if "type: ignore" in ln
        ]
        assert len(hits) <= limit, (tree, hits)
        assert all(re.search(r"type: ignore\[[a-z-]+\]\s+#\s+\S", ln) for ln in hits), hits


def test_version_agrees_across_pyproject_and_citation():
    version = PYPROJECT["project"]["version"]
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(version)}$", cff, re.MULTILINE)
    assert re.search(r"^license: Apache-2\.0$", cff, re.MULTILINE)


def test_changelog_covers_the_current_version_by_a_section_or_pending_entries():
    """Structure only. Whether Unreleased must be empty is decided at tag time by
    release.yml (`scripts/changelog.py release`), not here, so work can pile up between releases."""
    version = PYPROJECT["project"]["version"]
    changelog = (ROOT / "CHANGELOG.md").read_text()
    has_section = re.search(
        rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE
    )
    pending = changelog.split("## Unreleased")[1].split("\n## ")[0].strip()
    assert has_section or pending, f"no [{version}] section and nothing under Unreleased"


def test_ci_runs_mypy_in_the_hosted_job():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    hosted = ci.split("\n  test:")[0]
    assert "uv run mypy" in hosted and "uv sync --locked" in hosted


def test_ci_and_pre_commit_check_formatting():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    hosted = ci.split("\n  test:")[0]
    assert "uv run ruff format --check ." in hosted
    assert "ruff-format" in (ROOT / ".pre-commit-config.yaml").read_text()
