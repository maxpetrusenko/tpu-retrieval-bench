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
    too_long = {str(p.relative_to(ROOT)): n for p in files
                if (n := len(p.read_text().splitlines())) >= MAX_LOC}
    assert too_long == {}


def test_py_typed_marker_ships_in_the_wheel():
    assert (ROOT / "src" / "trbench" / "py.typed").is_file()
    wheel = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src/trbench"]
    assert "src/trbench/py.typed" in wheel["include"]
    assert "Typing :: Typed" in PYPROJECT["project"]["classifiers"]


def test_mypy_is_strict_for_the_library():
    cfg = PYPROJECT["tool"]["mypy"]
    assert cfg["check_untyped_defs"] is True and "src/trbench" in cfg["files"]
    strict = next(o for o in cfg["overrides"] if "trbench.*" in o["module"])
    assert strict["disallow_untyped_defs"] and strict["disallow_untyped_calls"]
    assert strict["warn_return_any"] and strict["disallow_any_generics"]


def test_type_ignores_are_few_targeted_and_commented():
    hits = [ln for p in (ROOT / "src").rglob("*.py") for ln in p.read_text().splitlines()
            if "type: ignore" in ln]
    assert len(hits) <= 4
    assert all(re.search(r"type: ignore\[[a-z-]+\]\s+#\s+\S", ln) for ln in hits), hits


def test_version_agrees_across_pyproject_citation_and_changelog():
    version = PYPROJECT["project"]["version"]
    cff = (ROOT / "CITATION.cff").read_text()
    assert re.search(rf"^version: {re.escape(version)}$", cff, re.MULTILINE)
    assert re.search(r"^license: Apache-2\.0$", cff, re.MULTILINE)
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}$", changelog, re.MULTILINE)
    unreleased = changelog.split("## Unreleased")[1].split("## [")[0]
    assert unreleased.strip() == ""  # nothing pending after a release section is cut


def test_ci_pins_every_action_to_a_sha_and_runs_mypy_in_the_hosted_job():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    uses = re.findall(r"^\s*-?\s*uses:\s*(\S+)", ci, re.MULTILINE)
    assert uses and all(re.fullmatch(r"[\w./-]+@[0-9a-f]{40}", u) for u in uses), uses
    hosted = ci.split("  test:")[0]
    assert "uv run mypy" in hosted and "uv sync --locked" in hosted
