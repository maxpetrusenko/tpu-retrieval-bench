"""scripts/changelog.py: structure check, tag-time release gate and the release cut."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
try:
    import changelog
finally:
    sys.path.pop(0)

REPO = changelog.REPO_URL
PENDING = f"""# Changelog

Intro.

## Unreleased

### Added
- new thing

## [0.2.0] - 2026-11-22

### Added
- old thing

## 0.1.0

- first

[Unreleased]: {REPO}/compare/v0.2.0...HEAD
[0.2.0]: {REPO}/releases/tag/v0.2.0
"""


def test_the_real_changelog_has_valid_structure():
    assert changelog.structure_problems((ROOT / "CHANGELOG.md").read_text()) == []


def test_structure_accepts_pending_entries_under_unreleased():
    assert changelog.structure_problems(PENDING) == []


@pytest.mark.parametrize(("mutate", "expect"), [
    (lambda t: t.replace("## Unreleased", "## Next"), "first section"),
    (lambda t: t.replace("## [0.2.0] - 2026-11-22", "## [0.2.0] 2026-11-22"), "bad heading"),
    (lambda t: t.replace(f"[0.2.0]: {REPO}/releases/tag/v0.2.0\n", ""),
     "missing link reference [0.2.0]"),
    (lambda t: t.replace("v0.2.0...HEAD", "v0.1.0...HEAD"), "[Unreleased] must compare"),
    (lambda t: t.replace(REPO, "https://example.com", 1), "does not point at"),
    (lambda t: t.replace("### Added\n- old thing\n", ""), "is empty"),
])
def test_structure_flags_broken_files(mutate, expect):
    errs = changelog.structure_problems(mutate(PENDING))
    assert any(expect in e for e in errs), errs


def test_release_gate_fails_while_entries_sit_under_unreleased():
    errs = changelog.release_problems(PENDING, "0.2.1")
    assert any("no '## [0.2.1]" in e for e in errs)
    assert any("Unreleased still has entries" in e for e in errs)
    only_unreleased = changelog.release_problems(PENDING, "0.2.0")
    assert len(only_unreleased) == 1 and "Unreleased still has entries" in only_unreleased[0]


def test_release_gate_passes_after_the_cut_and_cut_fixes_links():
    cut = changelog.cut(PENDING, "0.2.1", "2026-12-01")
    assert changelog.release_problems(cut, "0.2.1") == []
    assert changelog.structure_problems(cut) == []
    assert "## [0.2.1] - 2026-12-01\n\n### Added\n- new thing" in cut
    assert f"[Unreleased]: {REPO}/compare/v0.2.1...HEAD" in cut
    assert f"[0.2.1]: {REPO}/compare/v0.2.0...v0.2.1" in cut
    assert changelog.sections(cut)[0] == ("Unreleased", "")


def test_cut_refuses_empty_unreleased_and_duplicate_versions():
    cut = changelog.cut(PENDING, "0.2.1", "2026-12-01")
    with pytest.raises(ValueError, match="empty"):
        changelog.cut(cut, "0.2.2", "2026-12-02")
    with pytest.raises(ValueError, match="already exists"):
        changelog.cut(PENDING, "0.2.0", "2026-12-02")
    with pytest.raises(ValueError, match="want VERSION"):
        changelog.cut(PENDING, "v0.2.1", "2026-12-01")


def test_release_gate_rejects_a_missing_or_malformed_version():
    assert changelog.release_problems(PENDING, "latest")
    assert any("no '## [9.9.9]" in e for e in changelog.release_problems(PENDING, "9.9.9"))


def test_cli_cut_updates_the_citation_and_exit_codes(tmp_path):
    (tmp_path / "CHANGELOG.md").write_text(PENDING)
    (tmp_path / "CITATION.cff").write_text("version: 0.2.0\ndate-released: 2026-11-22\n")
    script = str(ROOT / "scripts" / "changelog.py")

    def run(*a: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([sys.executable, script, *a], capture_output=True, text=True,
                              check=False)

    path = str(tmp_path / "CHANGELOG.md")
    assert run("release", "0.2.1", path).returncode == 1
    assert run("cut", "0.2.1", "2026-12-01", path).returncode == 0
    assert (tmp_path / "CITATION.cff").read_text() == "version: 0.2.1\ndate-released: 2026-12-01\n"
    assert run("release", "0.2.1", path).returncode == 0
    assert run("structure", path).returncode == 0
    assert run("cut", "0.2.2", "2026-12-02", path).returncode == 1
    assert run().returncode == 2
