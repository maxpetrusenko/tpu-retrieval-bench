"""Workflow hygiene: pins, privileges, triggers, release notes. Text checks, no YAML parser."""

import re
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows"


def _wf(name: str) -> str:
    return (WF / name).read_text()


def test_every_action_in_every_workflow_is_sha_pinned_with_a_version_comment():
    files = sorted(WF.glob("*.yml"))
    assert {f.name for f in files} >= {"ci.yml", "codeql.yml", "scorecard.yml", "release.yml",
                                       "tpu-smoke.yml"}
    for f in files:
        uses = re.findall(r"^\s*-?\s*uses:\s*(.+)$", f.read_text(), re.MULTILINE)
        assert uses, f.name
        for u in uses:
            assert re.fullmatch(r"[\w./-]+@[0-9a-f]{40} # v\d+(\.\d+)*", u.strip()), (f.name, u)


def test_every_workflow_starts_with_no_permissions():
    for f in WF.glob("*.yml"):
        assert re.search(r"^permissions: \{\}$", f.read_text(), re.MULTILINE), f.name


def test_ci_self_hosted_job_never_runs_for_pull_requests():
    ci = _wf("ci.yml")
    job = ci.split("\n  test:")[1]
    cond = re.search(r"^\s+if: (.+)$", job, re.MULTILINE)
    assert cond is not None
    assert "pull_request" not in cond.group(1)
    assert "push" in cond.group(1) and "workflow_dispatch" in cond.group(1)
    assert "workflow_dispatch:" in ci.split("jobs:")[0]
    assert "self-hosted" not in ci.split("\n  test:")[0]  # hosted jobs stay on GitHub runners


def test_ci_has_the_checks_branch_protection_names():
    ci = _wf("ci.yml")
    assert "name: lint-test (py${{ matrix.python-version }})" in ci
    assert re.search(r"^  audit:\n    name: audit$", ci, re.MULTILINE)
    assert re.search(r"^  package:\n    name: package$", ci, re.MULTILINE)
    for step in ("pytest -q --cov", "validate_results --strict results/tpu/run3 results/tpu/run3-devtime",
                 "doc_stats --check docs/RESULTS.md", "DRY_RUN=1 bash scripts/reproduce_all.sh",
                 "pip-audit"):
        assert step in ci, step


def test_coverage_threshold_is_set():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["tool"]["coverage"]["report"]["fail_under"] >= 80


def test_release_job_only_holds_write_and_checks_the_tag():
    rel = _wf("release.yml")
    assert re.search(r"^on:\n  push:\n    tags: \[\"v\*\"\]", rel, re.MULTILINE)
    assert rel.count("contents: write") == 1
    assert rel.index("permissions: {}") < rel.index("contents: write")
    assert "pyproject.toml" in rel and 'v$version' in rel
    assert "scripts/release_notes.py" in rel and "dist/*" in rel


def test_scorecard_publishes_and_scopes_id_token_to_its_job():
    sc = _wf("scorecard.yml")
    assert "publish_results: true" in sc
    assert sc.count("id-token: write") == 1
    assert sc.index("jobs:") < sc.index("id-token: write")


def test_tpu_smoke_is_manual_only_and_targets_a_tpu_runner():
    wf = _wf("tpu-smoke.yml")
    assert re.search(r"^on:\n  workflow_dispatch:\n", wf, re.MULTILINE)
    assert "push:" not in wf and "pull_request" not in wf and "schedule:" not in wf
    assert "runs-on: [self-hosted, tpu]" in wf
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    jax = next(p["version"] for p in lock["package"] if p["name"] == "jax")
    assert f'"jax[tpu]=={jax}"' in wf


def test_release_notes_extracts_one_section():
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import release_notes
    finally:
        sys.path.pop(0)
    text = "# Changelog\n\n## [0.3.0] - 2027-01-01\n\n### Added\n- b\n\n## [0.2.0] - 2026-11-22\n\n### Added\n- a\n\n[0.3.0]: https://x\n"
    assert release_notes.section(text, "0.2.0") == "### Added\n- a"
    assert release_notes.section(text, "0.3.0") == "### Added\n- b"
    assert release_notes.section(text, "9.9.9") == ""


def test_release_notes_for_the_current_version_is_not_empty():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "release_notes.py"), version],
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "### Changed" in r.stdout, r.stderr
    assert "[0.2.0]:" not in r.stdout
