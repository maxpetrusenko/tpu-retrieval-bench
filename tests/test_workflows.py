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
    for step in ("pytest -q --cov", "validate_results --strict results/tpu/run3 results/tpu/run3-devtime results/tpu/run4 results/tpu/run4-devtime",
                 "doc_stats --check docs/RESULTS.md", "DRY_RUN=1 bash scripts/reproduce_all.sh",
                 "pip-audit"):
        assert step in ci, step


def test_coverage_threshold_is_set():
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert cfg["tool"]["coverage"]["report"]["fail_under"] >= 80


def _job(text: str, name: str) -> str:
    """Text of one top-level job in a workflow (from `  name:` to the next job)."""
    body = text.split("\njobs:\n")[1]
    parts = re.split(r"^  (?=[a-z][\w-]*:$)", body, flags=re.MULTILINE)
    return next(p for p in parts if p.startswith(f"{name}:"))


def test_release_privileges_sit_on_the_publish_job_only():
    rel = _wf("release.yml")
    assert re.search(r"^on:\n  push:\n    tags: \[\"v\*\"\]", rel, re.MULTILINE)
    assert rel.index("permissions: {}") < rel.index("jobs:")
    build, publish = _job(rel, "build"), _job(rel, "publish")
    for perm in ("contents: write", "id-token: write", "attestations: write"):
        assert rel.count(perm) == 1 and perm in publish and perm not in build, perm
    assert "contents: read" in build
    assert "needs: build" in publish


def test_release_builds_checks_attests_and_attaches_assets():
    rel = _wf("release.yml")
    build, publish = _job(rel, "build"), _job(rel, "publish")
    assert "pyproject.toml" in build and 'v$version' in build
    assert 'scripts/changelog.py release "$VERSION"' in build  # tag-time CHANGELOG gate
    assert "scripts/release_assets.sh" in build and "scripts/release_notes.py" in build
    assert "actions/attest-build-provenance@" in publish and "release/dist/*.whl" in publish
    assert "sha256sum -c" in publish
    assert "release/assets/SHA256SUMS" in publish and "release/assets/sbom.cdx.json" in publish
    assert "release/dist/*.tar.gz" in publish and "--verify-tag" in publish


def test_ci_has_fuzz_and_sast_jobs():
    ci = _wf("ci.yml")
    assert re.search(r"^  fuzz:\n    name: fuzz \(hypothesis\)$", ci, re.MULTILINE)
    assert re.search(r"^  sast:\n    name: sast \(bandit\)$", ci, re.MULTILINE)
    assert "pytest -m fuzz" in ci and "FUZZ_EXAMPLES" in ci
    assert "bandit -c pyproject.toml -ll" in ci
    assert "self-hosted" not in ci.split("\n  test:")[0]


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


def test_release_notes_for_the_current_version_come_from_a_cut_changelog(tmp_path):
    """The real CHANGELOG may still hold the version under Unreleased; cut a copy and extract."""
    version = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]
    copy = tmp_path / "CHANGELOG.md"
    copy.write_text((ROOT / "CHANGELOG.md").read_text())
    if subprocess.run([sys.executable, str(ROOT / "scripts" / "changelog.py"), "release", version,
                       str(copy)], capture_output=True, check=False).returncode:
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "changelog.py"), "cut", version,
                            "2099-01-01", str(copy)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "release_notes.py"), version,
                        str(copy)], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "### " in r.stdout, r.stderr
    assert f"[{version}]:" not in r.stdout
