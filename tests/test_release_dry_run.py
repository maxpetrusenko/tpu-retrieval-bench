"""The release dry run builds, checksums and writes the SBOM the way release.yml does."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("uv") is None, reason="uv is needed to build")


def test_release_dry_run_builds_checksums_and_sbom(tmp_path):
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "release_dry_run.sh"), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
        timeout=600,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "release dry run ok" in r.stdout
    dists = sorted(p.name for p in (tmp_path / "dist").iterdir() if not p.name.startswith("."))
    assert len(dists) == 2 and any(d.endswith(".whl") for d in dists)
    sums = (tmp_path / "assets" / "SHA256SUMS").read_text().splitlines()
    assert sorted(line.split("  ")[1] for line in sums) == dists  # names only, sha256sum -c format
    assert all(len(line.split("  ")[0]) == 64 for line in sums)
    bom = json.loads((tmp_path / "assets" / "sbom.cdx.json").read_text())
    assert bom["bomFormat"] == "CycloneDX" and bom["specVersion"] == "1.5"
    assert "jax" in {c["name"] for c in bom["components"]}
    assert (tmp_path / "notes.md").stat().st_size > 0


def test_dry_run_leaves_the_working_tree_alone(tmp_path):
    before = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout
    subprocess.run(
        ["bash", str(ROOT / "scripts" / "release_dry_run.sh"), str(tmp_path)],
        capture_output=True,
        check=True,
        cwd=ROOT,
        timeout=600,
    )
    after = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT, check=True
    ).stdout
    assert before == after


def test_release_assets_refuses_an_incomplete_dist_dir(tmp_path):
    (tmp_path / "only.whl").write_text("x")
    r = subprocess.run(
        ["bash", str(ROOT / "scripts" / "release_assets.sh"), str(tmp_path), str(tmp_path / "out")],
        capture_output=True,
        text=True,
        check=False,
        cwd=ROOT,
    )
    assert r.returncode == 1 and "expected a wheel and an sdist" in r.stderr
