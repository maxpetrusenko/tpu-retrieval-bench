import re
import shutil
import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONSTRAINTS = ROOT / "notebooks" / "constraints.txt"


def _pins() -> dict[str, str]:
    pins = {}
    for line in CONSTRAINTS.read_text().splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if m:
            pins[m.group(1).lower().replace("_", "-")] = m.group(2)
    return pins


def test_constraints_match_uv_lock():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = {p["name"]: p["version"] for p in lock["package"]}
    pins = _pins()
    assert pins, "constraints file has no pins"
    for name, version in pins.items():
        assert locked.get(name) == version, f"{name}: {version} vs lock {locked.get(name)}"
    for core in ("jax", "jaxlib", "numpy"):
        assert core in pins
    assert "pytest" not in pins and "trbench" not in pins


def test_constraints_regenerate_cleanly():
    bash, uv = shutil.which("bash"), shutil.which("uv")
    if bash is None or uv is None:
        pytest.skip("needs bash and uv")
    r = subprocess.run(
        [bash, str(ROOT / "scripts" / "export_constraints.sh"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    assert r.returncode == 0, r.stderr
