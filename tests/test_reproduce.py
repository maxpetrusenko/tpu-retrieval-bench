import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "reproduce_all.sh"
bash = shutil.which("bash")
pytestmark = pytest.mark.skipif(bash is None, reason="needs bash")


def dry(*args, env=None):
    import os
    return subprocess.run([bash, str(SCRIPT), *args], capture_output=True, text=True, cwd=ROOT,
                          check=False, env={**os.environ, "DRY_RUN": "1", **(env or {})})


def test_script_syntax():
    subprocess.run([bash, "-n", str(SCRIPT)], check=True)


def test_dry_run_covers_full_grid_in_order(tmp_path):
    r = dry(str(tmp_path / "out"))
    assert r.returncode == 0, r.stderr
    cmds = [ln[2:] for ln in r.stdout.splitlines() if ln.startswith("+ ")]
    text = "\n".join(cmds)
    out, dev = str(tmp_path / "out"), str(tmp_path / "out") + "-devtime"
    assert "--m 1,16,128,1024 --n 65536,1000000" in text  # main grid
    assert "bench.download sift" in text and "--dataset sift-128-euclidean" in text
    assert text.count("--sweep-blocks") == 1  # block sweep
    for target in ("0.80", "0.90", "0.95", "0.99"):  # recall sweep
        assert f"--recall-target {target}" in text
    devline = next(c for c in cmds if "--device-time" in c)
    assert f"--out {dev}" in devline
    assert "--reps" not in text  # new defaults (20 reps), never overridden here
    assert all(f"--out {out}" in c or f"--out {dev}" in c for c in cmds if "bench.run" in c)
    tail = [c for c in cmds if "bench.run" not in c and "download" not in c]
    assert "validate_results --strict" in tail[0] and "bench.summarize" in tail[1]
    assert cmds.index(next(c for c in cmds if "validate_results" in c)) > cmds.index(devline)


def test_allow_dirty_passthrough(tmp_path):
    assert "--allow-dirty" not in dry(str(tmp_path / "o")).stdout
    assert "--allow-dirty" in dry(str(tmp_path / "o"), env={"ALLOW_DIRTY": "1"}).stdout


def test_refuses_non_empty_out(tmp_path):
    import os
    (tmp_path / "x.json").write_text("{}")
    r = subprocess.run([bash, str(SCRIPT), str(tmp_path)], capture_output=True, text=True,
                       check=False, cwd=ROOT, env={k: v for k, v in os.environ.items() if k != "DRY_RUN"})
    assert r.returncode == 1 and "refusing" in r.stderr


def test_notebook_uses_single_entry_point_and_keeps_tpu_lock():
    nb = json.loads((ROOT / "notebooks" / "colab_tpu.ipynb").read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert sum("scripts/reproduce_all.sh" in c for c in code) == 1
    assert not any("bench/run.py" in c or "bench.run" in c for c in code)
    for c in code:  # TPU lock: jax is only imported inside `!python`, never in the kernel
        kernel = [ln for ln in c.splitlines() if not ln.lstrip().startswith(("!", "#"))]
        assert not any(ln.lstrip().startswith(("import jax", "from jax")) for ln in kernel)


def _first_cmd(env):
    """First command line a dry run prints, with PATH limited to the given directory."""
    r = subprocess.run([bash, str(SCRIPT), "/nonexistent/out"], capture_output=True, text=True,
                       cwd=ROOT, check=False, env={"DRY_RUN": "1", **env})
    assert r.returncode == 0, r.stderr
    return next(ln for ln in r.stdout.splitlines() if ln.startswith("+ "))


def _path_with(tmp_path, *tools):
    """A PATH holding only the shell utilities the script needs plus stub `tools`."""
    d = tmp_path / "bin"
    d.mkdir()
    for util in ("dirname", "date", "ls"):
        (d / util).symlink_to(shutil.which(util))
    for tool in tools:
        stub = d / tool
        stub.write_text("#!/bin/sh\nexit 0\n")
        stub.chmod(0o755)
    return str(d)


def test_interpreter_prefers_PYTHON_env(tmp_path):
    path = _path_with(tmp_path, "uv", "python3")
    assert _first_cmd({"PATH": path, "PYTHON": "mypy-python"}).startswith("+ mypy-python -m bench.run")


def test_interpreter_uses_uv_run_when_available(tmp_path):
    assert _first_cmd({"PATH": _path_with(tmp_path, "uv", "python3")}).startswith(
        "+ uv run python -m bench.run")


def test_interpreter_falls_back_to_python3_then_python(tmp_path):
    assert _first_cmd({"PATH": _path_with(tmp_path, "python3")}).startswith("+ python3 -m bench.run")
    other = tmp_path / "other"
    other.mkdir()
    assert _first_cmd({"PATH": _path_with(other)}).startswith("+ python -m bench.run")
