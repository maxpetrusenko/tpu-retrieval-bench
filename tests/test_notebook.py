import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _code() -> list[str]:
    nb = json.loads((ROOT / "notebooks" / "colab_tpu.ipynb").read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _text() -> str:
    nb = json.loads((ROOT / "notebooks" / "colab_tpu.ipynb").read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"])


def test_notebook_run_cell_calls_reproduce_all_with_run_id():
    runs = [c for c in _code() if "scripts/reproduce_all.sh" in c]
    assert len(runs) == 1
    assert "reproduce_all.sh {RUN_ID}" in runs[0]
    assert "PYTHON=python3" in runs[0]
    assert "colab_run" not in _text()
    setup = next(c for c in _code() if re.search(r"^RUN_ID\s*=", c, re.MULTILINE))
    assert "results/tpu/run-" in setup


def test_notebook_checks_out_a_pinned_ref():
    setup = next(c for c in _code() if re.search(r"^REF\s*=", c, re.MULTILINE))
    assert re.search(r'^REF = "v0\.2\.0"', setup, re.MULTILINE)
    assert "bump" in setup.lower()
    assert any("git checkout" in c and "{REF}" in c for c in _code())


def test_notebook_installs_from_constraints_and_locked_jax():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    jax = next(p["version"] for p in lock["package"] if p["name"] == "jax")
    install = next(c for c in _code() if "!pip install" in c)
    assert "-c notebooks/constraints.txt" in install
    assert f'"jax[tpu]=={jax}"' in install
    assert (ROOT / "notebooks" / "constraints.txt").is_file()


def test_notebook_text_does_not_name_a_stale_run():
    assert "Run-2 grid" not in _text()
