import json
import shutil
import subprocess
from pathlib import Path

import pytest
from bench.smoke_compare import compare, main

ROOT = Path(__file__).resolve().parents[1]
ALGOS = ["jit_topk", "blockwise"]


def _put(root: Path, algo: str, qps: float) -> None:
    p = root / "tpu" / "synthetic" / "k10" / algo / "m128_n65536.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"metrics": {"qps": qps}}))


def _tree(tmp_path, smoke_qps, ref_qps=None):
    smoke, ref = tmp_path / "smoke", tmp_path / "ref"
    for a, q in zip(ALGOS, smoke_qps):
        if q is not None:
            _put(smoke, a, q)
    for a in ALGOS:
        _put(ref, a, ref_qps or 1000.0)
    return smoke, ref


def test_smoke_compare_ok_within_tolerance(tmp_path):
    smoke, ref = _tree(tmp_path, [900.0, 1200.0])
    errors, warnings = compare(smoke, ref, ALGOS, 0.5)
    assert errors == [] and warnings == []


def test_smoke_compare_fails_on_missing_algo(tmp_path):
    smoke, ref = _tree(tmp_path, [900.0, None])
    errors, _ = compare(smoke, ref, ALGOS, 0.5)
    assert errors == ["missing algorithm: no record for blockwise"]


def test_smoke_compare_warns_on_large_drift_without_failing(tmp_path, capsys):
    smoke, ref = _tree(tmp_path, [100.0, 5000.0])
    errors, warnings = compare(smoke, ref, ALGOS, 0.5)
    assert errors == [] and len(warnings) == 2 and all("large drift" in w for w in warnings)
    assert main(["--smoke", str(smoke), "--ref", str(ref), "--algos", ",".join(ALGOS)]) == 0
    assert "large drift" in capsys.readouterr().err


def test_smoke_compare_unreadable_record_fails(tmp_path):
    smoke, ref = _tree(tmp_path, [900.0, 900.0])
    (smoke / "tpu" / "synthetic" / "k10" / "jit_topk" / "m128_n65536.json").write_text("{")
    errors, _ = compare(smoke, ref, ALGOS, 0.5)
    assert any("unreadable record" in e for e in errors)


def test_smoke_script_dry_run_covers_every_algo():
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("needs bash")
    r = subprocess.run([bash, str(ROOT / "scripts" / "tpu_smoke.sh"), "/nonexistent/out"],
                       capture_output=True, text=True, cwd=ROOT, check=False,
                       env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin", "PYTHON": "python3"})
    assert r.returncode == 0, r.stderr
    cmds = [ln for ln in r.stdout.splitlines() if ln.startswith("+ ")]
    for algo in ("jit_topk", "blockwise", "pallas_score", "pallas_fused_topk",
                 "pallas_fused_bucket", "approx_max_k", "score_only", "select_only"):
        assert any(f"--algos {algo} " in c and "--n 65536" in c for c in cmds), algo
    assert any("bench.validate_results" in c for c in cmds)
    assert any("bench.smoke_compare" in c and "results/tpu/run3" in c for c in cmds)


def test_smoke_reference_cell_is_committed():
    # The smoke shape must exist in Run 3 for every algorithm, or the comparison is empty.
    base = ROOT / "results" / "tpu" / "run3" / "tpu" / "synthetic" / "k10"
    for algo in ("jit_topk", "blockwise", "pallas_score", "pallas_fused_topk",
                 "pallas_fused_bucket", "approx_max_k", "score_only", "select_only"):
        assert list((base / algo).glob("m128_n65536_*.json")), algo
