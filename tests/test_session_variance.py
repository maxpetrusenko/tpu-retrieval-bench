import json
from pathlib import Path

import pytest
from bench.session_variance import (
    claims,
    join,
    load_run,
    main,
    overlaps,
    percentile,
    size_class,
    summarize,
)

ROOT = Path(__file__).resolve().parents[1]


def _rec(m, qps, ci=None, algo="jit_matmul_topk", n=1000, params=None):
    ci = ci or [qps * 0.99, qps * 1.01]
    return {"dataset": {"name": "synthetic", "n_train": n, "dim": 8},
            "query": {"k": 10, "batch_size": m},
            "algorithm": {"name": algo, "params": params or {}},
            "metrics": {"qps": qps, "qps_ci95": ci}}


def _put(root: Path, name: str, rec) -> None:
    p = root / "tpu" / "synthetic" / "k10" / "x" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec))


def test_percentile_linear_interpolation():
    assert percentile([1.0], 90) == 1.0
    assert percentile([0.0, 10.0], 50) == 5.0
    assert percentile([0.0, 10.0, 20.0, 30.0, 40.0], 90) == pytest.approx(36.0)


def test_size_class_boundaries():
    assert [size_class(m) for m in (1, 16, 17, 127, 128, 1024)] == [
        "small", "small", "mid", "mid", "large", "large"]


def test_overlaps_touching_counts():
    assert overlaps([1, 2], [2, 3]) and not overlaps([1, 2], [2.1, 3])


def test_join_ratio_delta_and_overlap(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _put(a, "m1.json", _rec(1, 100.0, [95.0, 105.0]))
    _put(b, "m1.json", _rec(1, 130.0, [125.0, 135.0]))
    _put(a, "m128.json", _rec(128, 1000.0, [990.0, 1010.0]))
    _put(b, "m128.json", _rec(128, 1005.0, [995.0, 1015.0]))
    rows = join(load_run(a), load_run(b))
    by_m = {r["m"]: r for r in rows}
    assert by_m[1]["ratio"] == pytest.approx(1.3)
    assert by_m[1]["abs_pct_delta"] == pytest.approx(30.0) and not by_m[1]["ci_overlap"]
    assert by_m[128]["ci_overlap"] and by_m[128]["abs_pct_delta"] == pytest.approx(0.5)
    s = summarize(rows)
    assert s["small"]["max"] == pytest.approx(30.0)
    assert s["large"]["median"] == pytest.approx(0.5) and s["large"]["ci_overlap"] == 1
    assert s["all"]["cells"] == 2 and "mid" not in s


def test_summary_splits_large_by_n(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _put(a, "1.json", _rec(128, 100.0, n=1_000_000))
    _put(b, "1.json", _rec(128, 101.0, n=1_000_000))
    _put(a, "2.json", _rec(128, 100.0, n=65_536))
    _put(b, "2.json", _rec(128, 120.0, n=65_536))
    s = summarize(join(load_run(a), load_run(b)))
    assert s["large, N>=1M"]["max"] == pytest.approx(1.0)
    assert s["large, N<1M"]["max"] == pytest.approx(20.0)


def test_sweep_record_does_not_collide_with_default(tmp_path):
    a = tmp_path / "a"
    p = {"bm": 128, "bn": 512}
    _put(a, "m128.json", _rec(128, 100.0, algo="pallas_fused_topk", params=p))
    _put(a, "m128_sweep.json", _rec(128, 110.0, algo="pallas_fused_topk", params=p))
    assert len(load_run(a)) == 2


def test_duplicate_cell_is_an_error(tmp_path):
    a = tmp_path / "a"
    _put(a, "m1.json", _rec(1, 100.0))
    p = a / "tpu" / "synthetic" / "k10" / "y" / "m1.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(_rec(1, 100.0)))
    with pytest.raises(ValueError, match="duplicate cell"):
        load_run(a)


def test_missing_ci_is_an_error(tmp_path):
    a = tmp_path / "a"
    rec = _rec(1, 100.0)
    del rec["metrics"]["qps_ci95"]
    _put(a, "m1.json", rec)
    with pytest.raises(ValueError, match="qps_ci95"):
        load_run(a)


def _claim_tree(root: Path, jit: float, v4: float, v4_ci_halfwidth: float = 0.01) -> None:
    _put(root, "jit.json", _rec(1024, jit))
    _put(root, "v4.json", _rec(1024, v4, [v4 * (1 - v4_ci_halfwidth), v4 * (1 + v4_ci_halfwidth)],
                               algo="pallas_fused_topk", params={"bm": 256, "bn": 512}))
    _put(root, "v4_sweep.json", _rec(1024, v4 * 1.2, algo="pallas_fused_topk",
                                     params={"bm": 256, "bn": 1024}))


def _verdicts(tmp_path, a_args, b_args):
    a, b = tmp_path / "a", tmp_path / "b"
    _claim_tree(a, *a_args)
    _claim_tree(b, *b_args)
    out = claims(load_run(a), load_run(b), "pallas_fused_topk", "jit_matmul_topk")
    return {c["kind"]: c for c in out}


def test_claim_robust_win_and_loss(tmp_path):
    v = _verdicts(tmp_path, (100.0, 120.0), (110.0, 130.0))
    assert v["default"]["status"] == "robust win" and v["tuned"]["status"] == "robust win"
    assert v["tuned"]["cfg_a"] == "256x1024"
    v = _verdicts(tmp_path, (100.0, 50.0), (100.0, 60.0))
    assert v["default"]["status"] == "robust loss"


def test_claim_fragile_on_tie_or_flip(tmp_path):
    v = _verdicts(tmp_path, (100.0, 120.0), (100.0, 100.5))
    assert v["default"]["verdict_b"] == "tie" and v["default"]["status"] == "fragile"
    v = _verdicts(tmp_path, (100.0, 120.0), (100.0, 80.0))
    assert (v["default"]["verdict_a"], v["default"]["verdict_b"]) == ("win", "loss")
    assert v["default"]["status"] == "fragile"


def test_claim_skips_shape_without_baseline(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for r in (a, b):
        _put(r, "v4.json", _rec(1024, 100.0, algo="pallas_fused_topk", params={"bm": 1, "bn": 2}))
    assert claims(load_run(a), load_run(b), "pallas_fused_topk", "jit_matmul_topk") == []


def test_main_writes_outputs_and_rejects_mismatched_runs(tmp_path, capsys):
    a, b = tmp_path / "a", tmp_path / "b"
    _claim_tree(a, 100.0, 120.0)
    _claim_tree(b, 100.0, 121.0)
    j, md = tmp_path / "o.json", tmp_path / "o.md"
    assert main([str(a), str(b), "--json", str(j), "--md", str(md)]) == 0
    data = json.loads(j.read_text())
    assert data["run_a"] == "a" and len(data["cells"]) == 3 and data["claims"]
    assert "robust win" in md.read_text()
    _put(b, "extra.json", _rec(7, 5.0))
    assert main([str(a), str(b)]) == 1
    assert "do not cover the same cells" in capsys.readouterr().err
    assert main([str(a), str(tmp_path / "missing")]) == 1


def test_committed_runs_have_one_to_one_join():
    run3, run4 = ROOT / "results/tpu/run3", ROOT / "results/tpu/run4"
    if not run4.is_dir():
        pytest.skip("run4 not committed")
    a, b = load_run(run3), load_run(run4)
    assert set(a) == set(b) and len(a) == 111
    clm = claims(a, b, "pallas_fused_topk", "jit_matmul_topk")
    assert {c["status"] for c in clm} <= {"robust win", "robust loss", "fragile"}
