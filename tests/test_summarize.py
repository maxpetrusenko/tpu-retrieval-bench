import json

from bench import summarize


def _rec(algo, qps, bm=None, recall=1.0, exact=True, peak=3.0):
    params = {} if bm is None else {"bm": bm, "bn": 512, "bk": 128}
    return {
        "dataset": {"name": "synthetic", "n_train": 1000, "dim": 8},
        "algorithm": {"name": algo, "params": params, "exact": exact},
        "query": {"k": 10, "batch_size": 16},
        "hardware": {"platform": "tpu"},
        "metrics": {
            "compile_time_s": 0.1,
            "qps": qps,
            "recall": recall,
            "percent_of_peak": peak,
            "roofline_label": "HBM-bound",
            "selection_over_matmul": None,
            "latency_ms_per_batch": {"p50": 1.0, "p95": 1.1},
        },
    }


def _write(tmp_path, recs):
    for i, r in enumerate(recs):
        (tmp_path / f"{i}.json").write_text(json.dumps(r))


def test_render_best_worst_and_flags(tmp_path):
    _write(
        tmp_path,
        [
            _rec("pallas_score_topk", 100.0, bm=16),
            _rec("pallas_score_topk", 400.0, bm=128),
            _rec("jit_matmul_topk", 300.0, recall=0.9),
            _rec("x", 1.0, peak=120.0),
        ],
    )
    out = summarize.render(summarize.load(tmp_path))
    assert "| 128x512x128 | 400 | 16x512x128 | 100 | 4.000 |" in out
    assert "exact variant recall 0.9000 < 1" in out
    assert "percent_of_peak 120.0 > 100" in out
    assert out.count("pallas_score_topk (bm") == 1  # one row per shape: best config only


def test_fast_p_and_reward_hack_flag(tmp_path):
    recs = [
        _rec("jit_matmul_topk", 100.0),
        _rec("pallas_fused_topk", 130.0),
        _rec("pallas_score_topk", 110.0),
        _rec("blockwise_topk", 2000.0),
        _rec("approx_max_k", 900.0, exact=False),
    ]
    recs[2]["metrics"]["recall"] = 0.5  # wrong: never counts as fast
    _write(tmp_path, recs)
    out = summarize.render(summarize.load(tmp_path))
    assert "| pallas_fused_topk | 1 | 1.000 | 1.000 |" in out
    assert "| pallas_score_topk | 1 | 0.000 | 0.000 |" in out
    assert "approx_max_k |" not in out.split("## fast_p")[1].split("## Flags")[0]
    assert "exact variant speedup 20.0x" in out


def _ci(rec, lo, hi):
    rec["metrics"]["qps_ci95"] = [lo, hi]
    return rec


def test_within_noise_flag_and_fast_p_exclusion(tmp_path):
    recs = [
        _ci(_rec("jit_matmul_topk", 100.0), 95.0, 105.0),
        _ci(_rec("pallas_fused_topk", 130.0), 100.0, 140.0),  # overlaps baseline CI
        _ci(_rec("blockwise_topk", 150.0), 140.0, 160.0),
    ]  # disjoint: significant
    _write(tmp_path, recs)
    out = summarize.render(summarize.load(tmp_path))
    assert "vs jit_topk" in out
    fused = next(ln for ln in out.splitlines() if ln.startswith("| pallas_fused_topk"))
    block = next(ln for ln in out.splitlines() if ln.startswith("| blockwise_topk"))
    assert fused.endswith("| 1.30x (within noise) |")
    assert block.endswith("| 1.50x |")
    assert "| pallas_fused_topk | 1 | 0.000 | 0.000 |" in out  # noise-level win is not fast
    assert "| blockwise_topk | 1 | 1.000 | 1.000 |" in out


def test_no_ci_flag_for_old_records(tmp_path):
    _write(tmp_path, [_rec("jit_matmul_topk", 100.0), _rec("pallas_fused_topk", 130.0)])
    out = summarize.render(summarize.load(tmp_path))
    assert "1.30x (no CI)" in out
    assert "| pallas_fused_topk | 1 | 1.000 | 1.000 |" in out  # old records keep old fast_p


def test_ci_overlap_edges():
    a, b = _ci(_rec("x", 1.0), 1.0, 2.0), _ci(_rec("y", 1.0), 2.0, 3.0)
    assert summarize.ci_overlap(a, b) is True  # touching counts as overlap
    assert summarize.ci_overlap(a, _ci(_rec("z", 1.0), 2.1, 3.0)) is False
    assert summarize.ci_overlap(a, _rec("w", 1.0)) is None


def _v2(rec, best, ci=None):
    """Make rec a schema-0.2 record: its `qps` is the median pass, `qps_best` the fastest."""
    rec["schema_version"] = "0.2"
    rec["metrics"]["qps_best"] = best
    if ci:
        rec["metrics"]["qps_ci95"] = ci
    return rec


def test_median_basis_when_every_record_is_schema_02(tmp_path):
    _write(
        tmp_path,
        [
            _v2(_rec("jit_matmul_topk", 100.0), 120.0, [95.0, 105.0]),
            _v2(_rec("blockwise_topk", 150.0), 200.0, [140.0, 160.0]),
        ],
    )
    out = summarize.render(summarize.load(tmp_path))
    assert "QPS = median pass (schema 0.2)" in out
    assert next(ln for ln in out.splitlines() if ln.startswith("| blockwise_topk")).endswith(
        "| 1.50x |"
    )  # median 150 / median 100, not best 200 / 120


def test_mixed_schemas_compare_on_best_pass_and_say_so(tmp_path):
    old = _rec("jit_matmul_topk", 100.0)  # schema-less = 0.1: qps is the best pass
    new = _v2(_rec("blockwise_topk", 90.0), 150.0, [85.0, 95.0])  # median 90 < old best 100
    _write(tmp_path, [old, new])
    out = summarize.render(summarize.load(tmp_path))
    assert "QPS = best pass; table mixes schema 0.1" in out
    row = next(ln for ln in out.splitlines() if ln.startswith("| blockwise_topk"))
    assert row.endswith("| 1.50x (mixed basis) |")  # best/best, never 90/100 = 0.90x
    assert "0.90x" not in out


def test_basis_helpers():
    from trbench import records

    old, new = _rec("x", 10.0), _v2(_rec("y", 8.0), 12.0)
    assert (records.qps_basis(old), records.qps_basis(new)) == ("best", "median")
    assert (records.qps_best(old), records.qps_best(new)) == (10.0, 12.0)
    assert records.comparison_basis([new, new]) == "median"
    assert records.comparison_basis([old, new]) == "best"
    assert records.qps_on(new, "best") == 12.0 and records.qps_on(new, "median") == 8.0
