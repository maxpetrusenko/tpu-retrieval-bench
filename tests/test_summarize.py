import json

from bench import summarize


def _rec(algo, qps, bm=None, recall=1.0, exact=True, peak=3.0):
    params = {} if bm is None else {"bm": bm, "bn": 512, "bk": 128}
    return {
        "dataset": {"name": "synthetic", "n_train": 1000, "dim": 8},
        "algorithm": {"name": algo, "params": params, "exact": exact},
        "query": {"k": 10, "batch_size": 16},
        "hardware": {"platform": "tpu"},
        "metrics": {"compile_time_s": 0.1, "qps": qps, "recall": recall, "percent_of_peak": peak,
                    "roofline_label": "HBM-bound", "selection_over_matmul": None,
                    "latency_ms_per_batch": {"p50": 1.0, "p95": 1.1}},
    }


def _write(tmp_path, recs):
    for i, r in enumerate(recs):
        (tmp_path / f"{i}.json").write_text(json.dumps(r))


def test_render_best_worst_and_flags(tmp_path):
    _write(tmp_path, [_rec("pallas_score_topk", 100.0, bm=16), _rec("pallas_score_topk", 400.0, bm=128),
                      _rec("jit_matmul_topk", 300.0, recall=0.9), _rec("x", 1.0, peak=120.0)])
    out = summarize.render(summarize.load(tmp_path))
    assert "| 128x512x128 | 400 | 16x512x128 | 100 | 4.000 |" in out
    assert "exact variant recall 0.9000 < 1" in out
    assert "percent_of_peak 120.0 > 100" in out
    assert out.count("pallas_score_topk (bm") == 1  # one row per shape: best config only
