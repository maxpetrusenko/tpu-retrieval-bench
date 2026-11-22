import copy
import json
from pathlib import Path

import pytest
from bench.validate_results import main

from trbench.validate import validate_record, validate_tree

GOOD = {
    "schema_version": "0.1",
    "run_id": "2026-11-18T13:31:44Z-52551c9",
    "dataset": {"name": "synthetic", "n_train": 65536, "n_test": 128, "dim": 128,
                "distance": "ip", "dtype_compute": "bf16"},
    "algorithm": {"name": "jit_matmul_topk", "params": {}, "exact": True},
    "query": {"k": 10, "batch_size": 128, "mode": "batch", "n_queries_timed": 128},
    "hardware": {"platform": "tpu", "device_kind": "TPU v5 lite", "n_devices": 1},
    "metrics": {
        "recall": 1.0, "qps": 128 / 0.0007, "best_search_time_s": 0.0007,
        "search_time_s_all": [0.00072, 0.0007, 0.00071],
        "latency_ms_per_batch": {"min": 0.7, "p50": 0.71, "p95": 0.719, "p99": 0.7198},
        "compile_time_s": 0.2, "index_size_bytes": 1, "flops_achieved": 3.0e12,
        "flops_peak": 1.97e14, "percent_of_peak": 1.52,
    },
    "timing": {}, "git": {}, "created_at": "2026-11-18T13:31:44Z",
}


def mutated(**paths):
    rec = copy.deepcopy(GOOD)
    for path, val in paths.items():
        *parents, leaf = path.split("__")
        obj = rec
        for p in parents:
            obj = obj[p]
        if val is KeyError:
            del obj[leaf]
        else:
            obj[leaf] = val
    return rec


def test_good_record_passes():
    assert validate_record(GOOD) == []


def test_stage_only_and_approx_records_pass():
    assert validate_record(mutated(metrics__recall=None, algorithm__name="score_only")) == []
    approx = mutated(algorithm__exact=False, metrics__recall=0.93)
    assert validate_record(approx) == []
    no_peak = mutated(metrics__flops_achieved=None, metrics__flops_peak=None,
                      metrics__percent_of_peak=None)
    assert validate_record(no_peak) == []


BAD = {
    "peak_over_100": (mutated(metrics__percent_of_peak=100.5), "percent_of_peak"),
    "peak_negative": (mutated(metrics__percent_of_peak=-1.0), "percent_of_peak"),
    "flops_above_peak": (mutated(metrics__flops_achieved=2.0e14), "flops_achieved"),
    "recall_above_1": (mutated(metrics__recall=1.2), "recall"),
    "recall_negative": (mutated(metrics__recall=-0.1), "recall"),
    "exact_recall_low": (mutated(metrics__recall=0.9), "exact variant"),
    "exact_recall_null": (mutated(metrics__recall=None), "null for an exact"),
    "exact_f32_recall_low": (
        mutated(metrics__recall=0.999, dataset__dtype_compute="f32",
                **{"metrics__recall_epsilon_0.01": 1.0}), "exact variant"),
    "qps_zero": (mutated(metrics__qps=0.0), "qps"),
    "qps_negative": (mutated(metrics__qps=-5.0), "qps"),
    "qps_nan": (mutated(metrics__qps=float("nan")), "qps"),
    "qps_inconsistent": (mutated(metrics__qps=5e9), "disagrees"),
    "best_time_zero": (mutated(metrics__best_search_time_s=0.0), "best_search_time_s"),
    "best_not_min": (mutated(metrics__best_search_time_s=0.00071), "min(search_time_s_all)"),
    "latency_not_monotone": (
        mutated(metrics__latency_ms_per_batch={"min": 0.7, "p50": 0.9, "p95": 0.8, "p99": 1.0}),
        "monotone"),
    "latency_min_over_p50": (
        mutated(metrics__latency_ms_per_batch={"min": 0.8, "p50": 0.7, "p95": 0.8, "p99": 0.9}),
        "monotone"),
    "latency_missing_p99": (
        mutated(metrics__latency_ms_per_batch={"min": 0.7, "p50": 0.71, "p95": 0.72}), "p99"),
    "latency_vs_best": (
        mutated(metrics__latency_ms_per_batch={"min": 0.1, "p50": 0.2, "p95": 0.3, "p99": 0.4}),
        "disagrees with best_search_time_s"),
    "missing_top_key": (mutated(git=KeyError), "missing key 'git'"),
    "missing_metric_key": (mutated(metrics__qps=KeyError), "missing key 'qps'"),
    "missing_dataset_key": (mutated(dataset__dim=KeyError), "missing key 'dim'"),
    "missing_exact": (mutated(algorithm__exact=KeyError), "missing key 'exact'"),
    "not_an_object": ([1, 2], "expected object"),
}


@pytest.mark.parametrize("name", sorted(BAD))
def test_bad_record_flagged(name):
    rec, needle = BAD[name]
    errs = validate_record(rec)
    assert errs, "record should be rejected"
    assert any(needle in e for e in errs), errs


def test_bf16_exact_may_use_eps_001_recall():
    rec = mutated(metrics__recall=0.97, **{"metrics__recall_epsilon_0.01": 1.0})
    assert validate_record(rec) == []


def test_validate_tree_and_cli(tmp_path, capsys):
    (tmp_path / "ok.json").write_text(json.dumps(GOOD))
    assert validate_tree(tmp_path) == {}
    assert main([str(tmp_path)]) == 0
    bad = tmp_path / "sub" / "bad.json"
    bad.parent.mkdir()
    bad.write_text(json.dumps(mutated(metrics__percent_of_peak=250.0)))
    (tmp_path / "broken.json").write_text("{not json")
    found = validate_tree(tmp_path)
    assert set(found) == {str(bad), str(tmp_path / "broken.json")}
    assert main([str(tmp_path)]) == 1
    assert "percent_of_peak" in capsys.readouterr().out


def test_cli_empty_dir_fails_unless_allowed(tmp_path):
    assert main([str(tmp_path)]) == 1
    assert main([str(tmp_path), "--allow-empty"]) == 0


def test_committed_tpu_results_are_plausible():
    root = Path(__file__).resolve().parents[1] / "results" / "tpu"
    if not root.exists():
        pytest.skip("no committed results/tpu")
    assert validate_tree(root) == {}


V2_TIMES = [0.00072, 0.0007, 0.00071, 0.00073, 0.00075]
SPREAD = {
    "n_repetitions": 5, "latency_stdev_ms": 0.01, "latency_p50_ci95_ms": [0.70, 0.72],
    "qps_median": 128 / 0.00072, "qps_stdev": 1000.0, "qps_ci95": [170000.0, 183000.0],
    "qps_best": 128 / 0.0007,
}
SPREAD_KEYS = tuple(SPREAD)


def v2(**paths):
    rec = mutated(schema_version="0.2")
    rec["metrics"].update(SPREAD)
    rec["metrics"]["qps"] = SPREAD["qps_median"]  # 0.2+: headline is the median pass
    rec["metrics"]["search_time_s_all"] = V2_TIMES
    for path, val in paths.items():
        *parents, leaf = path.split("__")
        obj = rec
        for p in parents:
            obj = obj[p]
        obj[leaf] = val
    return rec


def test_spread_required_from_schema_02_but_not_for_01():
    assert validate_record(GOOD) == []  # old record, no spread: still valid
    assert validate_record(mutated(schema_version="0.2"))  # new record without spread: invalid
    for key in SPREAD_KEYS:
        rec = v2()
        del rec["metrics"][key]
        assert any(key in e for e in validate_record(rec)), key
    assert validate_record(v2()) == []


@pytest.mark.parametrize("path,val", [
    ("metrics__qps_ci95", [2.0, 1.0]),
    ("metrics__qps_ci95", [1.0]),
    ("metrics__latency_p50_ci95_ms", [-1.0, 1.0]),
    ("metrics__qps_stdev", -1.0),
    ("metrics__n_repetitions", 4),
    ("metrics__n_repetitions", 6),  # differs from len(search_time_s_all)
    ("metrics__qps_median", 0),
])
def test_bad_spread_flagged(path, val):
    assert validate_record(v2(**{path: val}))


def test_dirty_warning_count_and_strict(tmp_path, capsys):
    (tmp_path / "a.json").write_text(json.dumps(mutated(git={"commit": "a", "dirty": False})))
    (tmp_path / "b.json").write_text(json.dumps(mutated(git={"commit": "a", "dirty": True})))
    assert main([str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "warning: 1 of 2 records have git.dirty=true" in out
    assert main([str(tmp_path), "--strict"]) == 1
    out = capsys.readouterr().out
    assert "b.json" in out and "a.json" not in out and "warning:" not in out


def test_headline_qps_is_median_pass_with_its_ci_from_schema_02():
    assert validate_record(v2()) == []
    best_as_headline = v2(metrics__qps=128 / 0.0007)  # the old min-time headline
    assert any("equal qps_median" in e for e in validate_record(best_as_headline))
    outside = v2(metrics__qps_ci95=[178000.0, 183000.0])  # headline below its own CI
    assert any("outside its own qps_ci95" in e for e in validate_record(outside))
    wrong_best = v2(metrics__qps_best=1.0)
    assert any("qps_best" in e for e in validate_record(wrong_best))
    fake_median = v2(metrics__qps_median=181000.0, metrics__qps=181000.0)
    assert any("median of" in e for e in validate_record(fake_median))


def test_schema_01_keeps_best_pass_headline_rule():
    assert validate_record(GOOD) == []  # qps = n / best time, no qps_best needed
    assert any("disagrees" in e for e in validate_record(mutated(metrics__qps=5e9)))
