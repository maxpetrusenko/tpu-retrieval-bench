import json

from bench.run import main

TOP = {"schema_version", "run_id", "dataset", "algorithm", "query", "hardware", "metrics",
       "timing", "git", "created_at"}
METRICS = {"recall", "qps", "best_search_time_s", "search_time_s_all", "latency_ms_per_batch",
           "compile_time_s", "index_size_bytes", "flops_achieved", "flops_peak", "percent_of_peak"}
HW = {"platform", "device_kind", "n_devices", "jax_version", "jaxlib_version", "libtpu_version"}


def test_record_keys(tmp_path, capsys):
    main(["--algos", "jit_topk", "--m", "4", "--n", "1000", "--d", "32", "--k", "5",
          "--out", str(tmp_path), "--dump-hlo"])
    files = list(tmp_path.rglob("*.json"))
    assert len(files) == 1
    assert "synthetic/k5/jit_topk" in files[0].as_posix()
    rec = json.loads(files[0].read_text())
    assert TOP <= rec.keys()
    assert METRICS <= rec["metrics"].keys()
    assert HW <= rec["hardware"].keys()
    assert {"commit", "dirty"} <= rec["git"].keys()
    assert set(rec["metrics"]["latency_ms_per_batch"]) >= {"min", "p50", "p95", "p99", "mean"}
    assert rec["metrics"]["recall"] >= 0.99
    assert rec["query"]["warmup_iters"] >= 3 and rec["query"]["repetitions"] >= 5
    assert list(tmp_path.rglob("*.hlo.txt"))
