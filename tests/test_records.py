import json

from bench.run import main

from trbench import records, roofline

TOP = {
    "schema_version",
    "run_id",
    "dataset",
    "algorithm",
    "query",
    "hardware",
    "metrics",
    "timing",
    "git",
    "created_at",
}
METRICS = {
    "recall",
    "qps",
    "best_search_time_s",
    "search_time_s_all",
    "latency_ms_per_batch",
    "compile_time_s",
    "index_size_bytes",
    "flops_achieved",
    "flops_peak",
    "percent_of_peak",
}
HW = {"platform", "device_kind", "n_devices", "jax_version", "jaxlib_version", "libtpu_version"}


def test_record_keys(tmp_path, capsys):
    main(
        [
            "--algos",
            "jit_topk",
            "--m",
            "4",
            "--n",
            "1000",
            "--d",
            "32",
            "--k",
            "5",
            "--out",
            str(tmp_path),
            "--dump-hlo",
        ]
    )
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


def test_single_peak_table_and_f32_null():
    assert records.peak_flops("TPU v5 lite") == roofline.PEAKS["TPU v5 lite"].bf16_flops
    assert records.peak_flops("cpu") is None
    kw = {
        "dataset": {"n_test": 4, "n_train": 8, "dim": 2},
        "algorithm": {},
        "query": {},
        "hardware": {"device_kind": "TPU v5 lite"},
        "git": {"commit": "abc"},
        "metrics": {"best_search_time_s": 1e-9},
    }
    bf = records.build_record(**kw)["metrics"]
    f32 = records.build_record(**kw, compute_dtype="f32")["metrics"]
    assert bf["percent_of_peak"] is not None
    assert f32["percent_of_peak"] is None and f32["flops_peak"] is None
