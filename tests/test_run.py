import json

import numpy as np
import pytest
from bench import plot, run
from tests._h5 import make_file


def read_all(root):
    return {p.parent.name: json.loads(p.read_text()) for p in root.rglob("*.json")}


def test_stage_split_nulls_and_ratio(tmp_path):
    run.main(["--algos", "score_only,select_only,jit_topk", "--m", "4", "--n", "2048", "--d", "16",
              "--k", "5", "--out", str(tmp_path)])
    recs = read_all(tmp_path)
    sel, sc, jt = recs["select_only"]["metrics"], recs["score_only"]["metrics"], recs["jit_topk"]
    assert sel["flops_achieved"] is None and sel["percent_of_peak"] is None
    assert sel["selection_time_ms"] > 0 and sel["selection_over_matmul"] > 0
    assert sc["selection_over_matmul"] == sel["selection_over_matmul"]
    assert sc["flops_achieved"] > 0
    for r in recs.values():
        assert {"roofline_label", "bytes_moved", "arithmetic_intensity"} <= set(r["metrics"])
    assert jt["metrics"]["bytes_moved"] > 0


def test_prepared_index_is_untimed_and_stored_in_compute_dtype(tmp_path):
    run.main(["--algos", "jit_topk,blockwise,pallas_score,approx_max_k", "--m", "4", "--n", "600",
              "--d", "16", "--k", "5", "--out", str(tmp_path), "--bn", "128", "--bk", "128"])
    for name, r in read_all(tmp_path).items():
        mt = r["metrics"]
        assert mt["build_time_s"] > 0, name
        if name in ("jit_topk", "approx_max_k"):  # (N, D) bf16 db
            assert mt["index_size_bytes"] == 600 * 16 * 2
        assert mt["recall"] >= 0.99, name


def test_dataset_flag_with_batches(tmp_path):
    make_file(tmp_path / "sift-128-euclidean.hdf5", "euclidean", n=300, nq=6, d=16)
    run.main(["--dataset", "sift-128-euclidean", "--data-dir", str(tmp_path), "--algos",
              "jit_topk,blockwise", "--k", "10", "--dtype", "f32", "--n-train", "250",
              "--n-test", "5", "--batch-size", "2", "--out", str(tmp_path / "out")])
    recs = read_all(tmp_path / "out")
    assert set(recs) == {"jit_topk", "blockwise"}
    for r in recs.values():
        assert r["dataset"]["name"] == "sift-128-euclidean"
        assert (r["dataset"]["n_train"], r["dataset"]["n_test"]) == (250, 5)
        assert r["query"]["batch_size"] == 2 and r["query"]["n_batches"] == 3  # 2 + 2 + 1
        assert r["metrics"]["recall"] == 1.0
        assert r["metrics"]["qps"] == pytest.approx(5 / r["metrics"]["best_search_time_s"])
        assert r["metrics"]["score_matrix_bytes"] == 4 * 2 * 250 or r["algorithm"]["name"] == (
            "blockwise_topk")
    assert "sift-128-euclidean" in str(next((tmp_path / "out").rglob("*.json")))


@pytest.mark.parametrize("dist", ["euclidean", "angular"])
def test_dataset_metrics(tmp_path, dist):
    make_file(tmp_path / "glove-100-angular.hdf5", dist, n=200, nq=4, d=8)
    run.main(["--dataset", "glove-100-angular", "--data-dir", str(tmp_path), "--algos",
              "jit_topk", "--k", "5", "--dtype", "f32", "--out", str(tmp_path / "out")])
    assert read_all(tmp_path / "out")["jit_topk"]["metrics"]["recall"] == 1.0


def test_oom_is_recorded_and_run_continues(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("RESOURCE_EXHAUSTED: out of memory allocating 99GB")

    monkeypatch.setitem(run.FACTORIES, "jit_topk", boom)
    run.main(["--algos", "jit_topk,blockwise", "--m", "4", "--n", "512", "--d", "16",
              "--k", "5", "--out", str(tmp_path)])
    recs = read_all(tmp_path)
    assert "RESOURCE_EXHAUSTED" in recs["jit_topk"]["error"]
    assert recs["blockwise"]["metrics"]["recall"] >= 0.99


def test_non_oom_errors_propagate(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise ValueError("bug")

    monkeypatch.setitem(run.FACTORIES, "jit_topk", boom)
    with pytest.raises(ValueError):
        run.main(["--algos", "jit_topk", "--m", "4", "--n", "512", "--d", "16", "--k", "5",
                  "--out", str(tmp_path)])


def test_synthetic_recall_uses_sampled_queries_when_oracle_too_big(tmp_path, monkeypatch):
    monkeypatch.setattr(run, "REF_LIMIT", 1)
    run.main(["--algos", "jit_topk", "--m", "100", "--n", "512", "--d", "8", "--k", "5",
              "--batch-size", "40", "--out", str(tmp_path)])
    mt = read_all(tmp_path)["jit_topk"]["metrics"]
    assert mt["recall_sampled_queries"] == run.SAMPLE_QUERIES == 64
    assert mt["recall"] >= 0.99


def test_full_oracle_when_small(tmp_path):
    run.main(["--algos", "jit_topk", "--m", "10", "--n", "512", "--d", "8", "--k", "5",
              "--out", str(tmp_path)])
    assert read_all(tmp_path)["jit_topk"]["metrics"]["recall_sampled_queries"] == 10


def test_f32_compute_has_no_percent_of_peak(tmp_path):
    run.main(["--algos", "jit_topk", "--m", "4", "--n", "256", "--d", "8", "--k", "5",
              "--dtype", "f32", "--out", str(tmp_path)])
    mt = read_all(tmp_path)["jit_topk"]["metrics"]
    assert mt["percent_of_peak"] is None and mt["flops_peak"] is None


def test_sweep_blocks_dedupes_after_clamping():
    a = run.parse(["--sweep-blocks"])
    # tiny shape: every bm/bn/bk clamps to the same (16, 128, 128)
    assert run.block_params("pallas_score", a, 4, 100, 8) == [(16, 128, 128)]
    assert len(run.block_params("pallas_score", a, 1024, 1_000_000, 128)) == 3 * 3 * 1


def test_exact_variants_relabelled_when_selection_dominates(tmp_path, monkeypatch):
    monkeypatch.setattr(run.env, "hardware_info", lambda: {
        "platform": "tpu", "device_kind": "TPU v5 lite", "n_devices": 1, "jax_version": "x",
        "jaxlib_version": "x", "libtpu_version": None})
    real = run.timing.time_batched
    calls = iter([1.0, 1.0, 9.0])  # jit_topk, score_only, select_only

    def fake(fn, make_args, nb, **kw):
        return real(fn, make_args, nb, **kw) | {"times_s": [next(calls)] * 5}

    monkeypatch.setattr(run.timing, "time_batched", fake)
    run.main(["--algos", "jit_topk,score_only,select_only", "--m", "4", "--n", "512",
              "--d", "8", "--k", "5", "--out", str(tmp_path)])
    recs = read_all(tmp_path)
    ratio = recs["select_only"]["metrics"]["selection_over_matmul"]
    assert ratio == recs["score_only"]["metrics"]["selection_over_matmul"]
    assert recs["score_only"]["metrics"]["roofline_label"] != "VPU/COP-bound"
    assert ratio == pytest.approx(9.0)
    assert recs["jit_topk"]["metrics"]["roofline_label"] == "VPU/COP-bound"
    assert recs["select_only"]["metrics"]["roofline_label"] == "VPU/COP-bound"


def _rec(platform, algo, n, qps, batch=16, **params):
    return {
        "hardware": {"platform": platform},
        "dataset": {"name": "synthetic", "n_train": n, "dim": 8},
        "query": {"k": 5, "batch_size": batch},
        "algorithm": {"name": algo, "exact": True, "params": {"precision": "bf16", **params}},
        "metrics": {"qps": qps, "recall": 1.0, "percent_of_peak": None,
                    "latency_ms_per_batch": {"p99": 1.0}},
    }


def test_plot_groups_never_mix_platform_or_shape_and_keep_best_pallas():
    recs = [
        _rec("cpu", "pallas_score_topk", 512, 10.0, bm=128),
        _rec("cpu", "pallas_score_topk", 512, 30.0, bm=256),
        _rec("tpu", "pallas_score_topk", 512, 999.0),
        _rec("cpu", "pallas_score_topk", 1024, 5.0),
    ]
    g = plot.groups(recs)
    assert len(g) == 3
    cpu512 = next(v for k, v in g.items() if k[0] == "cpu" and k[2] == 512)
    assert [r["metrics"]["qps"] for r in cpu512] == [30.0]


def test_plot_cli_writes_one_set_per_group_and_filters(tmp_path):
    for plat in ("cpu", "tpu"):
        r = _rec(plat, "jit_matmul_topk", 512, 10.0)
        r["metrics"]["latency_ms_per_batch"] = {"p99": 1.0}
        p = tmp_path / "res" / plat / "x.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(r))
    plot.main(["--results", str(tmp_path / "res"), "--out", str(tmp_path / "img")])
    names = sorted(x.name for x in (tmp_path / "img").glob("*.png"))
    assert len(names) == 6 and any("cpu" in n for n in names) and any("tpu" in n for n in names)
    plot.main(["--results", str(tmp_path / "res"), "--out", str(tmp_path / "img2"),
               "--platform", "tpu"])
    assert not any("cpu" in x.name for x in (tmp_path / "img2").glob("*.png"))
    assert np.all(["tpu" in x.name for x in (tmp_path / "img2").glob("*.png")])
