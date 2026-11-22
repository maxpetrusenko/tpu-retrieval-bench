"""Sweep planning and the pieces of the split bench.run driver that main() does not reach."""

import numpy as np
import pytest
from bench import emit, plan
from bench.algos import FACTORIES, build
from bench.cli import parse
from bench.execute import Measurement

from trbench import timing


def test_plan_jobs_order_is_algo_k_shape_block():
    a = parse(["--algos", "jit_topk,pallas_score", "--k", "5,10", "--m", "4,8", "--n", "256",
               "--d", "8", "--bn", "128", "--bk", "128"])
    jobs = list(plan.plan_jobs(a, None))
    assert [(j.algo, j.k, j.m) for j in jobs] == [
        ("jit_topk", 5, 4), ("jit_topk", 5, 8), ("jit_topk", 10, 4), ("jit_topk", 10, 8),
        ("pallas_score", 5, 4), ("pallas_score", 5, 8), ("pallas_score", 10, 4),
        ("pallas_score", 10, 8)]
    assert all(j.bp is None for j in jobs if j.algo == "jit_topk")
    assert all(j.bp is not None for j in jobs if j.algo == "pallas_score")


def test_plan_jobs_dataset_run_takes_shape_from_file():
    ann = {"test": np.zeros((6, 4)), "train": np.zeros((50, 4))}
    a = parse(["--algos", "jit_topk", "--dataset", "sift-128-euclidean"])
    (job,) = plan.plan_jobs(a, ann)
    assert (job.m, job.n, job.d) == (6, 50, 4)


@pytest.mark.parametrize("argv,algo,bp,expect", [
    ([], "jit_topk", None, "m4_n256_d8_ip_bf16"),
    (["--block-size", "128"], "blockwise", None, "m4_n256_d8_ip_bf16_B128"),
    (["--recall-target", "0.9"], "approx_max_k", None, "m4_n256_d8_ip_bf16_r0.9"),
    ([], "pallas_score", (16, 128, 128), "m4_n256_d8_ip_bf16_bm16_bn128_bk128"),
    (["--sweep-blocks"], "pallas_fused_topk", (16, 128, 128),
     "m4_n256_d8_ip_bf16_bm16_bn128_bk128_sweep"),
])
def test_params_str_names_each_algorithms_knobs(argv, algo, bp, expect):
    assert plan.params_str(algo, 4, 256, 8, parse(argv), bp) == expect


def test_build_covers_every_registered_algo_and_fused_gets_db_size():
    a = parse(["--bn", "128", "--bk", "128"])
    for algo in FACTORIES:
        prep, search, name, exact, params = build(algo, 5, np.float32, a, (16, 128, 128), 256)
        assert callable(prep) and callable(search) and isinstance(name, str)
        assert isinstance(exact, bool) and isinstance(params, dict)
    assert build("pallas_fused_bucket", 5, np.float32, a, (16, 128, 128), 256)[3] is False


def _measurement(res, algo="jit_topk"):
    return Measurement(algo, algo, True, {}, 5, 8, 100, 4, 8, [8], res, 0.0, 1, "h", 1.0, 8,
                       None, np.zeros((8, 4), np.float32), np.zeros((100, 4), np.float32))


def test_timing_metrics_use_median_pass_and_keep_best_time():
    wall = {"times_s": [0.4, 0.1, 0.2, 0.3, 0.5]}
    loop = {"times_s": [0.04, 0.01, 0.02, 0.03, 0.05], "latency_ms": {"p50": 1.0}}
    tm = emit.timing_metrics({"wall": wall, "loop": loop}, 8)
    assert tm["best_search_time_s_wall"] == 0.1 and tm["best_search_time_s_loop"] == 0.01
    assert tm["qps_wall"] == pytest.approx(8 / 0.3)  # median of the per-pass QPS
    assert tm["qps_loop"] == pytest.approx(8 / 0.03)
    assert tm["qps_loop"] == timing.median_qps(loop["times_s"], 8)


def test_timing_metrics_numpy_ref_has_wall_only():
    tm = emit.timing_metrics({"times_s": [0.2, 0.1, 0.3, 0.2, 0.2]}, 4)
    assert tm["qps_wall"] == pytest.approx(4 / 0.2) and tm["qps_loop"] is None
    assert tm["serial_latency_ms"] is None and tm["device_time_ms"] is None


def test_roofline_fields_none_for_numpy_ref():
    a = parse([])
    meas = _measurement({"times_s": [0.1]}, algo="numpy_ref")
    assert emit._roofline_fields(meas, a, "cpu") == (None, None, "unknown")
