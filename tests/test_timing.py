import json
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bench import run

from trbench import timing


def _search(k=5):
    return jax.jit(lambda q, e: jax.lax.top_k(q @ e.T, k))


def _args(m=64, n=8192, d=128):
    rng = np.random.default_rng(0)
    return (
        jnp.asarray(rng.standard_normal((m, d), dtype=np.float32)),
        jnp.asarray(rng.standard_normal((n, d), dtype=np.float32)),
    )


def test_loop_mode_finite_positive():
    res = timing.time_loop(_search(), *_args(), inner=4)
    assert res["inner"] == 4 and len(res["times_s"]) == timing.DEFAULT_REPS
    assert all(np.isfinite(t) and t > 0 for t in res["times_s"])
    assert res["latency_ms"]["p50"] > 0


def test_loop_not_optimized_away():
    """Per-dispatch time grows with inner; a hoisted/CSE'd body would stay flat."""
    fn, args = _search(), _args(m=128, n=32768, d=128)

    def per_dispatch(inner):
        return min(timing.time_loop(fn, *args, inner=inner)["times_s"]) * inner

    t_small, t_big = per_dispatch(2), per_dispatch(32)
    assert t_big > 3 * t_small, (t_small, t_big)


def test_loop_hlo_keeps_loop_and_work():
    args = _args(m=8, n=512, d=16)
    txt = timing.make_loop_fn(_search(), 8).lower(*args).compile().as_text()
    assert "while" in txt and ("dot" in txt or "convolution" in txt or "ynn" in txt)


def test_loop_rejects_bad_inner():
    with pytest.raises(ValueError):
        timing.make_loop_fn(_search(), 0)


def test_serial_count_and_percentiles():
    e = _args()[1]
    q = _args(m=10)[0]
    calls = []

    def make_args(i):
        calls.append(i)
        return (q[i % 10 : i % 10 + 1], e)

    res = timing.time_serial(_search(), make_args, 25)
    assert len(res["times_s"]) == 25
    lat = res["latency_ms"]
    assert 0 < lat["min"] <= lat["p50"] <= lat["p95"] <= lat["p99"]


def test_device_time_from_profile_fake_planes():
    ev = lambda ns: NS(duration_ns=ns)
    plane = NS(
        name="/device:TPU:0",
        lines=[
            NS(name="XLA Ops", events=[ev(9e6)]),
            NS(name="XLA Modules", events=[ev(1e6), ev(3e6)]),
        ],
    )
    host = NS(name="/host:CPU", lines=[NS(name="XLA Modules", events=[ev(5e6)])])
    assert timing.device_time_from_profile(NS(planes=[host, plane])) == (4.0, 2)
    assert timing.device_time_from_profile(NS(planes=[host])) == (None, 0)


def test_device_time_cpu_is_null_with_note():
    out = timing.device_time(_search(), *_args(m=8, n=512, d=16), calls=2)
    assert out["device_time_ms"] is None and out["device_time_note"]


def _run(tmp_path, *extra):
    run.main(
        [
            "--algos",
            "jit_topk",
            "--m",
            "8",
            "--n",
            "2048",
            "--d",
            "16",
            "--k",
            "5",
            "--out",
            str(tmp_path),
            *extra,
        ]
    )
    (path,) = tmp_path.rglob("*.json")
    return json.loads(path.read_text())


def test_cli_wall_default_records_mode(tmp_path):
    r = _run(tmp_path)
    mt = r["metrics"]
    assert r["timing"]["mode"] == "wall" and r["timing"]["headline"] == "wall"
    assert mt["best_search_time_s_wall"] == mt["best_search_time_s"]
    assert mt["best_search_time_s_loop"] is None and mt["serial_latency_ms"] is None


def test_cli_both_serial_device(tmp_path):
    r = _run(tmp_path, "--timing", "both", "--loop-iters", "4", "--serial", "12", "--device-time")
    mt, tm = r["metrics"], r["timing"]
    assert tm["mode"] == "both" and tm["headline"] == "wall" and tm["loop_iters"] == 4
    assert mt["best_search_time_s"] == mt["best_search_time_s_wall"] > 0
    assert mt["best_search_time_s_loop"] > 0 and mt["qps_loop"] > 0
    assert mt["serial_n"] == 12 and mt["serial_latency_ms"]["p99"] > 0
    assert mt["recall"] >= 0.99
    assert mt["device_time_ms"] is None and mt["device_time_note"]  # CPU


def test_cli_loop_headline(tmp_path):
    r = _run(tmp_path, "--timing", "loop", "--loop-iters", "4")
    mt = r["metrics"]
    assert r["timing"]["headline"] == "loop"
    assert mt["best_search_time_s"] == mt["best_search_time_s_loop"]
    assert mt["best_search_time_s_wall"] is None and mt["recall"] >= 0.99


def test_bootstrap_ci_deterministic_and_brackets_median():
    x = np.random.default_rng(1).normal(10.0, 1.0, 30)
    lo, hi = timing.bootstrap_ci(x)
    assert (lo, hi) == timing.bootstrap_ci(x)  # fixed seed
    assert lo < np.median(x) < hi
    assert timing.bootstrap_ci(x, np.mean, seed=1) != timing.bootstrap_ci(x, np.mean, seed=2)
    assert timing.bootstrap_ci([5.0, 5.0, 5.0]) == (5.0, 5.0)
    with pytest.raises(ValueError):
        timing.bootstrap_ci([])


def test_bootstrap_ci_narrows_with_more_samples():
    rng = np.random.default_rng(2)
    w = [np.subtract(*timing.bootstrap_ci(rng.normal(10, 1, n))[::-1]) for n in (10, 400)]
    assert w[1] < w[0]


def test_summarize_has_spread_and_old_keys():
    s = timing.summarize([0.001, 0.002, 0.0015, 0.0012, 0.0011])
    assert {"min", "p50", "p95", "p99", "mean", "stdev", "p50_ci95"} <= s.keys()
    assert s["stdev"] > 0 and s["p50_ci95"][0] <= s["p50_ci95"][1]
    assert timing.summarize([0.001])["stdev"] == 0.0


def test_spread_metrics_fields():
    times = [0.010, 0.011, 0.0105, 0.012, 0.0102, 0.0108]
    sp = timing.spread_metrics(times, 100, timing.summarize(times))
    assert sp["n_repetitions"] == 6
    lo, hi = sp["qps_ci95"]
    assert lo <= sp["qps_median"] <= hi and sp["qps_stdev"] > 0
    assert sp["bootstrap"]["seed"] == timing.BOOT_SEED


def test_default_reps_is_20_and_minimum_enforced():
    assert timing.DEFAULT_REPS == 20 and timing.MIN_REPS == 5
    with pytest.raises(ValueError):
        timing.time_fn(_search(), *_args(), repetitions=4)
