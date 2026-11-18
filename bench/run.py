"""CLI sweep: runs algos over shapes, writes results/<platform>/synthetic/k<k>/<algo>/<params>.json."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp

from trbench import data, env, records, reference, timing
from trbench.jit_topk import make_jit_topk

DTYPES = {"bf16": jnp.bfloat16, "f32": jnp.float32}
REF_LIMIT = 4e8  # M*N*D above this skips the numpy oracle recall check


def ints(s):
    return [int(x) for x in s.split(",")]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algos", default="jit_topk", help="comma list: numpy_ref,jit_topk")
    p.add_argument("--m", type=ints, default=[16])
    p.add_argument("--n", type=ints, default=[65536])
    p.add_argument("--d", type=ints, default=[128])
    p.add_argument("--k", type=ints, default=[10])
    p.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    p.add_argument("--metric", default="ip", choices=list(reference.METRICS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--out", default="results")
    p.add_argument("--dump-hlo", action="store_true")
    return p.parse_args(argv)


def run_one(algo, m, n, d, k, a, hw, git):
    q, e, _ = data.synthetic(m, n, d, seed=a.seed)
    dtype = DTYPES[a.dtype]
    ref_scores = None
    if m * n * d <= REF_LIMIT:
        ref_scores, _ = reference.numpy_ref(q, e, k, metric=a.metric)
    hlo = None
    if algo == "numpy_ref":
        import time

        times = []
        for _ in range(a.warmup + a.reps):
            t0 = time.perf_counter()
            got, _ = reference.numpy_ref(q, e, k, metric=a.metric)
            times.append(time.perf_counter() - t0)
        times = times[a.warmup :]
        res = {"compile_time_s": 0.0, "times_s": times, "latency_ms": timing.summarize(times)}
        got_scores, exact, name = got, True, "numpy_ref"
    elif algo == "jit_topk":
        qd, ed = jax.device_put(q), jax.device_put(e)  # db resident on device
        f = make_jit_topk(k, metric=a.metric, compute_dtype=dtype)
        res = timing.time_fn(f, qd, ed, warmup=a.warmup, repetitions=a.reps)
        got_scores = jax.device_get(res["compiled"](qd, ed)[0])
        exact, name = True, "jit_matmul_topk"
        if a.dump_hlo:
            hlo = res["compiled"].as_text()
    else:
        raise SystemExit(f"unknown algo {algo}")
    recall = data.recall_at_k(ref_scores, got_scores, eps=1e-2 if a.dtype == "bf16" else 1e-4) if ref_scores is not None else None
    best = min(res["times_s"])
    rec = records.build_record(
        dataset={
            "name": "synthetic", "source_url": None, "n_train": n, "n_test": m, "dim": d,
            "distance": a.metric, "dtype_stored": "float32", "dtype_compute": a.dtype,
            "seed": a.seed, "hash": data.data_hash(q, e),
        },
        algorithm={"name": name, "params": {"precision": a.dtype}, "exact": exact},
        query={"k": k, "batch_size": m, "mode": "batch", "n_queries_timed": m,
               "warmup_iters": a.warmup, "repetitions": a.reps},
        hardware=hw,
        metrics={
            "recall": recall,
            "qps": m / best,
            "best_search_time_s": best,
            "search_time_s_all": res["times_s"],
            "latency_ms_per_batch": res["latency_ms"],
            "latency_ms_per_query_amortized": res["latency_ms"]["p50"] / m,
            "compile_time_s": res["compile_time_s"],
            "index_size_bytes": int(e.nbytes),
            "score_matrix_bytes": 4 * m * n if algo == "jit_topk" else None,
        },
        git=git,
        timing_extra={"xla_flags": __import__("os").environ.get("XLA_FLAGS", "")},
    )
    params = f"m{m}_n{n}_d{d}_{a.metric}_{a.dtype}"
    path = records.write_record(
        records.result_path(a.out, hw["platform"], "synthetic", k, algo, params), rec
    )
    if hlo is not None:
        (path.parent / f"{params}.hlo.txt").write_text(hlo)
    return path, rec


def main(argv=None):
    a = parse(argv)
    hw, git = env.hardware_info(), env.git_info(Path(__file__).resolve().parents[1])
    for algo, m, n, d, k in itertools.product(
        a.algos.split(","), a.m, a.n, a.d, a.k
    ):
        path, rec = run_one(algo, m, n, d, k, a, hw, git)
        mt = rec["metrics"]
        print(
            f"{path}\n  algo={algo} compile_time_s={mt['compile_time_s']:.4f} "
            f"warm_p50_ms={mt['latency_ms_per_batch']['p50']:.4f} recall={mt['recall']} "
            f"flops={mt['flops_achieved']:.3e} pct_peak={mt['percent_of_peak']}"
        )


if __name__ == "__main__":
    main()
