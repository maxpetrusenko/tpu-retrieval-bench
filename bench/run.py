"""CLI sweep: runs algos over shapes, writes results/<platform>/<dataset>/k<k>/<algo>/<params>.json.

--dataset synthetic (default) sweeps --m/--n/--d. An ann-benchmarks name takes shapes and metric
from the HDF5 file (--n-train/--n-test subset it); recall uses the file ground truth.
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from trbench import data, datasets, env, records, reference, roofline, timing
from trbench.approx import make_approx_parts
from trbench.blockwise import make_blockwise_parts, make_score_only_parts, make_select_only
from trbench.jit_topk import make_jit_topk_parts
from trbench.pallas_fused import LANES, make_fused_parts
from trbench.pallas_score import make_pallas_parts, resolve_blocks

DTYPES = {"bf16": jnp.bfloat16, "f32": jnp.float32}
REF_LIMIT = 4e8  # M*N*D above this skips the numpy oracle recall check


def ints(s):
    return [int(x) for x in s.split(",")]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algos", default="jit_topk", help="comma list: numpy_ref,jit_topk,blockwise,pallas_score,"
                   "pallas_fused_topk,pallas_fused_bucket,approx_max_k,score_only,select_only")
    p.add_argument("--kprime", type=int, default=None,
                   help="pallas_fused_topk: per-tile candidates (default k = exact; < k approximate)")
    p.add_argument("--lanes", type=int, default=LANES, help="pallas_fused_bucket: bins per tile")
    p.add_argument("--dataset", default="synthetic",
                   choices=["synthetic", *datasets.DATASETS], help="synthetic or ann-benchmarks")
    p.add_argument("--data-dir", default=None, help="dir with <dataset>.hdf5 (default repo data/)")
    p.add_argument("--n-train", type=int, default=None, help="dataset: use first N train rows")
    p.add_argument("--n-test", type=int, default=None, help="dataset: use first N test rows")
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
    p.add_argument("--block-size", type=int, default=65536, help="blockwise db block")
    p.add_argument("--bm", type=int, default=256)
    p.add_argument("--bn", type=int, default=512)
    p.add_argument("--bk", type=int, default=512)
    p.add_argument("--chunk", type=int, default=65536, help="pallas_score selection chunk")
    p.add_argument("--recall-target", type=float, default=0.95)
    p.add_argument("--sweep-blocks", action="store_true",
                   help="pallas_score / pallas_fused_*: sweep bm/bn/bk grid, one JSON each")
    p.add_argument("--batch-size", type=int, default=1024,
                   help="queries per search call; a pass chunks M queries into batches so "
                   "scores stay <= ~4 GB. QPS = all queries / pass time")
    p.add_argument("--timing", default="wall", choices=["wall", "loop", "both"],
                   help="wall = per-call block_until_ready (headline); loop = in-jit fori_loop "
                   "repeat, device-amortized per iteration (headline); both = wall headline, "
                   "loop recorded next to it")
    p.add_argument("--loop-iters", type=int, default=16,
                   help="iterations per dispatch in loop mode")
    p.add_argument("--serial", type=int, default=0,
                   help="also time N queries one at a time (batch 1, concurrency 1); records "
                   "p50/p95/p99 per query")
    p.add_argument("--device-time", action="store_true",
                   help="trace a few warm calls with jax.profiler and record device time "
                   "(null off-TPU/GPU)")
    p.add_argument("--dump-hlo", action="store_true")
    return p.parse_args(argv)


MATERIALIZING = roofline.MATERIALIZING | {"score_only", "select_only"}
SWEEP = {"bm": [128, 256, 512], "bn": [256, 512, 1024], "bk": [128, 256, 512]}
EXACT_ALGOS = ("jit_topk", "blockwise", "pallas_score", "pallas_fused_topk")
FUSED = ("pallas_fused_topk", "pallas_fused_bucket")
PALLAS_ALGOS = ("pallas_score", *FUSED)
SAMPLE_QUERIES = 64  # oracle queries when the full numpy oracle is too big


# Each factory returns (prepare, search, name, exact, params). prepare(db) -> index is untimed.
def _jit(k, dtype, a, bp):
    p, s = make_jit_topk_parts(k, metric=a.metric, compute_dtype=dtype)
    return p, s, "jit_matmul_topk", True, {}


def _blockwise(k, dtype, a, bp):
    p, s = make_blockwise_parts(k, metric=a.metric, compute_dtype=dtype,
                                block_size=a.block_size)
    return p, s, "blockwise_topk", True, {"block_size": a.block_size}


def _pallas(k, dtype, a, bp):
    bm, bn, bk = bp
    p, s = make_pallas_parts(k, metric=a.metric, compute_dtype=dtype, bm=bm, bn=bn, bk=bk,
                             chunk=a.chunk)
    return p, s, "pallas_score_topk", True, {"bm": bm, "bn": bn, "bk": bk, "chunk": a.chunk,
                                             "interpret": jax.default_backend() != "tpu"}


def _fused(mode):
    def factory(k, dtype, a, bp, n=None):
        bm, bn, bk = bp
        kprime = a.kprime or k
        p, s = make_fused_parts(k, n, metric=a.metric, compute_dtype=dtype, bm=bm, bn=bn, bk=bk,
                                chunk=a.chunk, kprime=kprime, mode=mode, lanes=a.lanes)
        params = {"bm": bm, "bn": bn, "bk": bk, "chunk": a.chunk,
                  "interpret": jax.default_backend() != "tpu"}
        params.update({"kprime": kprime} if mode == "topk" else {"lanes": a.lanes})
        return p, s, f"pallas_fused_{mode}", mode == "topk" and kprime == k, params
    return factory


def _approx(k, dtype, a, bp):
    p, s = make_approx_parts(k, recall_target=a.recall_target, metric=a.metric,
                             compute_dtype=dtype)
    return p, s, "approx_max_k", False, {"recall_target": a.recall_target}


def _score_only(k, dtype, a, bp):
    p, s = make_score_only_parts(metric=a.metric, compute_dtype=dtype)
    return p, s, "score_only", True, {}


def _select_only(k, dtype, a, bp):
    p, s = make_score_only_parts(metric=a.metric, compute_dtype=dtype)
    return p, make_select_only(k), "select_only", True, {"score_fn": s}


FACTORIES = {"jit_topk": _jit, "blockwise": _blockwise, "pallas_score": _pallas,
             "pallas_fused_topk": _fused("topk"), "pallas_fused_bucket": _fused("bucket"),
             "approx_max_k": _approx, "score_only": _score_only, "select_only": _select_only}


def build(algo, k, dtype, a, bp, n):
    """Factory call; fused factories need the db size (static in the jitted search)."""
    return FACTORIES[algo](k, dtype, a, bp, n) if algo in FUSED else FACTORIES[algo](
        k, dtype, a, bp)


def get_inputs(a, m, n, d, ann=None):
    """(queries, db, truth, dataset_hash). truth = ann dict or None (synthetic)."""
    if ann is not None:
        return ann["test"], ann["train"], ann, ann["hash"]
    q, e, _ = data.synthetic(m, n, d, seed=a.seed)
    return q, e, None, data.data_hash(q, e)


def compute_recall(a, k, q, e, ann, got_scores):
    """(recall, n_queries_used, recall_epsilon_0.01). Dataset: knn_recall vs file truth, eps 1e-3
    (`recall`) and 1e-2 (`recall_epsilon_0.01`, ann-benchmarks "epsilon" metric). bf16 scoring
    perturbs distances by ~1e-3, so exact bf16 variants can fall below 1.0 at eps 1e-3 only.
    Synthetic: threshold recall vs the numpy oracle (eps 1e-2 bf16 / 1e-4 f32), on a sampled
    subset of queries when M*N*D > REF_LIMIT; no second recall."""
    if got_scores is None:
        return None, None, None
    m = q.shape[0]
    if ann is not None:
        dist = datasets.scores_to_distances(got_scores, a.metric, q)
        return (datasets.knn_recall(dist, ann["distances"], k), m,
                datasets.knn_recall(dist, ann["distances"], k, eps=1e-2))
    idx = np.arange(m)
    if m * e.shape[0] * e.shape[1] > REF_LIMIT:
        idx = np.unique(np.linspace(0, m - 1, min(m, SAMPLE_QUERIES)).astype(int))
    ref_scores, _ = reference.numpy_ref(q[idx], e, k, metric=a.metric)
    rec = data.recall_at_k(ref_scores, np.asarray(got_scores)[idx],
                           eps=1e-2 if a.dtype == "bf16" else 1e-4)
    return rec, len(idx), None


def params_str(algo, m, n, d, a, bp):
    params = f"m{m}_n{n}_d{d}_{a.metric}_{a.dtype}"
    if algo == "blockwise":
        params += f"_B{a.block_size}"
    elif algo in PALLAS_ALGOS:
        params += f"_bm{bp[0]}_bn{bp[1]}_bk{bp[2]}"
        if a.sweep_blocks:
            params += "_sweep"  # keep sweep records apart from default-config records
    elif algo == "approx_max_k":
        params += f"_r{a.recall_target}"
    return params


def time_numpy(q, e, k, a):
    times = []
    for _ in range(a.warmup + a.reps):
        t0 = time.perf_counter()
        got, _ = reference.numpy_ref(q, e, k, metric=a.metric)
        times.append(time.perf_counter() - t0)
    times = times[a.warmup :]
    res = {"compile_time_s": 0.0, "times_s": times, "latency_ms": timing.summarize(times)}
    if a.serial:
        st = []
        for i in range(a.serial):
            t0 = time.perf_counter()
            reference.numpy_ref(q[i % len(q) : i % len(q) + 1], e, k, metric=a.metric)
            st.append(time.perf_counter() - t0)
        res["serial"] = {"times_s": st, "latency_ms": timing.summarize(st)}
    return res, got


def time_jax(algo, q, e, k, a, bp):
    """Prepare once (untimed index build, timed separately), then time batched search."""
    dtype = DTYPES[a.dtype]
    m = q.shape[0]
    bs = min(a.batch_size, m)
    qd, ed = jax.device_put(q), jax.device_put(e)  # db resident on device
    prep_fn, search_fn, name, exact, params = build(algo, k, dtype, a, bp, e.shape[0])
    jax.block_until_ready(prep_fn(ed))  # compile + run once
    t0 = time.perf_counter()
    index = prep_fn(ed)
    jax.block_until_ready(index)
    prepare_s = time.perf_counter() - t0
    qb = [qd[i : i + bs] for i in range(0, m, bs)]
    score_fn = params.pop("score_fn", None)
    if score_fn is not None:  # select_only: scores built untimed per batch
        def make_args(i):
            return (score_fn(qb[i], index),)
    else:
        def make_args(i):
            return (qb[i], index)
    keep = algo not in STAGE_ONLY
    wall = loop = None
    if a.timing in ("wall", "both"):
        wall = timing.time_batched(search_fn, make_args, len(qb), warmup=a.warmup,
                                   repetitions=a.reps, keep_out=keep)
    if a.timing in ("loop", "both"):
        loop = timing.time_batched_loop(search_fn, make_args, len(qb), inner=a.loop_iters,
                                        warmup=a.warmup, repetitions=a.reps)
    if wall is not None:
        res = dict(wall)
    else:  # loop headline: the loop returns a checksum, so outputs come from one untimed pass
        outs = [search_fn(*make_args(i)) for i in range(len(qb))] if keep else None
        res = {**loop, "outs": outs}
    res["wall"], res["loop"] = wall, loop
    if a.serial:
        def serial_args(i):
            q1 = qd[i % m : i % m + 1]
            return (score_fn(q1, index),) if score_fn is not None else (q1, index)
        res["serial"] = timing.time_serial(search_fn, serial_args, a.serial, warmup=a.warmup)
    if a.device_time:
        res["device"] = timing.device_time(search_fn, *make_args(0), calls=a.reps)
    got = None
    if res["outs"] is not None:
        got = np.concatenate([jax.device_get(o[0]) for o in res["outs"]], axis=0)
    size = sum(x.nbytes for x in jax.tree_util.tree_leaves(index))
    return res, got, name, exact, params, prepare_s, int(size), [len(x) for x in qb]


STAGE_ONLY = ("score_only", "select_only")


def run_one(algo, m, n, d, k, a, hw, git, bp=None, ann=None):
    q, e, truth, dhash = get_inputs(a, m, n, d, ann)
    m, n = q.shape[0], e.shape[0]
    bs = min(a.batch_size, m)
    sizes, prepare_s, size = [m], 0.0, int(e.nbytes)
    if algo == "numpy_ref":
        res, got_scores = time_numpy(q, e, k, a)
        name, exact, params = "numpy_ref", True, {}
    elif algo in FACTORIES:
        res, got_scores, name, exact, params, prepare_s, size, sizes = time_jax(
            algo, q, e, k, a, bp)
    else:
        raise SystemExit(f"unknown algo {algo}")
    recall, n_recall_q, recall_eps2 = compute_recall(a, k, q, e, truth, got_scores)
    best = min(res["times_s"])
    mode = "wall" if algo == "numpy_ref" else a.timing
    tm = timing_metrics(res, m)
    rf_algo = "pallas_fused" if algo in FUSED else algo
    in_roofline = rf_algo in roofline.VARIANTS
    dkind = hw["device_kind"]
    # db is stored in compute dtype after prepare; queries arrive as f32
    dt = {"q_dtype": "f32", "db_dtype": a.dtype}
    if algo in FUSED:  # candidates written per db tile: M * n_tiles * round_up(k', 128) * 8
        bn = resolve_blocks(bs, n, d, bp[0], bp[1], bp[2], a.chunk)[1]
        dt.update(n_tiles=-(-n // bn), kprime=params.get("kprime", params.get("lanes")))
    rf_bytes = sum(roofline.bytes_moved(rf_algo, s, n, d, k, a.dtype, **dt) for s in sizes
                   ) if in_roofline else None
    rf_ai = roofline.arithmetic_intensity(rf_algo, bs, n, d, k, a.dtype, **dt
                                          ) if in_roofline else None
    rf_label = roofline.label(rf_algo, bs, n, d, k, a.dtype, dkind, **dt
                              ) if in_roofline else "unknown"
    has_matmul = algo != "select_only"
    rec = records.build_record(
        dataset={
            "name": a.dataset, "source_url": None, "n_train": n, "n_test": m, "dim": d,
            "distance": a.metric, "dtype_stored": a.dtype if algo in FACTORIES else "float32",
            "dtype_compute": a.dtype, "seed": a.seed, "hash": dhash,
        },
        algorithm={"name": name, "params": {"precision": a.dtype, **params}, "exact": exact},
        query={"k": k, "batch_size": bs, "n_batches": len(sizes), "mode": "batch",
               "n_queries_timed": m, "warmup_iters": a.warmup, "repetitions": a.reps},
        hardware=hw,
        metrics={
            "recall": recall,
            "recall_sampled_queries": n_recall_q,
            "recall_epsilon_0.01": recall_eps2,
            "recall_compute_dtype": a.dtype,
            "qps": m / best,
            "best_search_time_s": best,
            "search_time_s_all": res["times_s"],
            "latency_ms_per_batch": res["latency_ms"],
            "latency_ms_per_query_amortized": best * 1e3 / m,
            "compile_time_s": res["compile_time_s"],
            "build_time_s": prepare_s,
            "index_size_bytes": size,
            "score_matrix_bytes": 4 * bs * n if algo in MATERIALIZING else None,
            "bytes_moved": rf_bytes,
            "arithmetic_intensity": rf_ai,
            "roofline_label": rf_label,
            "selection_time_ms": best * 1e3 if algo == "select_only" else None,
            "matmul_time_ms": best * 1e3 if algo == "score_only" else None,
            "selection_over_matmul": None,
            **tm,
        },
        git=git,
        timing_extra={
            "xla_flags": os.environ.get("XLA_FLAGS", ""),
            "mode": mode,
            "headline": "loop" if mode == "loop" else "wall",
            "loop_iters": a.loop_iters if mode in ("loop", "both") else None,
            "loop_checksum": timing.LOOP_CHECKSUM if mode in ("loop", "both") else None,
            "serial_n": a.serial or None,
            "device_time": bool(a.device_time),
        },
        matmul=has_matmul,
        compute_dtype=a.dtype,
    )
    path = records.write_record(
        records.result_path(a.out, hw["platform"], a.dataset, k, algo,
                            params_str(algo, m, n, d, a, bp)), rec)
    if a.dump_hlo and algo in FACTORIES:
        hlo = compile_hlo(algo, q, e, k, a, bp)
        (path.parent / f"{params_str(algo, m, n, d, a, bp)}.hlo.txt").write_text(hlo)
    return path, rec


def timing_metrics(res, m):
    """Record fields for the wall / loop / serial / device-time lanes (None when not run)."""
    wall, loop = res.get("wall"), res.get("loop")
    if wall is None and loop is None:  # numpy_ref: wall only
        wall = res
    bw = min(wall["times_s"]) if wall else None
    bl = min(loop["times_s"]) if loop else None
    ser, dev = res.get("serial"), res.get("device") or {}
    sl = ser["latency_ms"] if ser else None
    return {
        "best_search_time_s_wall": bw,
        "best_search_time_s_loop": bl,
        "qps_wall": m / bw if bw else None,
        "qps_loop": m / bl if bl else None,
        "search_time_s_all_loop": loop["times_s"] if loop else None,
        "latency_ms_per_batch_loop": loop["latency_ms"] if loop else None,
        "serial_latency_ms": sl,
        "serial_n": len(ser["times_s"]) if ser else None,
        "serial_qps": 1e3 / sl["mean"] if sl else None,
        "device_time_ms": dev.get("device_time_ms"),
        "device_time_source": dev.get("device_time_source"),
        "device_time_note": dev.get("device_time_note"),
    }


def compile_hlo(algo, q, e, k, a, bp):
    """HLO text of the search step for the first batch."""
    prep_fn, search_fn, *_ = build(algo, k, DTYPES[a.dtype], a, bp, e.shape[0])
    index = prep_fn(jax.device_put(e))
    qb = jax.device_put(q[: min(a.batch_size, q.shape[0])])
    if algo == "select_only":
        _, _, _, _, params = build(algo, k, DTYPES[a.dtype], a, bp, e.shape[0])
        return search_fn.lower(params["score_fn"](qb, index)).compile().as_text()
    return search_fn.lower(qb, index).compile().as_text()


def write_error(a, algo, m, n, d, k, bp, hw, exc):
    path = records.result_path(a.out, hw["platform"], a.dataset, k, algo,
                               params_str(algo, m, n, d, a, bp))
    return records.write_record(path, {
        "error": f"{type(exc).__name__}: {str(exc)[:500]}", "algo": algo,
        "shape": {"m": m, "n": n, "d": d, "k": k}, "batch_size": min(a.batch_size, m),
    })


def is_oom(exc):
    return (type(exc).__name__ == "XlaRuntimeError" or "RESOURCE_EXHAUSTED" in str(exc)
            or isinstance(exc, MemoryError))


def load_dataset(a):
    """Load the ann-benchmarks file once; metric comes from the dataset, not --metric."""
    name = a.dataset
    ann = datasets.load_ann(
        Path(a.data_dir) / f"{name}.hdf5" if a.data_dir else name,
        n_train=a.n_train, n_test=a.n_test,
    )
    ann["hash"] = data.data_hash(ann["train"], ann["test"])
    a.metric = ann["metric"]
    return ann


def link_stages(stage, exact, a, dkind):
    """R5: when score_only and select_only ran on the same shape, write the ratio into both.

    select_only is labelled from the ratio; score_only never is (it has no selection). Exact
    full-search variants on the same shape are relabelled VPU/COP-bound when the ratio is > 1.
    """
    dt = {"q_dtype": "f32", "db_dtype": a.dtype}
    for key, recs in stage.items():
        if "score_only" not in recs or "select_only" not in recs:
            continue
        (sp, sc), (lp, se) = recs["score_only"], recs["select_only"]
        t_mm = sc["metrics"]["best_search_time_s"]
        t_sel = se["metrics"]["best_search_time_s"]
        ratio = t_sel / t_mm
        for path, rec in ((sp, sc), (lp, se)):
            mt = rec["metrics"]
            mt["selection_time_ms"], mt["matmul_time_ms"] = t_sel * 1e3, t_mm * 1e3
            mt["selection_over_matmul"] = ratio
        ds = se["dataset"]
        se["metrics"]["roofline_label"] = roofline.label(
            "select_only", se["query"]["batch_size"], ds["n_train"], ds["dim"], key[3], a.dtype,
            dkind, select_time_s=t_sel, matmul_time_s=t_mm, **dt)
        records.write_record(sp, sc)
        records.write_record(lp, se)
        if ratio > 1:
            for path, rec in exact.get(key, []):
                rec["metrics"]["roofline_label"] = "VPU/COP-bound"
                records.write_record(path, rec)


def main(argv=None):
    a = parse(argv)
    hw, git = env.hardware_info(), env.git_info(Path(__file__).resolve().parents[1])
    ann = load_dataset(a) if a.dataset != "synthetic" else None
    shapes = [(len(ann["test"]), len(ann["train"]), ann["train"].shape[1])] if ann else None
    stage, exact = {}, {}
    for algo, k in itertools.product(a.algos.split(","), a.k):
        for m, n, d in shapes or itertools.product(a.m, a.n, a.d):
            for bp in block_params(algo, a, min(a.batch_size, m), n, d):
                try:
                    path, rec = run_one(algo, m, n, d, k, a, hw, git, bp, ann)
                except Exception as exc:
                    if not is_oom(exc):
                        raise
                    path = write_error(a, algo, m, n, d, k, bp, hw, exc)
                    print(f"{path}\n  algo={algo} error={type(exc).__name__} (continuing)")
                    continue
                report(algo, path, rec)
                if algo in STAGE_ONLY:
                    stage.setdefault((m, n, d, k), {})[algo] = (path, rec)
                elif algo in EXACT_ALGOS:
                    exact.setdefault((m, n, d, k), []).append((path, rec))
    link_stages(stage, exact, a, hw["device_kind"])
    for recs in stage.values():
        if len(recs) == 2:
            r = recs["select_only"][1]["metrics"]["selection_over_matmul"]
            print(f"selection_over_matmul={r:.3f}")


def block_params(algo, a, bm_cap, n, d):
    """Block configs to run. Sweep configs are deduped after clamping to the shape."""
    if algo not in PALLAS_ALGOS:
        return [None]
    grid = [(a.bm, a.bn, a.bk)]
    if a.sweep_blocks:
        grid = itertools.product(SWEEP["bm"], SWEEP["bn"], SWEEP["bk"])
    out = []
    for bm, bn, bk in grid:
        cl = resolve_blocks(bm_cap, n, d, bm, bn, bk, a.chunk)[:3]
        if cl not in out:
            out.append(cl)
    return out


def report(algo, path, rec):
    mt = rec["metrics"]
    fl = mt["flops_achieved"]
    print(
        f"{path}\n  algo={algo} compile_time_s={mt['compile_time_s']:.4f} "
        f"warm_p50_ms={mt['latency_ms_per_batch']['p50']:.4f} recall={mt['recall']} "
        f"flops={'null' if fl is None else format(fl, '.3e')} pct_peak={mt['percent_of_peak']} "
        f"roofline={mt['roofline_label']}"
    )


if __name__ == "__main__":
    main()
